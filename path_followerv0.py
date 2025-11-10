#!/usr/bin/env -S python3
# -*- coding: utf-8 -*-

import math, argparse, curses, time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time as RTime
from rclpy.logging import set_logger_level, LoggingSeverity

from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import Buffer, TransformListener


def yaw(q):
    return math.atan2(2*(q.w*q.z), 1 - 2*(q.z*q.z))

def clamp(v,a,b):
    return max(a,min(b,v))

def ang_diff(a,b):
    return (a-b+math.pi)%(2*math.pi)-math.pi


class MarkerFollower(Node):
    def __init__(self):
        super().__init__('marker_follower')

        # ---- Parámetros de operación ----
        self.declare_parameter('cmd_vel_topic','/cmd_vel')
        self.declare_parameter('pose_source','tf_map')   # tf_map | amcl
        self.declare_parameter('amcl_topic','/amcl_pose')
        self.declare_parameter('pose_stale_ms', 600)
        self.declare_parameter('rate_hz',30.0)

        # Control básico + suavizado
        self.declare_parameter('k_yaw',0.2)
        self.declare_parameter('k_dist',1.0)
        self.declare_parameter('vmax',0.4)
        self.declare_parameter('wmax',1.8)
        self.declare_parameter('yaw_tol_deg',3.0)
        self.declare_parameter('xy_tol',0.05)
        self.declare_parameter('lookahead_m',0.30)
        self.declare_parameter('accel_v',0.8)   # m/s^2
        self.declare_parameter('accel_w',3.0)   # rad/s^2

        # Paradas explícitas
        self.declare_parameter('stop_dwell_s', 0.10)   # pausa mínima detenido
        self.declare_parameter('stop_eps', 0.02)       # umbral de “0”

        # Logging
        self.declare_parameter('log_level','warn') # debug|info|warn|error

        # Lee parámetros
        self.topic = self.get_parameter('cmd_vel_topic').value
        self.pose_source = self.get_parameter('pose_source').value
        self.amcl_topic = self.get_parameter('amcl_topic').value
        self.pose_stale = float(self.get_parameter('pose_stale_ms').value)/1000.0
        self.dt = 1.0/float(self.get_parameter('rate_hz').value)

        self.k_yaw = float(self.get_parameter('k_yaw').value)
        self.k_dist = float(self.get_parameter('k_dist').value)
        self.vmax = float(self.get_parameter('vmax').value)
        self.wmax = float(self.get_parameter('wmax').value)
        self.yaw_tol = math.radians(float(self.get_parameter('yaw_tol_deg').value))
        self.xy_tol = float(self.get_parameter('xy_tol').value)
        self.lookahead = float(self.get_parameter('lookahead_m').value)
        self.accel_v = float(self.get_parameter('accel_v').value)
        self.accel_w = float(self.get_parameter('accel_w').value)
        self.stop_dwell = float(self.get_parameter('stop_dwell_s').value)
        self.stop_eps = float(self.get_parameter('stop_eps').value)
        self.stop_t0 = None

        # Histeresis simple (más laxa para salir)
        self.yaw_tol_exit = self.yaw_tol * 1.4
        self.xy_tol_exit  = self.xy_tol  * 1.4

        # Orientación objetivo final (opcional; se toma la tangente del último tramo)
        self.final_yaw = None

        lvl = str(self.get_parameter('log_level').value).lower()
        maplvl = {'debug':LoggingSeverity.DEBUG,'info':LoggingSeverity.INFO,
                  'warn':LoggingSeverity.WARN,'error':LoggingSeverity.ERROR}
        set_logger_level(self.get_name(), maplvl.get(lvl, LoggingSeverity.WARN))

        # ---- TF / Pose ----
        self.tf = Buffer(cache_time=Duration(seconds=5.0))
        self.tfl = TransformListener(self.tf, self, spin_thread=True)
        self.pose = None; self.pose_frame=None; self.pose_stamp=0.0
        self.create_subscription(PoseWithCovarianceStamped, self.amcl_topic, self.on_amcl, 10)

        # ---- Ruta (borrador / congelada / pendiente) ----
        self.frame=None
        self.draft=[]      # lo que llega por markers
        self.route=[]      # congelada al iniciar
        self.pending=None  # (frame, pts)
        self.i = 0
        self.phase='stop_before_turn'  # stop_before_turn|turn|stop_before_go|go
        self.state='idle'              # idle|running|paused|done|aborted

        # Rampas
        self.v_cmd = 0.0
        self.w_cmd = 0.0

        # Depuración (leído por el TUI)
        self.debug = {}

        # IO ROS
        self.sub_m = self.create_subscription(MarkerArray,'/ui/draft_markers',self.on_markers,10)
        self.sub_c = self.create_subscription(String,'/ui/exec_command',self.on_cmd,10)
        self.pub_v = self.create_publisher(Twist,self.topic,10)
        self.pub_s = self.create_publisher(String,'/ui/exec_status',10)
        self.timer = self.create_timer(self.dt, self.step)

    # ---------- Utilidades de ruta ----------
    def _nearest_on_poly(self, x, y, pts):
        best_k=0; best_px, best_py = pts[0]
        best_d2 = float('inf')
        for k in range(len(pts)-1):
            x1,y1=pts[k]; x2,y2=pts[k+1]
            vx,vy=x2-x1,y2-y1; wx,wy=x-x1,y-y1
            L2 = vx*vx+vy*vy or 1e-9
            t = max(0.0, min(1.0, (wx*vx+wy*vy)/L2))
            px,py = x1+t*vx, y1+t*vy
            d2=(x-px)**2+(y-py)**2
            if d2<best_d2:
                best_d2=d2; best_k=k; best_px, best_py = px,py
        return best_k,(best_px,best_py)

    def _lock_route_from_here(self, x, y):
        if len(self.draft)<1:
            return False
        k,(px,py)=self._nearest_on_poly(x,y,self.draft)
        self.route = [(px,py)] + self.draft[k+1:]
        self.i = 0
        self.phase='stop_before_turn'  # forzar parada y giro inicial
        self.stop_t0 = None

        # Orientación final = tangente del último tramo (opcional)
        if len(self.route) >= 2:
            (x1,y1),(x2,y2) = self.route[-2], self.route[-1]
            self.final_yaw = math.atan2(y2-y1, x2-x1)
        else:
            self.final_yaw = None
        return True

    def _lookahead_target(self, x, y):
        if self.i >= len(self.route):
            return self.route[-1]
        j = max(self.i, 0)
        while j < len(self.route)-1:
            if math.hypot(self.route[j][0]-x, self.route[j][1]-y) >= self.lookahead:
                break
            j += 1
        return self.route[j]

    # ---------- Pose ----------
    def on_amcl(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw(p.orientation))
        self.pose_frame = msg.header.frame_id or 'map'
        self.pose_stamp = time.time()

    def get_pose_in_frame(self, target_frame: str):
        if self.pose_source == 'tf_map':
            t0 = RTime(seconds=0, nanoseconds=0)
            if not self.tf.can_transform(target_frame, 'base_link', t0):
                return None
            tfm = self.tf.lookup_transform(target_frame,'base_link', t0)
            x = tfm.transform.translation.x
            y = tfm.transform.translation.y
            th = yaw(tfm.transform.rotation)
            return (x,y,th)

        if self.pose is None or (time.time()-self.pose_stamp) > self.pose_stale:
            return None
        if self.pose_frame == target_frame:
            return self.pose

        t0 = RTime(seconds=0, nanoseconds=0)
        if not self.tf.can_transform(target_frame, self.pose_frame, t0):
            return None
        tfm = self.tf.lookup_transform(target_frame, self.pose_frame, t0)
        tx = tfm.transform.translation.x; ty = tfm.transform.translation.y
        T = yaw(tfm.transform.rotation)
        cx,sx = math.cos(T), math.sin(T)
        x0,y0,th0 = self.pose
        x = cx*x0 - sx*y0 + tx
        y = sx*x0 + cx*y0 + ty
        th = ang_diff(th0 + T, 0.0)
        return (x,y,th)

    # ---------- Callbacks ----------
    def on_markers(self, ma: MarkerArray):
        line = next((m for m in ma.markers if m.type==Marker.LINE_STRIP and m.points), None)
        if not line:
            return
        pts = [(p.x,p.y) for p in line.points]
        f = line.header.frame_id or 'map'
        if self.state == 'running':
            self.pending = (f, pts)
            self.get_logger().info(f"Ruta pendiente recibida ({len(pts)} pts) en '{f}'.")
            return
        self.frame, self.draft = f, pts
        self.i=0; self.phase='stop_before_turn'; self.stop_t0=None
        self.get_logger().info(f"Ruta (borrador) recibida ({len(pts)} pts) en '{f}'.")

    def on_cmd(self, msg: String):
        c = msg.data.strip().lower()
        if c in ('start','resume'):
            if not self.draft:
                self.get_logger().warn("No hay ruta (borrador)."); return
            pose = self.get_pose_in_frame(self.frame or 'map')
            if pose is None:
                self.get_logger().warn("Sin pose válida en el frame de la ruta."); return
            x,y,_=pose
            if not self._lock_route_from_here(x,y):
                self.get_logger().warn("No se pudo congelar la ruta."); return
            self.state='running'; self.publish_status('running')

        elif c=='pause':
            self.state='paused'; self.pub_v.publish(Twist()); self.publish_status('paused')

        elif c=='abort':
            self.state='aborted'; self.pub_v.publish(Twist()); self.publish_status('aborted')

        elif c=='replan':
            if self.pending:
                self.frame, self.draft = self.pending; self.pending=None
                pose = self.get_pose_in_frame(self.frame or 'map')
                if pose:
                    x,y,_=pose; self._lock_route_from_here(x,y)
                self.get_logger().info("Replan aplicado sobre la marcha.")
            else:
                self.get_logger().info("No hay ruta pendiente para replan.")

    def publish_status(self, s):
        self.pub_s.publish(String(data=s))

    # ---------- Meta final ----------
    def _at_goal(self, dist, th):
        """Meta: último punto dentro de xy_tol y, si hay, orientación final dentro de yaw_tol."""
        if self.i == len(self.route)-1 and dist <= self.xy_tol:
            if self.final_yaw is None:
                return True
            e = abs(ang_diff(self.final_yaw, th))
            return e <= self.yaw_tol
        return False

    # ---------- Control con FSM separada (paradas, giro, avance) ----------
    def step(self):
        if self.state!='running':
            if self.state=='done' and self.pending:
                self.frame, self.draft = self.pending; self.pending=None
                self.state='idle'; self.publish_status('idle')
            return

        if not self.route or self.frame is None:
            self.get_logger().info_once("Sin ruta congelada: usa 'start'.")
            return

        pose = self.get_pose_in_frame(self.frame)
        if pose is None:
            self.pub_v.publish(Twist()); return
        x,y,th = pose

        if self.i >= len(self.route):
            self.state='done'; self.pub_v.publish(Twist()); self.publish_status('done'); return

        tx, ty = self.route[self.i]
        dx, dy = tx-x, ty-y
        dist = math.hypot(dx,dy)

        # Referencia angular nominal
        th_ref = math.atan2(dy,dx)

        # Evitar referencia angular inestable cerca del último punto
        if self.i == len(self.route)-1 and dist <= max(2*self.xy_tol, 0.5*self.lookahead):
            th_ref = self.final_yaw if self.final_yaw is not None else th

        # Meta final temprana
        if self._at_goal(dist, th):
            self.state='done'; self.pub_v.publish(Twist()); self.publish_status('done'); return

        v_des = 0.0; w_des = 0.0

        if self.phase == 'turn':
            e_yaw = ang_diff(th_ref, th)
            if abs(e_yaw) <= self.yaw_tol:
                self.phase = 'stop_before_go'; self.stop_t0 = None
            else:
                w_des = clamp(self.k_yaw*e_yaw, -self.wmax, self.wmax)

        elif self.phase == 'go':
            if dist <= self.xy_tol:
                self.phase = 'stop_before_turn'; self.stop_t0=None
            else:
                v_des = clamp(self.k_dist*dist, 0.0, self.vmax)

        elif self.phase == 'stop_before_turn':
            v_des = 0.0; w_des = 0.0
            if abs(self.v_cmd) < self.stop_eps and abs(self.w_cmd) < self.stop_eps:
                if self.stop_t0 is None:
                    self.stop_t0 = time.time()
                if (time.time()-self.stop_t0) >= self.stop_dwell:
                    # Si ya estoy en el último punto y no exijo orientación, terminar
                    if dist <= self.xy_tol and self.i == len(self.route)-1 and self.final_yaw is None:
                        self.state='done'; self.pub_v.publish(Twist()); self.publish_status('done'); return
                    # Si estoy ya dentro del punto actual, avanzar índice antes de girar
                    if dist <= self.xy_tol and self.i < len(self.route)-1:
                        self.i = min(self.i+1, len(self.route)-1)
                        tx, ty = self.route[self.i]
                        dx, dy = tx-x, ty-y
                        th_ref = math.atan2(dy,dx)
                    self.phase = 'turn'

        elif self.phase == 'stop_before_go':
            v_des = 0.0; w_des = 0.0
            if abs(self.v_cmd) < self.stop_eps and abs(self.w_cmd) < self.stop_eps:
                if self.stop_t0 is None:
                    self.stop_t0 = time.time()
                if (time.time()-self.stop_t0) >= self.stop_dwell:
                    if self._at_goal(dist, th):
                        self.state='done'; self.pub_v.publish(Twist()); self.publish_status('done'); return
                    self.phase = 'go'
            else:
                # Si no alcanzó la parada limpia, vuelve a preparar giro
                self.phase = 'stop_before_turn'; self.stop_t0 = None

        # Rampas
        dv = self.accel_v*self.dt
        dw = self.accel_w*self.dt
        self.v_cmd = clamp(self.v_cmd + clamp(v_des - self.v_cmd, -dv, dv), -self.vmax, self.vmax)
        self.w_cmd = clamp(self.w_cmd + clamp(w_des - self.w_cmd, -dw, dw), -self.wmax, self.wmax)

        # Publica
        tw = Twist(); tw.linear.x = self.v_cmd; tw.angular.z = self.w_cmd
        self.pub_v.publish(tw)

        # Debug
        self.debug = {
            'frame': self.frame,
            'idx': self.i,
            'phase': self.phase,
            'pose_in_frame': (x,y,th),
            'target': (tx,ty),
            'lookahead': (tx,ty),
            'dist': dist,
            'v_des': v_des,
            'w_des': w_des,
            'v_cmd': self.v_cmd,
            'w_cmd': self.w_cmd,
        }


# ---------- TUI opcional ----------
class FollowerTUI(Node):
    def __init__(self):
        super().__init__('marker_follower_tui')

def _fmt_pose(p):
    if not p: return "—"
    x,y,th = p
    return f"x={x:+.3f} y={y:+.3f} th={math.degrees(th):+.1f}°"

def _fmt_xy(p):
    if not p: return "—"
    x,y = p
    return f"({x:+.3f}, {y:+.3f})"

def tui(stdscr, tui_node: FollowerTUI, follower: MarkerFollower):
    curses.curs_set(0); stdscr.nodelay(True)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)    # header
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # metrics
    curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)   # ok
    curses.init_pair(4, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # targets
    help_ = "[s start] [SPACE pause/resume] [a abort] [r replan] "
    help_+= "[+/- vmax] [</> yaw_tol] [{/} wmax] [/[ ] k_yaw] [q]"

    def line(y,x,text,color=0):
        stdscr.addstr(y,x,text,curses.color_pair(color))

    while rclpy.ok():
        rclpy.spin_once(follower, timeout_sec=0.03)
        c = stdscr.getch()
        if c!=-1:
            if c in (ord('q'),27): break
            elif c==ord('s'): follower.on_cmd(String(data='start'))
            elif c==ord(' '): follower.on_cmd(String(data='pause' if follower.state=='running' else 'resume'))
            elif c==ord('a'): follower.on_cmd(String(data='abort'))
            elif c==ord('r'): follower.on_cmd(String(data='replan'))
            elif c==ord('+'): follower.vmax=min(follower.vmax*1.15,2.0)
            elif c==ord('-'): follower.vmax=max(follower.vmax/1.15,0.05)
            elif c==ord('<'): follower.yaw_tol=max(follower.yaw_tol/1.2, math.radians(1.0))
            elif c==ord('>'): follower.yaw_tol=min(follower.yaw_tol*1.2, math.radians(20.0))
            elif c==ord('{'): follower.wmax=max(follower.wmax/1.2,0.3)
            elif c==ord('}'): follower.wmax=min(follower.wmax*1.2,3.0)
            elif c==ord('/'): follower.k_yaw=max(follower.k_yaw/1.2,0.1)
            elif c==ord(']'): follower.k_yaw=min(follower.k_yaw*1.2,4.0)

        pose_map = follower.get_pose_in_frame('map')
        pose_odom = follower.get_pose_in_frame('odom')
        dbg = getattr(follower,'debug',{}) or {}
        tgt = dbg.get('target'); lha = dbg.get('lookahead')

        stdscr.erase()
        # Header
        line(0,0," Marker Follower – TUI ".ljust(100),1)
        line(1,0,f" state={follower.state:<8} draft={len(follower.draft):<3} route={len(follower.route):<3} idx={follower.i:<3} phase={follower.phase:<16}",2)
        line(2,0,f" v={follower.v_cmd:+.2f} w={follower.w_cmd:+.2f} | vmax={follower.vmax:.2f} wmax={follower.wmax:.2f} | yaw_tol={math.degrees(follower.yaw_tol):.1f}° k_yaw={follower.k_yaw:.2f}",2)
        line(3,0,help_)

        # Poses
        line(5,0," Poses ",1)
        line(6,2,f"odom: {_fmt_pose(pose_odom)}",0)
        line(7,2,f"map : {_fmt_pose(pose_map)}",0)

        # Ruta / targets
        line(9,0," Ruta / Targets ",1)
        line(10,2,f"frame={dbg.get('frame','—')} dist={dbg.get('dist',float('nan')):.3f} idx={dbg.get('idx','—')} phase={dbg.get('phase','—')}",0)
        line(11,2,f"siguiente: {_fmt_xy(tgt)}",4)
        line(12,2,f"lookahead: {_fmt_xy(lha)}",4)

        # Control
        line(14,0," Control ",1)
        line(15,2,f"v_des={dbg.get('v_des',0.0):+8.3f} w_des={dbg.get('w_des',0.0):+8.3f} → cmd(v,w)=({dbg.get('v_cmd',0.0):+8.3f}, {dbg.get('w_cmd',0.0):+8.3f})",3)

        stdscr.refresh()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tui', action='store_true')
    args = ap.parse_args()

    rclpy.init()
    follower = MarkerFollower()

    if not args.tui:
        try:
            rclpy.spin(follower)
        except KeyboardInterrupt:
            pass
        follower.destroy_node(); rclpy.shutdown(); return

    tui_node = FollowerTUI()
    ex = MultiThreadedExecutor(); ex.add_node(follower); ex.add_node(tui_node)
    try:
        curses.wrapper(tui, tui_node, follower)
    finally:
        ex.shutdown(); follower.destroy_node(); rclpy.shutdown()


if __name__=='__main__':
    main()
