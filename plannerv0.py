#!/usr/bin/env -S python3
# -*- coding: utf-8 -*-
"""
Sketch Planner (Differential Drive) – PoC + TUI + Debug + TF Retries + Multi-frame
- Edita una ruta con /ui/advance (m) y /ui/turn_deg (grados).
- Visualiza Path y Markers en RViz.
- Commit opcional contra Nav2 (ComputePathThroughPoses).
- TF: reintentos no bloqueantes + búsqueda de combinaciones (map/odom × base_link/base_footprint).
Uso:
  python3 sketch_planner.py                 # solo nodo
  python3 sketch_planner.py --tui           # nodo + interfaz ASCII
"""

import math, time, argparse, threading, curses
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time

from std_msgs.msg import Float32, String
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import ComputePathThroughPoses
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException, ConnectivityException

# ---------- Utils ----------
def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion(); q.z = math.sin(yaw * 0.5); q.w = math.cos(yaw * 0.5); return q
def quat_to_yaw(q: Quaternion) -> float:
    return math.atan2(2.0*(q.w*q.z), 1.0 - 2.0*(q.z*q.z))
def norm_angle(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi

# =====================================================================
#                           SKETCH PLANNER NODE
# =====================================================================
class SketchPlanner(Node):
    def __init__(self):
        super().__init__('sketch_planner')
        self.get_logger().info("== SketchPlanner iniciado ==")

        # Parámetros
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('use_tf_start', True)
        self.declare_parameter('validate_on_commit', True)
        self.declare_parameter('planner_action_ns', '/planner_server/compute_path_through_poses')
        self.declare_parameter('marker_ns', 'sketch')
        self.declare_parameter('tf_retries', 10)
        self.declare_parameter('tf_retry_delay', 0.1)
        # Listas de candidatos (prioridad izq→der)
        self.declare_parameter('target_frames', ['map', 'odom'])
        self.declare_parameter('base_frames',   ['base_link', 'base_footprint'])

        self.frame_id   = self.get_parameter('frame_id').value
        self.base_link  = self.get_parameter('base_link').value
        self.use_tf     = self.get_parameter('use_tf_start').value
        self.do_validate= self.get_parameter('validate_on_commit').value
        self.planner_ns = self.get_parameter('planner_action_ns').value
        self.marker_ns  = self.get_parameter('marker_ns').value
        self.tf_retries = int(self.get_parameter('tf_retries').value)
        self.tf_delay   = float(self.get_parameter('tf_retry_delay').value)
        self.target_frames = list(self.get_parameter('target_frames').value)
        self.base_frames   = list(self.get_parameter('base_frames').value)

        self.get_logger().info(
            f"frame_id={self.frame_id} base_link={self.base_link} "
            f"use_tf_start={self.use_tf} validate_on_commit={self.do_validate} "
            f"tf_retries={self.tf_retries} tf_retry_delay={self.tf_delay}s "
            f"targets={self.target_frames} bases={self.base_frames}"
        )

        # TF
        self.tf_buf = Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self.tf_ls  = TransformListener(self.tf_buf, self, spin_thread=True)

        # Estado
        self.poses: List[PoseStamped] = []

        # Pub/Sub
        self.pub_path    = self.create_publisher(Path, '/ui/draft_path', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/ui/draft_markers', 10)
        self.pub_commit  = self.create_publisher(Path, '/ui/committed_path', 10)

        self.sub_advance = self.create_subscription(Float32, '/ui/advance', self.on_advance, 10)
        self.sub_turn    = self.create_subscription(Float32, '/ui/turn_deg', self.on_turn, 10)
        self.sub_reset   = self.create_subscription(PoseStamped, '/ui/reset_pose', self.on_reset_pose, 10)
        self.sub_cmd     = self.create_subscription(String, '/ui/command', self.on_command, 10)

        # Planner
        self.ac_planner = ActionClient(self, ComputePathThroughPoses, self.planner_ns)

        self.bootstrap_start()

    # ---------- Handlers ----------
    def on_advance(self, msg: Float32):
        self.get_logger().info(f"ADVANCE {msg.data:.3f} m")
        if not self.ensure_start(): return
        d = float(msg.data)
        if abs(d) < 1e-6: self.get_logger().warn("Distancia ~0, ignorada"); return
        last = self.poses[-1]; yaw = quat_to_yaw(last.pose.orientation)
        new = PoseStamped(); new.header.frame_id = self.frame_id
        new.header.stamp = self.get_clock().now().to_msg()
        new.pose.position.x = last.pose.position.x + d * math.cos(yaw)
        new.pose.position.y = last.pose.position.y + d * math.sin(yaw)
        new.pose.orientation = last.pose.orientation
        self.poses.append(new)
        self.publish_visuals()

    def on_turn(self, msg: Float32):
        self.get_logger().info(f"TURN {msg.data:.3f}°")
        if not self.ensure_start(): return
        deg = float(msg.data); last = self.poses[-1]
        yaw2 = norm_angle(quat_to_yaw(last.pose.orientation) + math.radians(deg))
        turned = PoseStamped(); turned.header.frame_id=self.frame_id
        turned.header.stamp = self.get_clock().now().to_msg()
        turned.pose = Pose(); turned.pose.position = last.pose.position
        turned.pose.orientation = yaw_to_quat(yaw2)
        self.poses.append(turned)
        self.publish_visuals()

    def on_reset_pose(self, msg: PoseStamped):
        self.get_logger().info("RESET pose manual")
        self.poses = [self._stamped_in_frame(msg)]
        self.publish_visuals()

    def on_command(self, msg: String):
        cmd = msg.data.strip().lower()
        self.get_logger().info(f"CMD: {cmd}")
        if cmd == 'undo':
            if self.poses: self.poses.pop()
            if not self.poses: self.bootstrap_start()
            self.publish_visuals()
        elif cmd == 'clear':
            self.poses.clear(); self.publish_visuals()
        elif cmd == 'commit':
            self.commit()
        elif cmd == 'retake_tf':
            self.bootstrap_start(force_tf=True)
        else:
            self.get_logger().warn("Comando desconocido")

    # ---------- Core ----------
    def bootstrap_start(self, force_tf=False):
        self.get_logger().info("Bootstrap start…")
        self.poses.clear()
        if self.use_tf or force_tf:
            start = self._pose_from_tf_auto()
            if start: self.poses.append(start)
            else: self.get_logger().warn("TF no disponible; espera /ui/reset_pose")
        else:
            self.get_logger().info("use_tf_start=false; espera /ui/reset_pose")
        self.publish_visuals()

    def ensure_start(self):
        if self.poses: return True
        self.get_logger().warn("Sin start; intentando TF…")
        start = self._pose_from_tf_auto()
        if start: self.poses.append(start); return True
        self.get_logger().error("No hay start (TF ni manual)"); return False

    def commit(self):
        if len(self.poses) < 2: self.get_logger().warn("≥2 vértices para commit"); return
        if not self.do_validate:
            self.pub_commit.publish(self._path_from(self.poses)); return
        if not self.ac_planner.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Planner no disponible; publico boceto")
            self.pub_commit.publish(self._path_from(self.poses)); return
        goal = ComputePathThroughPoses.Goal(); goal.goals = [p for p in self.poses]
        send = self.ac_planner.send_goal_async(goal); rclpy.spin_until_future_complete(self, send)
        if not send.result().accepted: self.publish_visuals(error=True); return
        getres = send.result().get_result_async(); rclpy.spin_until_future_complete(self, getres)
        result = getres.result()
        if not result or not result.path.poses: self.publish_visuals(error=True); return
        result.path.header.frame_id = self.frame_id
        result.path.header.stamp = self.get_clock().now().to_msg()
        self.pub_commit.publish(result.path)

    # ---------- Visual ----------
    def publish_visuals(self, error=False):
        self.pub_path.publish(self._path_from(self.poses))
        ma = MarkerArray(); now = self.get_clock().now().to_msg()
        line = Marker(); line.header.frame_id=self.frame_id; line.header.stamp=now
        line.ns=self.marker_ns; line.id=0; line.type=Marker.LINE_STRIP; line.action=Marker.ADD
        line.scale.x=0.03
        line.color.r,line.color.g,line.color.b,line.color.a = (1.0,0.2,0.2,1.0) if error else (0.2,0.5,1.0,1.0)
        for p in self.poses:
            pt=Point(); pt.x,pt.y,pt.z=p.pose.position.x,p.pose.position.y,0.02; line.points.append(pt)
        ma.markers.append(line)
        vid=1
        for p in self.poses:
            dot=Marker(); dot.header.frame_id=self.frame_id; dot.header.stamp=now
            dot.ns=self.marker_ns; dot.id=vid; vid+=1
            dot.type=Marker.SPHERE; dot.action=Marker.ADD; dot.pose=p.pose
            dot.scale.x=dot.scale.y=dot.scale.z=0.07
            dot.color.r,dot.color.g,dot.color.b,dot.color.a=0.9,0.9,0.9,1.0
            ma.markers.append(dot)
        if self.poses:
            last=self.poses[-1]; arr=Marker(); arr.header.frame_id=self.frame_id; arr.header.stamp=now
            arr.ns=self.marker_ns; arr.id=vid; arr.type=Marker.ARROW; arr.action=Marker.ADD
            arr.pose=last.pose; arr.scale.x=0.25; arr.scale.y=0.05; arr.scale.z=0.05
            arr.color.r,arr.color.g,arr.color.b,arr.color.a=0.3,1.0,0.3,1.0
            ma.markers.append(arr)
        for m in ma.markers: m.lifetime = Duration(seconds=0.3).to_msg()
        self.pub_markers.publish(ma)

    # ---------- Helpers ----------
    def _path_from(self, poses: List[PoseStamped]) -> Path:
        path = Path(); path.header.frame_id=self.frame_id
        path.header.stamp=self.get_clock().now().to_msg(); path.poses=poses; return path

    def _stamped_in_frame(self, ps: PoseStamped) -> PoseStamped:
        out=PoseStamped(); out.header.frame_id=self.frame_id
        out.header.stamp=self.get_clock().now().to_msg(); out.pose=ps.pose; out.pose.position.z=0.0; return out

    def _pose_from_tf_auto(self) -> PoseStamped | None:
        """Busca la mejor combinación disponible: (target in target_frames) × (base in base_frames)."""
        t0 = Time(seconds=0, nanoseconds=0)  # TIME_ZERO
        for attempt in range(1, self.tf_retries + 1):
            for tgt in self.target_frames:
                for base in self.base_frames:
                    ok = self.tf_buf.can_transform(tgt, base, t0, timeout=Duration(seconds=0.0))
                    if not ok: continue
                    try:
                        tf = self.tf_buf.lookup_transform(tgt, base, t0, timeout=Duration(seconds=0.0))
                        ps = PoseStamped()
                        ps.header.frame_id = tgt
                        ps.header.stamp    = tf.header.stamp
                        ps.pose.position.x = tf.transform.translation.x
                        ps.pose.position.y = tf.transform.translation.y
                        ps.pose.position.z = 0.0
                        ps.pose.orientation = tf.transform.rotation
                        self.frame_id, self.base_link = tgt, base
                        self.get_logger().info(f"TF OK {tgt}<-{base} (intento {attempt})")
                        return ps
                    except (LookupException, ExtrapolationException, ConnectivityException) as e:
                        self.get_logger().warn(f"lookup {tgt}<-{base} falló (int {attempt}): {e}")
            self.get_logger().warn(f"TF no disponible (intento {attempt}/{self.tf_retries})")
            time.sleep(self.tf_delay)
        self.get_logger().error("Sin combinación válida de target/base")
        return None

# =====================================================================
#                          TUI (opcional)
# =====================================================================
class SketchTUI(Node):
    def __init__(self, step_m=1.0, step_deg=90.0):
        super().__init__('sketch_tui')
        self.step_m = step_m; self.step_deg = step_deg
        self.path: List[Tuple[float,float,float]] = []
        self.pub_adv  = self.create_publisher(Float32, '/ui/advance', 10)
        self.pub_turn = self.create_publisher(Float32, '/ui/turn_deg', 10)
        self.pub_cmd  = self.create_publisher(String, '/ui/command', 10)
        self.sub_path = self.create_subscription(Path, '/ui/draft_path', self.on_path, 10)

    def on_path(self, msg: Path):
        self.path = [(p.pose.position.x, p.pose.position.y, quat_to_yaw(p.pose.orientation)) for p in msg.poses]

    def _ascii_plot(self, H, W):
        if not self.path: return ["(sin datos)"]
        xs=[p[0] for p in self.path]; ys=[p[1] for p in self.path]
        minx,maxx=min(xs),max(xs); miny,maxy=min(ys),max(ys)
        pad=0.2; minx-=pad;maxx+=pad;miny-=pad;maxy+=pad
        W=max(10,W-2); H=max(10,H-2)
        grid=[[" "]*W for _ in range(H)]
        def to_scr(x,y):
            X=int((x-minx)/(maxx-minx+1e-9)*(W-1))
            Y=int((1-(y-miny)/(maxy-miny+1e-9))*(H-1))
            return max(0,min(W-1,X)), max(0,min(H-1,Y))
        for i,(x,y,_) in enumerate(self.path):
            X,Y=to_scr(x,y); grid[Y][X] = "▸" if i==len(self.path)-1 else "o"
        return ["".join(r) for r in grid]

    def _send(self, k, v):
        if k=='adv': self.pub_adv.publish(Float32(data=float(v)))
        elif k=='turn': self.pub_turn.publish(Float32(data=float(v)))
        elif k=='cmd': self.pub_cmd.publish(String(data=str(v)))

    def tui_loop(self, stdscr):
        stdscr.nodelay(True); curses.curs_set(0)
        header="[w/W avanzar] [a/d/A/D girar] [u undo] [c clear] [m commit] [r retake_tf] [q salir]"
        last=0.0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.03)
            c=stdscr.getch()
            if c!=-1:
                if c in (ord('q'),27): break
                elif c==ord('w'): self._send('adv', self.step_m)
                elif c==ord('W'):
                    curses.echo(); stdscr.addstr(2,0,"metros: "); stdscr.clrtoeol()
                    try: v=float(stdscr.getstr(2,8,20).decode()); self._send('adv', v)
                    except: pass; curses.noecho()
                elif c==ord('a'): self._send('turn', +self.step_deg)
                elif c==ord('d'): self._send('turn', -self.step_deg)
                elif c==ord('A'):
                    curses.echo(); stdscr.addstr(2,0,"grados(+CCW): "); stdscr.clrtoeol()
                    try: v=float(stdscr.getstr(2,14,20).decode()); self._send('turn', v)
                    except: pass; curses.noecho()
                elif c==ord('D'):
                    curses.echo(); stdscr.addstr(2,0,"grados(-CW): "); stdscr.clrtoeol()
                    try: v=float(stdscr.getstr(2,13,20).decode()); self._send('turn', -abs(v))
                    except: pass; curses.noecho()
                elif c==ord('u'): self._send('cmd','undo')
                elif c==ord('c'): self._send('cmd','clear')
                elif c==ord('m'): self._send('cmd','commit')
                elif c==ord('r'): self._send('cmd','retake_tf')
            t=time.time()
            if t-last>0.08:
                stdscr.erase(); stdscr.addstr(0,0,"TUI "+header)
                if self.path:
                    x,y,yaw=self.path[-1]; stdscr.addstr(1,0,f"Pose x={x:+.2f} y={y:+.2f} yaw={math.degrees(yaw):+.1f}° verts={len(self.path)}")
                else:
                    stdscr.addstr(1,0,"Pose: (sin start)  verts=0")
                plot=self._ascii_plot(*stdscr.getmaxyx())
                for i,line in enumerate(plot[:max(0,stdscr.getmaxyx()[0]-4)]):
                    stdscr.addstr(4+i,0,line[:max(1,stdscr.getmaxyx()[1]-1)])
                stdscr.refresh(); last=t

# =====================================================================
# main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Sketch Planner + TUI + TF Retries + Multi-frame")
    parser.add_argument('--tui', action='store_true', help='activar interfaz ASCII')
    parser.add_argument('--step-m', type=float, default=1.0)
    parser.add_argument('--step-deg', type=float, default=90.0)
    args = parser.parse_args()

    rclpy.init()
    planner = SketchPlanner()

    if not args.tui:
        try: rclpy.spin(planner)
        except KeyboardInterrupt: pass
        planner.destroy_node(); rclpy.shutdown(); return

    tui = SketchTUI(step_m=args.step_m, step_deg=args.step_deg)
    exec_ = MultiThreadedExecutor(); exec_.add_node(planner); exec_.add_node(tui)
    th = threading.Thread(target=exec_.spin, daemon=True); th.start()
    try: curses.wrapper(tui.tui_loop)
    finally:
        exec_.shutdown(); tui.destroy_node(); planner.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()
