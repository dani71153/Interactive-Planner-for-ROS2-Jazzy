#!/usr/bin/env -S python3
# -*- coding: utf-8 -*-
import math, argparse, curses
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import Buffer, TransformListener

def yaw(q): return math.atan2(2*(q.w*q.z), 1-2*(q.z*q.z))
def clamp(v,a,b): return max(a,min(b,v))
def ang_diff(a,b): return (a-b+math.pi)%(2*math.pi)-math.pi

class MarkerFollower(Node):
    def __init__(self):
        super().__init__('marker_follower')
        # Params
        self.declare_parameter('base_link','base_link')
        self.declare_parameter('cmd_vel_topic','/cmd_vel')
        self.declare_parameter('yaw_tol_deg',6.0)
        self.declare_parameter('xy_tol',0.05)
        self.declare_parameter('k_yaw',1.6)
        self.declare_parameter('k_dist',1.0)
        self.declare_parameter('vmax',0.4)
        self.declare_parameter('wmax',1.8)
        self.declare_parameter('rate_hz',30.0)
        self.base = self.get_parameter('base_link').value
        self.topic = self.get_parameter('cmd_vel_topic').value
        self.yaw_tol = math.radians(float(self.get_parameter('yaw_tol_deg').value))
        self.xy_tol  = float(self.get_parameter('xy_tol').value)
        self.k_yaw   = float(self.get_parameter('k_yaw').value)
        self.k_dist  = float(self.get_parameter('k_dist').value)
        self.vmax    = float(self.get_parameter('vmax').value)
        self.wmax    = float(self.get_parameter('wmax').value)
        self.dt      = 1.0/float(self.get_parameter('rate_hz').value)

        # TF
        self.tf = Buffer(cache_time=Duration(seconds=5.0))
        self.tfl= TransformListener(self.tf, self, spin_thread=True)

        # IO
        self.points = []; self.frame=None
        self.state='idle'; self.i=0; self.phase='turn'  # turn|go
        self.sub_m = self.create_subscription(MarkerArray,'/ui/draft_markers',self.on_markers,10)
        self.sub_c = self.create_subscription(String,'/ui/exec_command',self.on_cmd,10)
        self.pub_v = self.create_publisher(Twist,self.topic,10)
        self.timer = self.create_timer(self.dt, self.step)

    def on_markers(self, ma: MarkerArray):
        line = next((m for m in ma.markers if m.type==Marker.LINE_STRIP), None)
        if not line or not line.points: return
        self.frame = line.header.frame_id or 'map'
        self.points = [(p.x,p.y) for p in line.points]
        self.i = 0; self.phase='turn'
        self.get_logger().info(f"Ruta recibida ({len(self.points)} puntos) en '{self.frame}'.")

    def on_cmd(self, msg: String):
        c = msg.data.strip().lower()
        if c=='start' and self.points: self.state='running'
        elif c=='resume' and self.points: self.state='running'
        elif c=='pause': self.state='paused'; self.pub_v.publish(Twist())
        elif c=='abort': self.state='aborted'; self.pub_v.publish(Twist())
        else: pass

    def step(self):
        if self.state!='running' or not self.points or self.frame is None: return
        t0 = Time(seconds=0, nanoseconds=0)
        if not self.tf.can_transform(self.frame, self.base, t0): return
        tf = self.tf.lookup_transform(self.frame, self.base, t0)
        x = tf.transform.translation.x; y = tf.transform.translation.y; th = yaw(tf.transform.rotation)

        # fin de ruta
        if self.i >= len(self.points): self.state='idle'; self.pub_v.publish(Twist()); return
        tx, ty = self.points[self.i]
        dx, dy = tx-x, ty-y
        dist = math.hypot(dx,dy)
        th_ref = math.atan2(dy,dx)
        e_yaw = ang_diff(th_ref, th)

        tw = Twist()
        if self.phase=='turn':
            if abs(e_yaw) <= self.yaw_tol or dist < 2*self.xy_tol:
                self.phase='go'
            else:
                tw.angular.z = clamp(self.k_yaw*e_yaw, -self.wmax, self.wmax)
        if self.phase=='go':
            if dist <= self.xy_tol:
                self.i += 1; self.phase='turn'
            else:
                tw.linear.x  = clamp(self.k_dist*dist, 0.0, self.vmax)
                tw.angular.z = clamp(self.k_yaw*e_yaw, -0.6*self.wmax, 0.6*self.wmax)
        self.pub_v.publish(tw)

# ---------- TUI opcional ----------
class FollowerTUI(Node):
    def __init__(self): super().__init__('marker_follower_tui')

def tui(stdscr, tui_node: FollowerTUI, follower: MarkerFollower):
    curses.curs_set(0); stdscr.nodelay(True)
    help_ = "[s start] [SPACE pause/resume] [a abort] [+/- vmax] [</> yaw_tol] [q]"
    while rclpy.ok():
        rclpy.spin_once(follower, timeout_sec=0.03)
        c = stdscr.getch()
        if c!=-1:
            if c in (ord('q'),27): break
            elif c==ord('s'): follower.on_cmd(String(data='start'))
            elif c==ord(' '): follower.on_cmd(String(data='pause' if follower.state=='running' else 'resume'))
            elif c==ord('a'): follower.on_cmd(String(data='abort'))
            elif c==ord('+'): follower.vmax=min(follower.vmax*1.15,2.0)
            elif c==ord('-'): follower.vmax=max(follower.vmax/1.15,0.05)
            elif c==ord('<'): follower.yaw_tol=max(follower.yaw_tol/1.2, math.radians(1.0))
            elif c==ord('>'): follower.yaw_tol=min(follower.yaw_tol*1.2, math.radians(20.0))
        stdscr.erase()
        stdscr.addstr(0,0,"Marker Follower – TUI")
        stdscr.addstr(1,0,f"state={follower.state}  pts={len(follower.points)}  idx={follower.i}  phase={follower.phase}")
        stdscr.addstr(2,0,f"vmax={follower.vmax:.2f}  yaw_tol={math.degrees(follower.yaw_tol):.1f}°")
        stdscr.addstr(3,0,help_); stdscr.refresh()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tui', action='store_true')
    args = ap.parse_args()
    rclpy.init()
    follower = MarkerFollower()
    if not args.tui:
        try: rclpy.spin(follower)
        except KeyboardInterrupt: pass
        follower.destroy_node(); rclpy.shutdown(); return
    tui_node = FollowerTUI()
    ex = MultiThreadedExecutor(); ex.add_node(follower); ex.add_node(tui_node)
    try: curses.wrapper(tui, tui_node, follower)
    finally: ex.shutdown(); follower.destroy_node(); rclpy.shutdown()

if __name__=='__main__':
    main()
