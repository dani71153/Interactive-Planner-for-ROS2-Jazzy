#!/usr/bin/env -S python3
# -*- coding: utf-8 -*-
"""
ui_tui_teleop.py — TUI para SketchPlanner
Controles:
  ↑ / i : avanzar +step_m        ↓ / k : retroceder -step_m
  ← / j : girar +step_deg (CCW)  → / l : girar -step_deg (CW)
  u: undo   c: clear   m: commit   r: retake_tf
  R: RESET manual (pide x,y,yaw,frame)   s: guardar snapshot CSV
  +/-: ajustar step_m             </>: ajustar step_deg
  q/Esc: salir
Registra /ui/draft_path en ui_route_YYYYmmdd_HHMMSS.csv
"""

import curses, math, csv
from datetime import datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path

MARKERS_TOPIC = "/ui/draft_markers"

def yaw_to_quat(yaw):
    q = Quaternion(); q.z = math.sin(yaw*0.5); q.w = math.cos(yaw*0.5); return q

class UITeleop(Node):
    def __init__(self, step_m=1.0, step_deg=90.0):
        super().__init__('ui_tui_teleop')
        self.step_m = step_m
        self.step_deg = step_deg
        self.path = []  # [(x,y,yaw)]
        self.pub_adv  = self.create_publisher(Float32, '/ui/advance', 10)
        self.pub_turn = self.create_publisher(Float32, '/ui/turn_deg', 10)
        self.pub_cmd  = self.create_publisher(String, '/ui/command', 10)
        self.pub_rst  = self.create_publisher(PoseStamped, '/ui/reset_pose', 10)
        self.sub_path = self.create_subscription(Path, '/ui/draft_path', self.on_path, 10)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_name = f"ui_route_{ts}.csv"
        self.csv = open(self.csv_name, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.csv); self.w.writerow(["x","y","yaw_deg"])

    def on_path(self, msg: Path):
        self.path = []
        for ps in msg.poses:
            z,w = ps.pose.orientation.z, ps.pose.orientation.w
            yaw = math.degrees(math.atan2(2.0*(w*z), 1.0 - 2.0*(z*z)))
            self.path.append((ps.pose.position.x, ps.pose.position.y, yaw))

    # ---- helpers ----
    def adv(self, m): self.pub_adv.publish(Float32(data=float(m)))
    def turn(self, deg): self.pub_turn.publish(Float32(data=float(deg)))
    def cmd(self, s): self.pub_cmd.publish(String(data=s))

    def reset_manual(self, stdscr):
        curses.echo()
        try:
            stdscr.addstr(4,0,"RESET x y yaw_deg frame(map/odom): "); stdscr.clrtoeol()
            line = stdscr.getstr(4,38,60).decode().strip()
            parts = line.split()
            if len(parts) >= 3:
                x=float(parts[0]); y=float(parts[1]); yaw=float(parts[2]); frame = parts[3] if len(parts)>3 else "map"
                ps=PoseStamped(); ps.header.frame_id=frame
                ps.pose.position.x=x; ps.pose.position.y=y; ps.pose.orientation=yaw_to_quat(math.radians(yaw))
                self.pub_rst.publish(ps)
        except Exception:
            pass
        curses.noecho()

    def snapshot_csv(self):
        for x,y,yaw in self.path: self.w.writerow([x,y,yaw]); self.csv.flush()

def tui(stdscr, node: UITeleop):
    curses.curs_set(0); stdscr.nodelay(True)
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        h,w = stdscr.getmaxyx()
        stdscr.erase()
        stdscr.addstr(0,0,"[↑/i:+m][↓/k:-m][←/j:+deg][→/l:-deg]  u undo  c clear  m commit  r retake_tf  R RESET  s saveCSV  +/- m  </> deg  q quit")
        if node.path:
            x,y,yaw = node.path[-1]
            stdscr.addstr(1,0,f"Verts:{len(node.path)}  Pose x={x:+.2f} y={y:+.2f} yaw={yaw:+.1f}°   step_m={node.step_m:.2f}  step_deg={node.step_deg:.1f}   markers:{MARKERS_TOPIC}")
        else:
            stdscr.addstr(1,0,f"Verts:0  (sin start)   step_m={node.step_m:.2f}  step_deg={node.step_deg:.1f}   markers:{MARKERS_TOPIC}")

        # pequeño mini-mapa ASCII
        if node.path:
            xs=[p[0] for p in node.path]; ys=[p[1] for p in node.path]
            minx,maxx=min(xs),max(xs); miny,maxy=min(ys),max(ys)
            pad=0.2; minx-=pad;maxx+=pad;miny-=pad;maxy+=pad
            W=max(20, w-2); H=max(8, min(h-4, 24))
            grid=[[" "]*W for _ in range(H)]
            def to_scr(xv,yv):
                X=int((xv-minx)/(maxx-minx+1e-9)*(W-1))
                Y=int((1-(yv-miny)/(maxy-miny+1e-9))*(H-1))
                return max(0,min(W-1,X)), max(0,min(H-1,Y))
            for i,(xx,yy,_) in enumerate(node.path):
                X,Y=to_scr(xx,yy); grid[Y][X] = "▸" if i==len(node.path)-1 else "o"
            for r,line in enumerate(grid):
                if 3+r >= h-1: break
                stdscr.addstr(3+r, 0, "".join(line)[:w-1])

        c = stdscr.getch()
        if c == -1: continue
        if c in (ord('q'), 27): break
        elif c in (curses.KEY_UP, ord('i')): node.adv(+node.step_m)
        elif c in (curses.KEY_DOWN, ord('k')): node.adv(-node.step_m)
        elif c in (curses.KEY_LEFT, ord('j')): node.turn(+node.step_deg)
        elif c in (curses.KEY_RIGHT, ord('l')): node.turn(-node.step_deg)
        elif c == ord('u'): node.cmd('undo')
        elif c == ord('c'): node.cmd('clear')
        elif c == ord('m'): node.cmd('commit')
        elif c == ord('r'): node.cmd('retake_tf')
        elif c == ord('R'): node.reset_manual(stdscr)
        elif c == ord('s'): node.snapshot_csv()
        elif c == ord('+'): node.step_m = round(node.step_m*1.25, 3)
        elif c == ord('-'): node.step_m = round(max(0.01, node.step_m/1.25), 3)
        elif c == ord('<'): node.step_deg = max(1.0, round(node.step_deg/1.25, 1))
        elif c == ord('>'): node.step_deg = round(node.step_deg*1.25, 1)

def main():
    rclpy.init()
    node = UITeleop(step_m=1.0, step_deg=90.0)
    try: curses.wrapper(tui, node)
    finally:
        node.snapshot_csv()
        node.csv.close()
        node.destroy_node()
        rclpy.shutdown()
        print(f"CSV guardado en: {node.csv_name}\nTópico de marcadores: {MARKERS_TOPIC}")

if __name__ == "__main__":
    main()
