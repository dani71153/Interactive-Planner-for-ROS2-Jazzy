#!/usr/bin/env -S python3
# -*- coding: utf-8 -*-
"""
viz_markers_stress.py — Publica MarkerArray para testear RViz.
Comandos por tópico: /viz/cmd (std_msgs/String)
  "regen N"     -> vuelve a crear N marcadores
  "clear"       -> limpia
  "type SPHERE|CUBE|ARROW|TEXT" -> cambia tipo
  "random"      -> posiciones y colores aleatorios
Parámetros:
  frame_id: "map" (o "odom")
  ns: "viztest"
  count: 500       (# inicial de marcadores)
  size: 0.08       (escala base)
  lifetime: 0.3    (s) refresco rápido
  pulse: true      (oscilar alpha)
"""
import math, random, re, time
from typing import List
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import String
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

TYPES = {
    "SPHERE": Marker.SPHERE, "CUBE": Marker.CUBE, "ARROW": Marker.ARROW, "TEXT": Marker.TEXT_VIEW_FACING
}

class VizMarkers(Node):
    def __init__(self):
        super().__init__('viz_markers_stress')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('ns', 'viztest')
        self.declare_parameter('count', 500)
        self.declare_parameter('size', 0.08)
        self.declare_parameter('lifetime', 0.3)
        self.declare_parameter('pulse', True)

        self.frame_id = self.get_parameter('frame_id').value
        self.ns       = self.get_parameter('ns').value
        self.count    = int(self.get_parameter('count').value)
        self.size     = float(self.get_parameter('size').value)
        self.life     = float(self.get_parameter('lifetime').value)
        self.pulse    = bool(self.get_parameter('pulse').value)
        self.marker_type = TYPES["SPHERE"]
        self.randomize = False

        self.pub = self.create_publisher(MarkerArray, '/ui/draft_markers', 10)
        self.sub = self.create_subscription(String, '/viz/cmd', self.on_cmd, 10)

        self.points = self._make_points(self.count)
        self.start  = time.time()
        self.timer  = self.create_timer(0.05, self.tick)  # ~20 Hz

        self.get_logger().info(f"frame_id={self.frame_id} count={self.count} type=SPHERE size={self.size} lifetime={self.life}s")

    # ---------- comandos ----------
    def on_cmd(self, msg: String):
        s = msg.data.strip()
        if s.startswith("regen"):
            m = re.search(r"regen\s+(\d+)", s)
            n = int(m.group(1)) if m else self.count
            self.count = n
            self.points = self._make_points(self.count)
            self.get_logger().info(f"regen {n}")
        elif s == "clear":
            self.points = []
            self.get_logger().info("clear")
        elif s.startswith("type"):
            t = s.split(maxsplit=1)[1].strip().upper()
            if t in TYPES:
                self.marker_type = TYPES[t]
                self.get_logger().info(f"type={t}")
            else:
                self.get_logger().warn(f"type inválido: {t}")
        elif s == "random":
            self.randomize = True
            self.points = self._make_points(self.count, randomize=True)
            self.get_logger().info("random on (regen con posiciones/colores aleatorios)")
        else:
            self.get_logger().warn(f"cmd desconocido: {s}")

    # ---------- generación ----------
    def _make_points(self, n: int, randomize: bool=False):
        pts: List[Point] = []
        if randomize or self.randomize:
            for _ in range(n):
                p = Point()
                p.x = random.uniform(-5.0, 5.0)
                p.y = random.uniform(-5.0, 5.0)
                p.z = random.uniform(0.0, 0.5)
                pts.append(p)
        else:
            # grilla en espiral para ver densidad
            k = math.ceil(math.sqrt(n))
            s = self.size * 3.0
            for i in range(n):
                r = i // k; c = i % k
                p = Point(); p.x = (c - k/2) * s; p.y = (r - k/2) * s; p.z = 0.02
                pts.append(p)
        return pts

    # ---------- publicación periódica ----------
    def tick(self):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        pulse_alpha = 1.0
        if self.pulse:
            pulse_alpha = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(2*math.pi*0.7*(time.time()-self.start)))

        # nube principal
        mid = 0
        for i, p in enumerate(self.points):
            m = Marker()
            m.header.frame_id = self.frame_id; m.header.stamp = now
            m.ns = self.ns; m.id = mid; mid += 1
            m.type = self.marker_type; m.action = Marker.ADD
            m.pose.position = p
            m.pose.orientation.w = 1.0
            if self.marker_type == Marker.TEXT_VIEW_FACING:
                m.text = f"{i}"
            # escala
            m.scale.x = self.size
            m.scale.y = self.size if self.marker_type != Marker.ARROW else self.size*0.25
            m.scale.z = self.size if self.marker_type != Marker.ARROW else self.size*0.25
            # color
            if self.randomize:
                random.seed(i)
                m.color.r = random.random(); m.color.g = random.random(); m.color.b = random.random()
            else:
                m.color.r, m.color.g, m.color.b = 0.2, 0.6, 1.0
            m.color.a = pulse_alpha
            m.lifetime = Duration(seconds=self.life).to_msg()
            ma.markers.append(m)

        # línea de referencia (LINE_STRIP) alrededor del borde
        if self.points:
            line = Marker()
            line.header.frame_id = self.frame_id; line.header.stamp = now
            line.ns = self.ns; line.id = mid; mid += 1
            line.type = Marker.LINE_STRIP; line.action = Marker.ADD
            line.scale.x = self.size * 0.5
            line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.2, 0.2, 1.0
            # rectángulo que encierra los puntos
            xs = [p.x for p in self.points]; ys = [p.y for p in self.points]
            minx,maxx,miny,maxy = min(xs),max(xs),min(ys),max(ys)
            for (x,y) in [(minx,miny),(maxx,miny),(maxx,maxy),(minx,maxy),(minx,miny)]:
                q = Point(); q.x,q.y,q.z = x,y,0.02; line.points.append(q)
            line.lifetime = Duration(seconds=self.life).to_msg()
            ma.markers.append(line)

        self.pub.publish(ma)

def main():
    rclpy.init()
    node = VizMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()

