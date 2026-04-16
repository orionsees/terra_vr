#!/usr/bin/env python3
"""TCP wireless listener for Hand Tracking Streamer (HTS).

Listens for incoming TCP connections over the network (wireless), parses
Left/Right wrist pose messages, and publishes them to ROS2 topics:

    /left_wrist   (geometry_msgs/PoseStamped)
    /right_wrist  (geometry_msgs/PoseStamped)

Message format expected from HTS:
    Left wrist | f = 126 | t = 2573627960400:, -0.1847, 0.7384, 3.2628, 0.335, 0.574, -0.076, -0.743
    Right wrist | f = 126 | t = 2573628730200:, 0.0176, 0.8052, 3.6171, 0.441, -0.332, -0.785, 0.280

    Fields after the timestamp: x, y, z, qx, qy, qz, qw

Example:
    python tcp_wireless.py --host 192.168.123.102 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import re
import socket
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


# Matches "Left wrist", "Right wrist", or "Head pose" lines.
# Groups: (1) label, (2) frame, (3) timestamp_ns, (4..10) x y z qx qy qz qw
_POSE_RE = re.compile(
    r"(Left wrist|Right wrist|Head pose)\s*\|\s*f\s*=\s*(\d+)\s*\|\s*t\s*=\s*(\d+):,\s*"
    r"([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*"
    r"([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)"
)

_TOPIC_MAP = {
    "Left wrist":  "left_wrist",
    "Right wrist": "right_wrist",
    "Head pose":   "head_pose",
}


class PosePublisher(Node):
    """ROS2 node that publishes wrist and head PoseStamped messages."""

    def __init__(self) -> None:
        super().__init__("hts_pose_publisher")
        self._pubs = {
            topic: self.create_publisher(PoseStamped, topic, 10)
            for topic in _TOPIC_MAP.values()
        }
        self.get_logger().info(
            "PosePublisher ready — publishing on: "
            + ", ".join(f"/{t}" for t in _TOPIC_MAP.values())
        )

    def publish_pose(
        self,
        label: str,
        frame_id: int,
        timestamp_ns: int,
        x: float, y: float, z: float,
        qx: float, qy: float, qz: float, qw: float,
    ) -> None:
        msg = PoseStamped()

        # Header
        msg.header.frame_id = "world"
        msg.header.stamp.sec     = timestamp_ns // 1_000_000_000
        msg.header.stamp.nanosec = timestamp_ns  % 1_000_000_000

        # Position
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        # Orientation (quaternion)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        topic = _TOPIC_MAP[label]
        self._pubs[topic].publish(msg)
        # ROS2 logger requires a single pre-formatted string
        self.get_logger().debug(
            f"{label} f={frame_id}  "
            f"pos=({x:.3f}, {y:.3f}, {z:.3f})  "
            f"quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})"
        )


def parse_and_publish(line: str, node: PosePublisher) -> None:
    """Try to parse a pose line and publish it; silently skip non-matching lines."""
    m = _POSE_RE.search(line)
    if not m:
        return

    label        = m.group(1)           # "Left wrist", "Right wrist", or "Head pose"
    frame_id     = int(m.group(2))
    timestamp_ns = int(m.group(3))
    x,  y,  z   = float(m.group(4)),  float(m.group(5)),  float(m.group(6))
    qx, qy, qz, qw = (
        float(m.group(7)), float(m.group(8)),
        float(m.group(9)), float(m.group(10)),
    )

    node.publish_pose(label, frame_id, timestamp_ns, x, y, z, qx, qy, qz, qw)


def handle_tcp_connection(conn, addr, node: PosePublisher) -> None:
    """Handle a single TCP connection in a separate thread."""
    with conn:
        logging.info("Accepted connection from %s", addr)
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    logging.info("Connection from %s closed", addr)
                    break
                try:
                    message = data.decode("utf-8")
                    for line in message.split("\n"):
                        line = line.strip()
                        if line:
                            logging.info("Message from %s: %s", addr, line)
                            parse_and_publish(line, node)
                except UnicodeDecodeError:
                    logging.warning("Non-UTF-8 data from %s (skipped): %s", addr, data)
        except Exception as e:
            logging.error("Error handling connection from %s: %s", addr, e)


def run_tcp_server(host: str, port: int, node: PosePublisher) -> None:
    """Listen for wireless TCP connections and dispatch each to a handler thread."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)

    logging.info("TCP server listening on %s:%d", host, port)
    logging.info("Waiting for wireless connections from HTS...")

    try:
        while True:
            conn, addr = server_sock.accept()
            thread = threading.Thread(
                target=handle_tcp_connection,
                args=(conn, addr, node),
                daemon=True,
            )
            thread.start()
    except KeyboardInterrupt:
        logging.info("Shutting down.")
    finally:
        server_sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tcp_wireless",
        description=(
            "Wireless TCP listener for HTS — parses Left/Right wrist poses "
            "and publishes them to /left_wrist and /right_wrist (geometry_msgs/PoseStamped)."
        ),
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP to bind to (default: 0.0.0.0 — all interfaces).",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rclpy.init()
    node = PosePublisher()

    # Spin the ROS2 node in a background thread so the TCP server runs on main.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_tcp_server(args.host, args.port, node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()