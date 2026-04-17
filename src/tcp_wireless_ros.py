#!/usr/bin/env python3
"""TCP wireless listener for Hand Tracking Streamer (HTS).

Listens for incoming TCP connections over the network (wireless), parses
Left/Right wrist poses, head pose, and Left/Right hand landmarks, then
publishes them to ROS2 topics:

    /left_wrist          (geometry_msgs/PoseStamped)
    /right_wrist         (geometry_msgs/PoseStamped)
    /head_pose           (geometry_msgs/PoseStamped)
    /left_hand/<joint>   (geometry_msgs/PointStamped)  × 21 joints
    /right_hand/<joint>  (geometry_msgs/PointStamped)  × 21 joints

NOTE: Data streamed from HTS follows Unity's left-hand coordinate convention.
      For most applications you will want to flip the Y-axis for a right-hand
      coordinate system.

Message formats expected from HTS:
    Left wrist | f = 126 | t = 2573627960400:, x, y, z, qx, qy, qz, qw
    Right wrist | f = 126 | t = 2573628730200:, x, y, z, qx, qy, qz, qw
    Head pose | f = 126 | t = ...:, x, y, z, qx, qy, qz, qw
    Left landmarks | f = 126 | t = ...:, [x,y,z] * 21 joints
    Right landmarks | f = 126 | t = ...:, [x,y,z] * 21 joints

Example:
    python tcp_wireless_listener.py --host 192.168.123.102 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import socket
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float32


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches pose lines (wrist / head): 7 floats after the timestamp
_POSE_RE = re.compile(
    r"(Left wrist|Right wrist|Head pose)\s*\|\s*f\s*=\s*(\d+)\s*\|\s*t\s*=\s*(\d+):,\s*"
    r"([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*"
    r"([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)"
)

# Matches landmark lines: 63 floats (21 joints × 3) after the timestamp
_LANDMARKS_RE = re.compile(
    r"(Left landmarks|Right landmarks)\s*\|\s*f\s*=\s*(\d+)\s*\|\s*t\s*=\s*(\d+):,\s*"
    r"([-\d.,\s]+)"
)

# ---------------------------------------------------------------------------
# Joint / topic mappings
# ---------------------------------------------------------------------------

_POSE_TOPIC_MAP = {
    "Left wrist":  "left_wrist",
    "Right wrist": "right_wrist",
    "Head pose":   "head_pose",
}

# 21 tracked joints in streamed order (from CONNECTIONS.md)
JOINT_NAMES: list[str] = [
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "little_proximal",
    "little_intermediate",
    "little_distal",
    "little_tip",
]

_LANDMARKS_SIDE_MAP = {
    "Left landmarks":  "left_hand",
    "Right landmarks": "right_hand",
}

NUM_JOINTS = 21
FLOATS_PER_LANDMARK_LINE = NUM_JOINTS * 3  # 63


class PosePublisher(Node):
    """ROS2 node that publishes wrist/head poses and hand landmark positions."""

    def __init__(self) -> None:
        super().__init__("hts_pose_publisher")

        # Pose publishers (wrist + head)
        self._pose_pubs = {
            topic: self.create_publisher(PoseStamped, topic, 10)
            for topic in _POSE_TOPIC_MAP.values()
        }

        # Landmark publishers: {("left_hand", "thumb_tip"): publisher, ...}
        self._landmark_pubs: dict[tuple[str, str], rclpy.publisher.Publisher] = {}
        for side in _LANDMARKS_SIDE_MAP.values():
            for joint in JOINT_NAMES:
                topic = f"{side}/{joint}"
                self._landmark_pubs[(side, joint)] = self.create_publisher(
                    PointStamped, topic, 10
                )

        # Pinch distance publishers
        self._pinch_distance_pubs = {
            "left_hand": self.create_publisher(Float32, "left_hand/pinch_distance", 10),
            "right_hand": self.create_publisher(Float32, "right_hand/pinch_distance", 10),
        }

        all_topics = (
            [f"/{t}" for t in _POSE_TOPIC_MAP.values()]
            + [f"/{s}/{j}" for s in _LANDMARKS_SIDE_MAP.values() for j in JOINT_NAMES]
            + [f"/{s}/pinch_distance" for s in _LANDMARKS_SIDE_MAP.values()]
        )
        self.get_logger().info(
            f"PosePublisher ready — publishing on {len(all_topics)} topics"
        )

    # ---- pose (wrist / head) ------------------------------------------------

    def publish_pose(
        self,
        label: str,
        frame_id: int,
        timestamp_ns: int,
        x: float, y: float, z: float,
        qx: float, qy: float, qz: float, qw: float,
    ) -> None:
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp.sec     = timestamp_ns // 1_000_000_000
        msg.header.stamp.nanosec = timestamp_ns  % 1_000_000_000
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        topic = _POSE_TOPIC_MAP[label]
        self._pose_pubs[topic].publish(msg)
        self.get_logger().debug(
            f"{label} f={frame_id}  "
            f"pos=({x:.3f}, {y:.3f}, {z:.3f})  "
            f"quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})"
        )

    # ---- landmarks -----------------------------------------------------------

    def publish_landmarks(
        self,
        label: str,
        frame_id: int,
        timestamp_ns: int,
        positions: list[tuple[float, float, float]],
    ) -> None:
        side = _LANDMARKS_SIDE_MAP[label]
        sec     = timestamp_ns // 1_000_000_000
        nanosec = timestamp_ns  % 1_000_000_000

        for idx, joint_name in enumerate(JOINT_NAMES):
            x, y, z = positions[idx]
            msg = PointStamped()
            msg.header.frame_id = "world"
            msg.header.stamp.sec     = sec
            msg.header.stamp.nanosec = nanosec
            msg.point.x = x
            msg.point.y = y
            msg.point.z = z

            self._landmark_pubs[(side, joint_name)].publish(msg)

        # Calculate and publish pinch distance (thumb_tip index 4 to index_tip index 8)
        thumb_tip = positions[4]  # index 4 in JOINT_NAMES
        index_tip = positions[8]  # index 8 in JOINT_NAMES
        distance_m = math.sqrt(
            (thumb_tip[0] - index_tip[0])**2 +
            (thumb_tip[1] - index_tip[1])**2 +
            (thumb_tip[2] - index_tip[2])**2
        )
        distance_cm = distance_m * 100  # Convert to centimeters

        pinch_msg = Float32()
        pinch_msg.data = distance_cm
        self._pinch_distance_pubs[side].publish(pinch_msg)

        self.get_logger().debug(
            f"{label} f={frame_id}  published {len(positions)} joints, pinch_distance={distance_cm:.2f} cm"
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_and_publish(line: str, node: PosePublisher) -> None:
    """Try to parse a pose or landmark line and publish; silently skip non-matching."""

    # --- try pose first (wrist / head) ---
    m = _POSE_RE.search(line)
    if m:
        label        = m.group(1)
        frame_id     = int(m.group(2))
        timestamp_ns = int(m.group(3))
        x, y, z      = float(m.group(4)), float(m.group(5)), float(m.group(6))
        qx, qy, qz, qw = (
            float(m.group(7)), float(m.group(8)),
            float(m.group(9)), float(m.group(10)),
        )
        node.publish_pose(label, frame_id, timestamp_ns, x, y, z, qx, qy, qz, qw)
        return

    # --- try landmarks ---
    m = _LANDMARKS_RE.search(line)
    if m:
        label        = m.group(1)
        frame_id     = int(m.group(2))
        timestamp_ns = int(m.group(3))
        raw_floats   = m.group(4)

        values = [float(v) for v in raw_floats.split(",") if v.strip()]
        if len(values) < FLOATS_PER_LANDMARK_LINE:
            logging.warning(
                "Landmark line has %d floats (expected %d), skipping",
                len(values), FLOATS_PER_LANDMARK_LINE,
            )
            return

        positions = [
            (values[i], values[i + 1], values[i + 2])
            for i in range(0, FLOATS_PER_LANDMARK_LINE, 3)
        ]
        node.publish_landmarks(label, frame_id, timestamp_ns, positions)


# ---------------------------------------------------------------------------
# TCP server
# ---------------------------------------------------------------------------

def handle_tcp_connection(conn, addr, node: PosePublisher) -> None:
    """Handle a single TCP connection in a separate thread."""
    with conn:
        logging.info("Accepted connection from %s", addr)
        buffer = ""
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    logging.info("Connection from %s closed", addr)
                    break
                try:
                    buffer += data.decode("utf-8")
                    # Split on newlines; keep any incomplete trailing fragment
                    *lines, buffer = buffer.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line:
                            logging.debug("Message from %s: %s", addr, line)
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
        prog="tcp_wireless_listener",
        description=(
            "Wireless TCP listener for HTS — parses wrist poses, head pose, "
            "and hand landmarks, publishing them as ROS2 topics."
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

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_tcp_server(args.host, args.port, node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()