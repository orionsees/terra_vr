#!/usr/bin/env python3
"""Publish wrist trajectories as RViz Path markers for live visualization."""

from __future__ import annotations

import argparse
from collections import deque
from time import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
import tf2_ros

from hand_tracking_sdk import (
    unity_left_to_flu_position,
    unity_left_to_flu_rotation,
)


class WristPathVisualizer(Node):
    """ROS2 node that publishes wrist trajectories as Path messages."""

    def __init__(self, max_points: int = 1000) -> None:
        super().__init__("wrist_path_visualizer")

        self._max_points = max_points

        # Store poses in deques
        self._left_path_deque: deque[PoseStamped] = deque(maxlen=max_points)
        self._right_path_deque: deque[PoseStamped] = deque(maxlen=max_points)

        # Frequency tracking
        self._left_timestamps: deque[float] = deque(maxlen=30)
        self._right_timestamps: deque[float] = deque(maxlen=30)
        self._last_left_log_time = time()
        self._last_right_log_time = time()

        # Publish "map" as root TF frame (parent=map, child=odom — identity)
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.rotation.w = 1.0
        self._static_tf_broadcaster.sendTransform(t)

        # Path publishers
        self._left_path_pub = self.create_publisher(Path, "left_wrist_path", 10)
        self._right_path_pub = self.create_publisher(Path, "right_wrist_path", 10)

        # Publish paths on a 10 Hz timer instead of every incoming message
        self.create_timer(0.1, self._publish_paths)

        # High-frequency QoS: best-effort, large queue to avoid drops at 140 Hz
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=300,
        )

        # Subscribe to wrist pose topics
        self.create_subscription(PoseStamped, "left_wrist", self._left_wrist_callback, sensor_qos)
        self.create_subscription(PoseStamped, "right_wrist", self._right_wrist_callback, sensor_qos)

        self.get_logger().info(
            f"Wrist path visualizer ready — Fixed Frame: map  (max {max_points} pts)"
        )

    # ---- callbacks -----------------------------------------------------------

    def _left_wrist_callback(self, msg: PoseStamped) -> None:
        now = time()
        self._left_timestamps.append(now)
        self._log_frequency("Left", self._left_timestamps, self._last_left_log_time)
        if self._left_timestamps[-1] - self._last_left_log_time >= 5.0:
            self._last_left_log_time = now
        self._left_path_deque.append(self._convert_pose(msg))

    def _right_wrist_callback(self, msg: PoseStamped) -> None:
        now = time()
        self._right_timestamps.append(now)
        self._log_frequency("Right", self._right_timestamps, self._last_right_log_time)
        if self._right_timestamps[-1] - self._last_right_log_time >= 5.0:
            self._last_right_log_time = now
        self._right_path_deque.append(self._convert_pose(msg))

    # ---- helpers -------------------------------------------------------------

    def _log_frequency(self, side: str, timestamps: deque, last_log_time: float) -> None:
        """Log message frequency every 5 seconds."""
        if len(timestamps) > 1 and (timestamps[-1] - last_log_time) >= 5.0:
            time_span = timestamps[-1] - timestamps[0]
            if time_span > 0:
                freq = (len(timestamps) - 1) / time_span
                self.get_logger().info(f"{side} wrist: {freq:.1f} Hz ({len(timestamps)} samples)")

    def _convert_pose(self, msg: PoseStamped) -> PoseStamped:
        """Convert pose from VR (Unity left-handed) to RViz (FLU right-handed)."""
        x, y, z = unity_left_to_flu_position(
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        )
        qx, qy, qz, qw = unity_left_to_flu_rotation(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )

        converted = PoseStamped()
        converted.header = msg.header
        converted.header.frame_id = "map"
        converted.pose.position.x = x
        converted.pose.position.y = y
        converted.pose.position.z = z
        converted.pose.orientation.x = qx
        converted.pose.orientation.y = qy
        converted.pose.orientation.z = qz
        converted.pose.orientation.w = qw
        return converted

    def _publish_paths(self) -> None:
        """Publish both paths at 10 Hz (timer callback)."""
        stamp = self.get_clock().now().to_msg()

        if self._left_path_deque:
            path = Path()
            path.header.frame_id = "map"
            path.header.stamp = stamp
            path.poses = list(self._left_path_deque)
            self._left_path_pub.publish(path)

        if self._right_path_deque:
            path = Path()
            path.header.frame_id = "map"
            path.header.stamp = stamp
            path.poses = list(self._right_path_deque)
            self._right_path_pub.publish(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wrist_path_visualizer",
        description="Visualize wrist trajectories live in RViz2",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1000,
        help="Maximum number of points per path (default: 1000)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = WristPathVisualizer(max_points=args.max_points)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
