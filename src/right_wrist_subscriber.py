#!/usr/bin/env python3
"""Subscribes to /right_wrist (geometry_msgs/PoseStamped) and prints XYZ + RPY."""

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


def quaternion_to_rpy(x, y, z, w):
    """Convert quaternion to roll, pitch, yaw (radians)."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class RightWristSubscriber(Node):
    def __init__(self):
        super().__init__("right_wrist_subscriber")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PoseStamped, "/right_wrist", self._callback, qos)
        self.get_logger().info("Subscribed to /right_wrist")

    def _callback(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        roll, pitch, yaw = quaternion_to_rpy(q.x, q.y, q.z, q.w)

        print(
            f"XYZ: ({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) | "
            f"XYZW: ({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f}) | "
            # f"RPY: ({math.degrees(roll):.2f}°, {math.degrees(pitch):.2f}°, {math.degrees(yaw):.2f}°)"
        )


def main():
    rclpy.init()
    node = RightWristSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
