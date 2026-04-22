#!/usr/bin/env python3
"""Zero Arms — Set both SO101 arms to zero joint angle state.

Sets all joints on both left and right arms to 0.0 degrees.

Uses separate topics for each arm:
  - /left_arm/cmd/set_positions
  - /right_arm/cmd/set_positions

Usage:
    python3 zero_arms.py
"""

import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ZeroArms(Node):
    def __init__(self):
        super().__init__("zero_arms")

        # Publishers for both arms with left/right prefixes
        self.pub_left = self.create_publisher(String, "/left_arm/cmd/set_positions", 10)
        self.pub_right = self.create_publisher(String, "/right_arm/cmd/set_positions", 10)

        # Default joint names
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

        # Create zero target (all joints at 0 degrees)
        zero_targets = {name: 0.0 for name in joint_names}
        payload = json.dumps(zero_targets)

        # Send to both arms
        msg = String()
        msg.data = payload

        self.pub_left.publish(msg)
        self.pub_right.publish(msg)

        self.get_logger().info(f"Published zero joint angles to both arms: {payload}")


def main():
    rclpy.init()
    node = None
    try:
        node = ZeroArms()
        # Give a moment for the publisher to deliver
        rclpy.spin_once(node, timeout_sec=1.0)
        print("Both arms set to zero joint angles!")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
