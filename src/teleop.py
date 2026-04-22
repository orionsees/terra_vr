#!/usr/bin/env python3
"""Teleoperation node: gripper control and wrist orientation monitoring.

Subscribes to /right_wrist and /left_wrist (PoseStamped) from TCP listener and prints
orientation in roll-pitch-yaw degrees.

Subscribes to pinch distance and publishes gripper servo commands to gripper servos.

Launch after tcp_wireless is running:
    ros2 run arm_package so101_teleop_node
"""

from __future__ import annotations

import json
import os
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation


class SO101TeleopNode(Node):
    def __init__(self):
        super().__init__("so101_teleop")

        # --- Parameters ---
        package_share_dir = get_package_share_directory('arm_vr')
        right_config_path = os.path.join(package_share_dir, 'config', 'right_arm.json')
        left_config_path = os.path.join(package_share_dir, 'config', 'left_arm.json')

        # --- Load gripper configs ---
        with open(right_config_path, 'r') as f:
            right_config = json.load(f)
        with open(left_config_path, 'r') as f:
            left_config = json.load(f)
            
        self.right_gripper_config = right_config["gripper"]
        self.left_gripper_config = left_config["gripper"]
        
        self.right_gripper_range_min = self.right_gripper_config["range_min"]
        self.right_gripper_range_max = self.right_gripper_config["range_max"]
        self.right_gripper_motor_id = self.right_gripper_config["id"]
        
        self.left_gripper_range_min = self.left_gripper_config["range_min"]
        self.left_gripper_range_max = self.left_gripper_config["range_max"]
        self.left_gripper_motor_id = self.left_gripper_config["id"]

        # --- Load motor ID 5 (wrist_roll) configs for wrist yaw control ---
        self.right_motor5_range_min = right_config["wrist_roll"]["range_min"]
        self.right_motor5_range_max = right_config["wrist_roll"]["range_max"]
        
        self.left_motor5_range_min = left_config["wrist_roll"]["range_min"]
        self.left_motor5_range_max = left_config["wrist_roll"]["range_max"]

        # --- Joint state initialization (zero position) ---
        self.n_joints = 5
        self.right_current_joint_deg = np.zeros(self.n_joints, dtype=float)
        self.left_current_joint_deg = np.zeros(self.n_joints, dtype=float)
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

        # --- Publishers ---
        self.right_pub_cmd = self.create_publisher(String, "/right_arm/cmd/set_positions", 10)
        self.right_pub_cmd_raw = self.create_publisher(String, "/right_arm/cmd/set_positions_raw", 10)
        
        self.left_pub_cmd = self.create_publisher(String, "/left_arm/cmd/set_positions", 10)
        self.left_pub_cmd_raw = self.create_publisher(String, "/left_arm/cmd/set_positions_raw", 10)

        # --- Subscribers ---
        self.create_subscription(
            PoseStamped,
            "/right_wrist",
            self._on_right_wrist,
            10,
        )
        
        self.create_subscription(
            PoseStamped,
            "/left_wrist",
            self._on_left_wrist,
            10,
        )

        self.create_subscription(
            Float32,
            "/right_hand/pinch_distance",
            self._on_right_pinch_distance,
            10,
        )
        
        self.create_subscription(
            Float32,
            "/left_hand/pinch_distance",
            self._on_left_pinch_distance,
            10,
        )

        # Send both arms to home (all zeros) on startup
        self._publish_joint_command_right(self.right_current_joint_deg)
        self._publish_joint_command_left(self.left_current_joint_deg)

        self.get_logger().info(
            "SO101 gripper teleop node ready"
        )
        self.get_logger().info(
            f"Right gripper: motor {self.right_gripper_motor_id} [{self.right_gripper_range_min}-{self.right_gripper_range_max}]"
        )
        self.get_logger().info(
            f"Left gripper: motor {self.left_gripper_motor_id} [{self.left_gripper_range_min}-{self.left_gripper_range_max}]"
        )


    # ------------------------------------------------------------------
    # Right arm wrist callback - print orientation
    # ------------------------------------------------------------------
    def _on_right_wrist(self, msg: PoseStamped) -> None:
        quat = msg.pose.orientation
        r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        self.get_logger().info(
            f"Right wrist yaw: {yaw:.2f}°"
        )
        
        # Map yaw (-180 to 180°) to motor ID 5 range
        # Normalize yaw from [-180, 180] to [0, 1]
        normalized_yaw = (yaw + 180.0) / 360.0
        servo_position = self.right_motor5_range_min + normalized_yaw * (self.right_motor5_range_max - self.right_motor5_range_min)
        
        # Clamp servo position to motor's physical limits
        servo_position = max(self.right_motor5_range_min, min(self.right_motor5_range_max, servo_position))
        servo_position = int(servo_position)
        
        # Publish raw servo command
        cmd = {"5": servo_position}
        msg_raw = String()
        msg_raw.data = json.dumps(cmd)
        self.right_pub_cmd_raw.publish(msg_raw)

    # ------------------------------------------------------------------
    # Left arm wrist callback - print orientation
    # ------------------------------------------------------------------
    def _on_left_wrist(self, msg: PoseStamped) -> None:
        quat = msg.pose.orientation
        r = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        self.get_logger().info(
            f"Left wrist yaw: {yaw:.2f}°"
        )
        
        # Map yaw (-180 to 180°) to motor ID 5 range
        # Normalize yaw from [-180, 180] to [0, 1]
        normalized_yaw = (yaw + 180.0) / 360.0
        servo_position = self.left_motor5_range_min + normalized_yaw * (self.left_motor5_range_max - self.left_motor5_range_min)
        
        # Clamp servo position to motor's physical limits
        servo_position = max(self.left_motor5_range_min, min(self.left_motor5_range_max, servo_position))
        servo_position = int(servo_position)
        
        # Publish raw servo command
        cmd = {"5": servo_position}
        msg_raw = String()
        msg_raw.data = json.dumps(cmd)
        self.left_pub_cmd_raw.publish(msg_raw)

    # ------------------------------------------------------------------
    # Right arm gripper callback
    # ------------------------------------------------------------------
    def _on_right_pinch_distance(self, msg: Float32) -> None:
        """Map pinch distance (2-10 cm) to gripper servo position.

        Linear mapping:
        - 2 cm → range_min
        - 10 cm → range_max
        """
        distance_cm = msg.data

        # Clamp distance to [2, 10]
        distance_cm = max(2.0, min(10.0, distance_cm))

        # Linear interpolation
        fraction = (distance_cm - 2.0) / 8.0  # 0.0 to 1.0
        servo_position = self.right_gripper_range_min + fraction * (self.right_gripper_range_max - self.right_gripper_range_min)
        servo_position = int(servo_position)

        # Publish raw servo command
        cmd = {str(self.right_gripper_motor_id): servo_position}
        msg_raw = String()
        msg_raw.data = json.dumps(cmd)
        self.right_pub_cmd_raw.publish(msg_raw)

        self.get_logger().debug(
            f"Right pinch distance: {distance_cm:.2f} cm → Motor {self.right_gripper_motor_id} position: {servo_position}"
        )

    # ------------------------------------------------------------------
    # Left arm gripper callback
    # ------------------------------------------------------------------
    def _on_left_pinch_distance(self, msg: Float32) -> None:
        """Map pinch distance (2-10 cm) to gripper servo position.

        Linear mapping:
        - 2 cm → range_min
        - 10 cm → range_max
        """
        distance_cm = msg.data

        # Clamp distance to [2, 10]
        distance_cm = max(2.0, min(10.0, distance_cm))

        # Linear interpolation
        fraction = (distance_cm - 2.0) / 8.0  # 0.0 to 1.0
        servo_position = self.left_gripper_range_min + fraction * (self.left_gripper_range_max - self.left_gripper_range_min)
        servo_position = int(servo_position)

        # Publish raw servo command
        cmd = {str(self.left_gripper_motor_id): servo_position}
        msg_raw = String()
        msg_raw.data = json.dumps(cmd)
        self.left_pub_cmd_raw.publish(msg_raw)

        self.get_logger().debug(
            f"Left pinch distance: {distance_cm:.2f} cm → Motor {self.left_gripper_motor_id} position: {servo_position}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _publish_joint_command_right(self, joint_deg: np.ndarray) -> None:
        """Publish joint targets as JSON dict on /right_arm/cmd/set_positions."""
        cmd = {name: float(joint_deg[i]) for i, name in enumerate(self.joint_names[:len(joint_deg)])}
        msg = String()
        msg.data = json.dumps(cmd)
        self.right_pub_cmd.publish(msg)

    def _publish_joint_command_left(self, joint_deg: np.ndarray) -> None:
        """Publish joint targets as JSON dict on /left_arm/cmd/set_positions."""
        cmd = {name: float(joint_deg[i]) for i, name in enumerate(self.joint_names[:len(joint_deg)])}
        msg = String()
        msg.data = json.dumps(cmd)
        self.left_pub_cmd.publish(msg)


def main():
    rclpy.init()
    node = SO101TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
