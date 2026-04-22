#!/usr/bin/env python3
"""Teleoperation node: maps VR wrist poses to dual SO101 arms via IK.

Subscribes to /right_wrist and /left_wrist (PoseStamped) from TCP listener and publishes
joint commands to /right_arm/cmd/set_positions and /left_arm/cmd/set_positions
(String/JSON) via the SO101 bridges.

Uses incremental (frame-to-frame) delta control for each arm independently:
each cycle computes the displacement since the *previous* frame rather than 
from a fixed origin. This avoids drift, enables clutch/re-centering, and keeps IK stable.

Coordinate mapping (VR → SO101):
    x_vr  → -x_arm
    y_vr  → +z_arm
    z_vr  → -y_arm

Launch after so101_bridge and tcp_wireless are running:
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

from so101_kinematics import SO101Kinematics, EndEffectorPose


class SO101TeleopNode(Node):
    def __init__(self):
        super().__init__("so101_teleop")

        # --- Parameters ---
        package_share_dir = get_package_share_directory('arm_vr')
        urdf_path = os.path.join(package_share_dir, 'urdf', 'so101_follower.urdf')
        right_config_path = os.path.join(package_share_dir, 'config', 'right_arm.json')
        left_config_path = os.path.join(package_share_dir, 'config', 'left_arm.json')
        
        self.declare_parameter("target_frame", "gripper_link")
        self.declare_parameter("scale", 1.0)  # displacement multiplier
        self.declare_parameter(
            "joint_names",
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        )
        target_frame = self.get_parameter("target_frame").value
        joint_names = list(self.get_parameter("joint_names").value)
        self.scale = self.get_parameter("scale").value

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

        # --- Kinematics (one per arm) ---
        self.right_kin = SO101Kinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=joint_names,
        )
        self.left_kin = SO101Kinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=joint_names,
        )
        self.n_joints = 6

        # --- State for right arm ---
        self.right_current_joint_deg = np.zeros(self.n_joints, dtype=float)
        self.right_home_ee = self.right_kin.forward_kinematics(self.right_current_joint_deg)
        self.right_desired_ee = self.right_home_ee.copy()
        self.right_prev_wrist_pos: np.ndarray | None = None

        # --- State for left arm ---
        self.left_current_joint_deg = np.zeros(self.n_joints, dtype=float)
        self.left_home_ee = self.left_kin.forward_kinematics(self.left_current_joint_deg)
        self.left_desired_ee = self.left_home_ee.copy()
        self.left_prev_wrist_pos: np.ndarray | None = None

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
            "SO101 dual-arm teleop node ready (incremental delta mode)"
        )
        self.get_logger().info(
            f"Right gripper: motor {self.right_gripper_motor_id} [{self.right_gripper_range_min}-{self.right_gripper_range_max}]"
        )
        self.get_logger().info(
            f"Left gripper: motor {self.left_gripper_motor_id} [{self.left_gripper_range_min}-{self.left_gripper_range_max}]"
        )

    # ------------------------------------------------------------------
    # Coordinate mapping: VR → SO101
    # ------------------------------------------------------------------
    @staticmethod
    def _vr_displacement_to_arm(d_vr: np.ndarray) -> np.ndarray:
        """Map a VR displacement vector to SO101 arm frame.

        x_vr → -x_arm
        y_vr → +z_arm
        z_vr → -y_arm
        """
        return np.array([
            -d_vr[0],   # arm x = -vr x
            -d_vr[2],   # arm y = -vr z
             d_vr[1],   # arm z = +vr y
        ], dtype=float)

    # ------------------------------------------------------------------
    # Right arm callbacks
    # ------------------------------------------------------------------
    def _on_right_wrist(self, msg: PoseStamped) -> None:
        vr_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)

        # First frame: just store as reference, no motion
        if self.right_prev_wrist_pos is None:
            self.right_prev_wrist_pos = vr_pos.copy()
            self.get_logger().info(
                f"Right arm: initial wrist position captured: {self.right_prev_wrist_pos}"
            )
            return

        # Step 1: frame-to-frame delta in VR space
        d_vr = vr_pos - self.right_prev_wrist_pos

        # Step 2: map to arm frame and apply scale
        d_arm = self._vr_displacement_to_arm(d_vr) * self.scale

        # Step 3: accumulate into desired EE pose (position only, keep orientation)
        self.right_desired_ee[:3, 3] += d_arm

        # Step 4: solve IK from current joint state
        try:
            target_deg = self.right_kin.inverse_kinematics(
                current_joint_pos_deg=self.right_current_joint_deg,
                desired_ee_pose=self.right_desired_ee,
            )
        except Exception as exc:
            self.get_logger().warn(f"Right arm IK failed: {exc}")
            # Still update prev so we don't accumulate a huge jump next frame
            self.right_prev_wrist_pos = vr_pos.copy()
            return

        # Step 5: publish and update state
        self.right_current_joint_deg = target_deg[: self.n_joints].copy()
        self._publish_joint_command_right(self.right_current_joint_deg)

        # Update previous wrist position for next delta
        self.right_prev_wrist_pos = vr_pos.copy()

    def _on_left_wrist(self, msg: PoseStamped) -> None:
        vr_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)

        # First frame: just store as reference, no motion
        if self.left_prev_wrist_pos is None:
            self.left_prev_wrist_pos = vr_pos.copy()
            self.get_logger().info(
                f"Left arm: initial wrist position captured: {self.left_prev_wrist_pos}"
            )
            return

        # Step 1: frame-to-frame delta in VR space
        d_vr = vr_pos - self.left_prev_wrist_pos

        # Step 2: map to arm frame and apply scale
        d_arm = self._vr_displacement_to_arm(d_vr) * self.scale

        # Step 3: accumulate into desired EE pose (position only, keep orientation)
        self.left_desired_ee[:3, 3] += d_arm

        # Step 4: solve IK from current joint state
        try:
            target_deg = self.left_kin.inverse_kinematics(
                current_joint_pos_deg=self.left_current_joint_deg,
                desired_ee_pose=self.left_desired_ee,
            )
        except Exception as exc:
            self.get_logger().warn(f"Left arm IK failed: {exc}")
            # Still update prev so we don't accumulate a huge jump next frame
            self.left_prev_wrist_pos = vr_pos.copy()
            return

        # Step 5: publish and update state
        self.left_current_joint_deg = target_deg[: self.n_joints].copy()
        self._publish_joint_command_left(self.left_current_joint_deg)

        # Update previous wrist position for next delta
        self.left_prev_wrist_pos = vr_pos.copy()

    # ------------------------------------------------------------------
    # Right arm gripper callback
    # ------------------------------------------------------------------
    def _on_right_pinch_distance(self, msg: Float32) -> None:
        """Map pinch distance (0-10 cm) to gripper servo position.

        Linear mapping:
        - 0 cm → range_min
        - 10 cm → range_max
        """
        distance_cm = msg.data

        # Clamp distance to [0, 10]
        distance_cm = max(0.0, min(10.0, distance_cm))

        # Linear interpolation
        fraction = distance_cm / 10.0  # 0.0 to 1.0
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
        """Map pinch distance (0-10 cm) to gripper servo position.

        Linear mapping:
        - 0 cm → range_min
        - 10 cm → range_max
        """
        distance_cm = msg.data

        # Clamp distance to [0, 10]
        distance_cm = max(0.0, min(10.0, distance_cm))

        # Linear interpolation
        fraction = distance_cm / 10.0  # 0.0 to 1.0
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
        """Publish joint targets as JSON dict on /right_arm/so101/cmd/set_positions."""
        cmd = self.right_kin.vector_to_joints_dict(joint_deg, suffix="")
        msg = String()
        msg.data = json.dumps(cmd)
        self.right_pub_cmd.publish(msg)

    def _publish_joint_command_left(self, joint_deg: np.ndarray) -> None:
        """Publish joint targets as JSON dict on /left_arm/so101/cmd/set_positions."""
        cmd = self.left_kin.vector_to_joints_dict(joint_deg, suffix="")
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
