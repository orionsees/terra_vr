#!/usr/bin/env python3
"""Teleoperation node: maps right-wrist VR pose to SO101 arm end-effector via IK.

Subscribes to /right_wrist (PoseStamped) from the TCP listener and publishes
joint commands to /so101/cmd/set_positions (String/JSON) via the SO101 bridge.

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
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

from so101_kinematics import SO101Kinematics, EndEffectorPose


class SO101TeleopNode(Node):
    def __init__(self):
        super().__init__("so101_teleop")

        # --- Parameters ---
        package_share_dir = get_package_share_directory('arm_vr')
        urdf_path = os.path.join(package_share_dir, 'urdf', 'so101_follower.urdf')
        self.declare_parameter("target_frame", "jaw_link")
        self.declare_parameter("scale", 1.0)  # displacement multiplier
        self.declare_parameter(
            "joint_names",
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        )
        target_frame = self.get_parameter("target_frame").value
        joint_names = list(self.get_parameter("joint_names").value)
        self.scale = self.get_parameter("scale").value

        # --- Kinematics ---
        self.kin = SO101Kinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=joint_names,
        )
        self.n_joints = 6

        # --- State ---
        # Start all joints at 0°
        self.current_joint_deg = np.zeros(self.n_joints, dtype=float)
        self.home_ee = self.kin.forward_kinematics(self.current_joint_deg)  # 4x4
        self.home_ee_pos = self.home_ee[:3, 3].copy()  # (x, y, z) of EE at home

        self.initial_wrist_pos: np.ndarray | None = None  # first VR wrist position

        # --- Publishers / Subscribers ---
        self.pub_cmd = self.create_publisher(String, "/so101/cmd/set_positions", 10)

        self.create_subscription(
            PoseStamped,
            "/right_wrist",
            self._on_right_wrist,
            10,
        )

        # Send the arm to home (all zeros) on startup
        self._publish_joint_command(self.current_joint_deg)

        self.get_logger().info(
            "SO101 teleop node ready — waiting for first /right_wrist pose…"
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
    # Callback
    # ------------------------------------------------------------------
    def _on_right_wrist(self, msg: PoseStamped) -> None:
        vr_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)

        # Capture the very first wrist position as the reference origin
        if self.initial_wrist_pos is None:
            self.initial_wrist_pos = vr_pos.copy()
            self.get_logger().info(
                f"Initial wrist position captured: {self.initial_wrist_pos}"
            )
            return

        # Step 1: displacement in VR space
        d_vr = vr_pos - self.initial_wrist_pos

        # Step 2: map to arm frame and apply scale
        d_arm = self._vr_displacement_to_arm(d_vr) * self.scale

        # Step 3: desired EE position = home EE position + mapped displacement
        desired_ee_pos = self.home_ee_pos + d_arm

        # Build desired 4×4 pose (keep home orientation, only change position)
        desired_ee = self.home_ee.copy()
        desired_ee[:3, 3] = desired_ee_pos

        # Step 4: solve IK from current joint state
        try:
            target_deg = self.kin.inverse_kinematics(
                current_joint_pos_deg=self.current_joint_deg,
                desired_ee_pose=desired_ee,
            )
        except Exception as exc:
            self.get_logger().warn(f"IK failed: {exc}")
            return

        # Step 5: publish
        self.current_joint_deg = target_deg[: self.n_joints].copy()
        self._publish_joint_command(self.current_joint_deg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _publish_joint_command(self, joint_deg: np.ndarray) -> None:
        """Publish joint targets as JSON dict on /so101/cmd/set_positions.

        Keys use the '<joint_name>.pos' convention expected by the bridge.
        """
        cmd = self.kin.vector_to_joints_dict(joint_deg, suffix="")
        msg = String()
        msg.data = json.dumps(cmd)
        self.pub_cmd.publish(msg)


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