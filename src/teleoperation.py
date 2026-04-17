#!/usr/bin/env python3
"""Teleoperation node: maps right-wrist VR pose to SO101 arm end-effector via IK.

Subscribes to /right_wrist (PoseStamped) from the TCP listener and publishes
joint commands to /so101/cmd/set_positions (String/JSON) via the SO101 bridge.

Uses incremental (frame-to-frame) delta control: each cycle computes the
displacement since the *previous* frame rather than from a fixed origin.
This avoids drift, enables clutch/re-centering, and keeps IK stable.

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
        config_path = os.path.join(package_share_dir, 'config', 'right_arm.json')
        self.declare_parameter("target_frame", "gripper_link")
        self.declare_parameter("scale", 1.0)  # displacement multiplier
        self.declare_parameter(
            "joint_names",
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        )
        target_frame = self.get_parameter("target_frame").value
        joint_names = list(self.get_parameter("joint_names").value)
        self.scale = self.get_parameter("scale").value

        # Load gripper servo config
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.gripper_config = config["gripper"]
        self.gripper_range_min = self.gripper_config["range_min"]
        self.gripper_range_max = self.gripper_config["range_max"]
        self.gripper_motor_id = self.gripper_config["id"]

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

        # Desired EE pose starts at home — updated incrementally each frame
        self.desired_ee = self.home_ee.copy()

        # Previous wrist position (set on first message)
        self.prev_wrist_pos: np.ndarray | None = None

        # --- Publishers / Subscribers ---
        self.pub_cmd = self.create_publisher(String, "/so101/cmd/set_positions", 10)
        self.pub_cmd_raw = self.create_publisher(String, "/so101/cmd/set_positions_raw", 10)

        self.create_subscription(
            PoseStamped,
            "/right_wrist",
            self._on_right_wrist,
            10,
        )

        self.create_subscription(
            Float32,
            "/right_hand/pinch_distance",
            self._on_pinch_distance,
            10,
        )

        # Send the arm to home (all zeros) on startup
        self._publish_joint_command(self.current_joint_deg)

        self.get_logger().info(
            "SO101 teleop node ready (incremental delta mode) — waiting for first /right_wrist pose…"
        )
        self.get_logger().info(
            f"Gripper control enabled: pinch_distance [0-10cm] → motor {self.gripper_motor_id} position [{self.gripper_range_min}-{self.gripper_range_max}]"
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

        # First frame: just store as reference, no motion
        if self.prev_wrist_pos is None:
            self.prev_wrist_pos = vr_pos.copy()
            self.get_logger().info(
                f"Initial wrist position captured: {self.prev_wrist_pos}"
            )
            return

        # Step 1: frame-to-frame delta in VR space
        d_vr = vr_pos - self.prev_wrist_pos

        # Step 2: map to arm frame and apply scale
        d_arm = self._vr_displacement_to_arm(d_vr) * self.scale

        # Step 3: accumulate into desired EE pose (position only, keep orientation)
        self.desired_ee[:3, 3] += d_arm

        # Step 4: solve IK from current joint state
        try:
            target_deg = self.kin.inverse_kinematics(
                current_joint_pos_deg=self.current_joint_deg,
                desired_ee_pose=self.desired_ee,
            )
        except Exception as exc:
            self.get_logger().warn(f"IK failed: {exc}")
            # Still update prev so we don't accumulate a huge jump next frame
            self.prev_wrist_pos = vr_pos.copy()
            return

        # Step 5: publish and update state
        self.current_joint_deg = target_deg[: self.n_joints].copy()
        self._publish_joint_command(self.current_joint_deg)

        # Update previous wrist position for next delta
        self.prev_wrist_pos = vr_pos.copy()

    # ------------------------------------------------------------------
    # Pinch distance callback for gripper control
    # ------------------------------------------------------------------
    def _on_pinch_distance(self, msg: Float32) -> None:
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
        servo_position = self.gripper_range_min + fraction * (self.gripper_range_max - self.gripper_range_min)
        servo_position = int(servo_position)

        # Publish raw servo command
        cmd = {str(self.gripper_motor_id): servo_position}
        msg_raw = String()
        msg_raw.data = json.dumps(cmd)
        self.pub_cmd_raw.publish(msg_raw)

        self.get_logger().debug(
            f"Pinch distance: {distance_cm:.2f} cm → Motor {self.gripper_motor_id} position: {servo_position}"
        )

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