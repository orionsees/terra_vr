#!/usr/bin/env python3
"""so101_vr_teleop.py — VR wrist-to-SO101 teleoperation node.

Subscribes to a VR wrist PoseStamped topic, solves IK with PyBullet using
the real so101_follower.urdf, and publishes degree commands to the arm bridge.

Coordinate mapping (VR → arm frame):
    x_vr  →  -x_arm
    y_vr  →  +z_arm
    z_vr  →  -y_arm

Usage:
    ros2 run arm_vr so101_vr_teleop \
        --ros-args \
        -p arm_side:=right \
        -p vr_pose_topic:=/vr/wrist/right \
        -p scale:=1.0

The URDF is resolved automatically from the arm_vr package:
    <ament_prefix>/share/arm_vr/urdf/so101_follower.urdf
Override with:
    -p urdf_path:=/absolute/path/to/so101_follower.urdf
"""

import argparse
import json
import math
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as pb
import pybullet_data
import rclpy
import rclpy.utilities
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32, String


# ---------------------------------------------------------------------------
# URDF-derived constants  (from so101_follower.urdf)
# ---------------------------------------------------------------------------

# Revolute joints in kinematic order, matching servo IDs 1–6
JOINT_NAMES = [
    "shoulder_pan",   # servo 1  yaw at base
    "shoulder_lift",  # servo 2  pitch at shoulder
    "elbow_flex",     # servo 3  pitch at elbow
    "wrist_flex",     # servo 4  pitch at wrist
    "wrist_roll",     # servo 5  roll at wrist
    "gripper",        # servo 6  jaw open/close
]

# Joint limits from the URDF <limit> tags (radians), converted to degrees
JOINT_LIMITS_DEG = {
    "shoulder_pan":  (math.degrees(-2.055), math.degrees( 2.058)),  # ±~118°
    "shoulder_lift": (math.degrees(-2.018), math.degrees( 2.018)),  # ±~116°
    "elbow_flex":    (math.degrees(-1.653), math.degrees( 1.654)),  # ±~95°
    "wrist_flex":    (math.degrees(-1.786), math.degrees( 1.790)),  # ±~102°
    "wrist_roll":    (math.degrees(-3.194), math.degrees( 4.120)),  # -183°…+236°
    "gripper":       (math.degrees(-0.175), math.degrees( 1.745)),  # -10°…+100°
}

# Joints solved by IK  (gripper is pass-through from VR trigger)
IK_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# End-effector link for IK — gripper_link is the physical tip in the URDF
EE_LINK_NAME = "gripper_link"

# Max Cartesian step per callback (metres) — safety hard clamp
DEFAULT_MAX_DELTA_M = 0.05


# ---------------------------------------------------------------------------
# URDF sanitiser
# ---------------------------------------------------------------------------

def _sanitise_urdf_for_pybullet(urdf_text: str) -> str:
    """Strip xacro directives and package:// mesh paths so PyBullet can parse
    the URDF as plain XML.  Kinematics (joints, origins, axes, limits) are
    preserved exactly; mesh geometry is replaced by a dummy box.
    """
    # Remove <xacro:include .../> self-reference
    urdf_text = re.sub(r'<xacro:include\s[^/]*/>', '', urdf_text)
    urdf_text = re.sub(r'<xacro:include\s[^>]*>', '',  urdf_text)

    # Remove xacro namespace attribute from <robot> tag
    urdf_text = urdf_text.replace(' xmlns:xacro="http://www.ros.org/wiki/xacro"', '')

    # Replace every mesh geometry block with a tiny box
    # PyBullet only needs kinematics; visuals/collisions are irrelevant for IK
    urdf_text = re.sub(
        r'<geometry>\s*<mesh\b[^/]*/>\s*</geometry>',
        '<geometry><box size="0.01 0.01 0.01"/></geometry>',
        urdf_text,
        flags=re.DOTALL,
    )

    return urdf_text


# ---------------------------------------------------------------------------
# PyBullet IK solver
# ---------------------------------------------------------------------------

class SO101IKSolver:
    """Headless (DIRECT mode) PyBullet IK solver for the SO101 arm."""

    def __init__(self, urdf_path: str):
        if not Path(urdf_path).exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        raw   = Path(urdf_path).read_text(encoding="utf-8")
        clean = _sanitise_urdf_for_pybullet(raw)

        # Write sanitised URDF to a temp file PyBullet can open
        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".urdf", mode="w", encoding="utf-8", delete=False
        )
        self._tmp.write(clean)
        self._tmp.flush()

        self._client = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)

        try:
            self._robot = pb.loadURDF(
                self._tmp.name,
                basePosition=[0, 0, 0],
                useFixedBase=True,
                physicsClientId=self._client,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load URDF: {e}\nTemp file: {self._tmp.name}")

        # Build name → PyBullet index maps
        self._joint_idx: dict[str, int] = {}
        self._link_idx:  dict[str, int] = {}
        for i in range(pb.getNumJoints(self._robot, physicsClientId=self._client)):
            info = pb.getJointInfo(self._robot, i, physicsClientId=self._client)
            self._joint_idx[info[1].decode()]  = i   # joint name
            self._link_idx[info[12].decode()]  = i   # child link name

        if EE_LINK_NAME not in self._link_idx:
            raise RuntimeError(
                f"EE link '{EE_LINK_NAME}' not found in URDF. "
                f"Available links: {sorted(self._link_idx)}"
            )
        self._ee_idx = self._link_idx[EE_LINK_NAME]

        # IK joint indices and URDF limit values (radians)
        self._ik_indices: list[int]   = []
        self._lower_rad:  list[float] = []
        self._upper_rad:  list[float] = []
        for name in IK_JOINT_NAMES:
            idx  = self._joint_idx[name]
            info = pb.getJointInfo(self._robot, idx, physicsClientId=self._client)
            self._ik_indices.append(idx)
            self._lower_rad.append(float(info[8]))   # jointLowerLimit
            self._upper_rad.append(float(info[9]))   # jointUpperLimit

        self._joint_ranges = [u - l for l, u in zip(self._lower_rad, self._upper_rad)]
        self._rest_poses   = [0.0] * len(self._ik_indices)

    # ------------------------------------------------------------------

    def solve(
        self,
        target_pos:  list[float],
        target_orn:  Optional[list[float]] = None,
        max_iters:   int   = 200,
        residual_thr: float = 5e-5,
    ) -> dict[str, float]:
        """Run IK and return {joint_name: degrees} for IK_JOINT_NAMES.

        Parameters
        ----------
        target_pos : [x, y, z] in arm/world frame (metres)
        target_orn : [qx, qy, qz, qw]  — None = position-only IK
        """
        kwargs: dict = dict(
            bodyUniqueId         = self._robot,
            endEffectorLinkIndex = self._ee_idx,
            targetPosition       = target_pos,
            lowerLimits          = self._lower_rad,
            upperLimits          = self._upper_rad,
            jointRanges          = self._joint_ranges,
            restPoses            = self._rest_poses,
            maxNumIterations     = max_iters,
            residualThreshold    = residual_thr,
            physicsClientId      = self._client,
        )
        if target_orn is not None:
            kwargs["targetOrientation"] = target_orn

        angles_rad = pb.calculateInverseKinematics(**kwargs)

        result: dict[str, float] = {}
        for name, rad in zip(IK_JOINT_NAMES, angles_rad):
            deg      = math.degrees(rad)
            lo, hi   = JOINT_LIMITS_DEG[name]
            result[name] = max(lo, min(hi, deg))

        # Seed the solver with the current solution → avoids configuration jumps
        for idx, rad in zip(self._ik_indices, angles_rad):
            pb.resetJointState(self._robot, idx, rad, physicsClientId=self._client)

        return result

    def fk_ee_position(self) -> np.ndarray:
        """Return EE world position at the current joint state."""
        state = pb.getLinkState(self._robot, self._ee_idx, physicsClientId=self._client)
        return np.array(state[4])   # worldLinkFramePosition

    def close(self) -> None:
        pb.disconnect(self._client)
        Path(self._tmp.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Single arm controller
# ---------------------------------------------------------------------------

class _ArmController:
    """Manages IK and teleoperation for one arm."""

    def __init__(self, side: str, node: Node, config: dict, urdf_path: str, scale: float, max_delta: float, ctrl_orient: bool):
        self.side = side
        self.node = node
        self._scale = scale
        self._max_delta = max_delta
        self._ctrl_orient = ctrl_orient

        # Load gripper configs
        gripper_config = config["gripper"]
        self._gripper_motor_id = gripper_config["id"]
        self._gripper_range_min = gripper_config["range_min"]
        self._gripper_range_max = gripper_config["range_max"]

        # Wrist roll motor ID 5 config
        self._motor5_range_min = config["wrist_roll"]["range_min"]
        self._motor5_range_max = config["wrist_roll"]["range_max"]

        # Wrist flex motor ID 4 config
        self._motor4_range_min = config["wrist_flex"]["range_min"]
        self._motor4_range_max = config["wrist_flex"]["range_max"]

        # Initialize IK solver
        self._ik = SO101IKSolver(urdf_path)
        self._target_arm: np.ndarray = self._ik.fk_ee_position()
        node.get_logger().info(f"[{side}] IK solver ready, EE: {np.round(self._target_arm, 4)} m")

        # VR reference pose
        self._ref_pos: Optional[np.ndarray] = None
        self._ref_quat: Optional[np.ndarray] = None

        # Gripper state
        self._gripper_deg: float = 0.0
        self._gripper_servo_pos: int = int((self._gripper_range_min + self._gripper_range_max) / 2)  # center position

        # Publishers
        cmd_topic = f"/{side}_arm/cmd/set_positions"
        cmd_raw_topic = f"/{side}_arm/cmd/set_positions_raw"
        qos = QoSProfile(depth=10)
        self._pub = node.create_publisher(String, cmd_topic, qos)
        self._pub_raw = node.create_publisher(String, cmd_raw_topic, qos)

    def send_zero_position(self) -> None:
        """Send all joints to zero position on startup."""
        joints = {name: 0.0 for name in IK_JOINT_NAMES}
        out = String()
        out.data = json.dumps(joints)
        self._pub.publish(out)
        
        # Send gripper to center position (neutral)
        gripper_cmd = String()
        gripper_cmd.data = json.dumps({str(self._gripper_motor_id): self._gripper_servo_pos})
        self._pub_raw.publish(gripper_cmd)
        
        self.node.get_logger().info(f"[{self.side}] Sent initial zero position")

    def on_gripper(self, msg: Float32) -> None:
        """Map VR gripper trigger (0.0-1.0) to servo position."""
        fraction = float(msg.data)  # 0.0 to 1.0
        self._gripper_servo_pos = int(
            self._gripper_range_min + fraction * (self._gripper_range_max - self._gripper_range_min)
        )

    def on_pinch_distance(self, msg: Float32) -> None:
        """Map pinch distance (2-10 cm) to gripper servo position.
        
        Linear mapping:
        - 2 cm → range_min (closed)
        - 10 cm → range_max (open)
        """
        distance_cm = msg.data
        
        # Clamp distance to [2, 10]
        distance_cm = max(2.0, min(10.0, distance_cm))
        
        # Linear interpolation from 2cm to 10cm range
        fraction = (distance_cm - 2.0) / 8.0  # 0.0 to 1.0
        self._gripper_servo_pos = int(
            self._gripper_range_min + fraction * (self._gripper_range_max - self._gripper_range_min)
        )

    def on_vr_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        vr_pos = np.array([p.x, p.y, p.z])
        vr_quat = np.array([q.x, q.y, q.z, q.w])

        # Extract wrist yaw for motor ID 5 control
        r = R.from_quat(vr_quat)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        
        # Motor ID 5 (wrist_roll) — controlled by yaw
        normalized_yaw = (yaw + 180.0) / 360.0
        motor5_position = self._motor5_range_min + normalized_yaw * (self._motor5_range_max - self._motor5_range_min)
        motor5_position = max(self._motor5_range_min, min(self._motor5_range_max, motor5_position))
        motor5_position = int(motor5_position)

        # Motor ID 4 (wrist_flex) — controlled by roll
        normalized_roll = (roll + 180.0) / 360.0
        motor4_position = self._motor4_range_min + normalized_roll * (self._motor4_range_max - self._motor4_range_min)
        motor4_position = max(self._motor4_range_min, min(self._motor4_range_max, motor4_position))
        motor4_position = int(motor4_position)

        # Latch reference on first message
        if self._ref_pos is None:
            self._ref_pos = vr_pos.copy()
            self._ref_quat = vr_quat.copy()
            self.node.get_logger().info(f"[{self.side}] Reference pose latched")
            return

        # Scaled displacement in VR frame
        delta_vr = (vr_pos - self._ref_pos) * self._scale

        if np.linalg.norm(delta_vr) < 1e-6:
            return

        # Remap to arm frame
        delta_arm = np.array([
            -delta_vr[0],
            -delta_vr[2],
            delta_vr[1],
        ])

        # Safety clamp
        norm = np.linalg.norm(delta_arm)
        if norm > self._max_delta:
            delta_arm *= self._max_delta / norm

        # Advance target
        new_target = self._target_arm + delta_arm

        # Optional orientation
        target_orn: Optional[list[float]] = None
        if self._ctrl_orient:
            r_ref = R.from_quat(self._ref_quat)
            r_cur = R.from_quat(vr_quat)
            r_rel = r_cur * r_ref.inv()
            remap = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
            r_arm = R.from_matrix(remap @ r_rel.as_matrix() @ remap.T)
            target_orn = r_arm.as_quat().tolist()

        # Solve IK
        try:
            joints = self._ik.solve(new_target.tolist(), target_orn=target_orn)
        except Exception as exc:
            self.node.get_logger().error(f"[{self.side}] IK failed: {exc}")
            return

        # Commit state
        self._target_arm = new_target
        self._ref_pos = vr_pos.copy()
        self._ref_quat = vr_quat.copy()

        # Publish IK command (without gripper)
        out = String()
        out.data = json.dumps({k: round(v, 3) for k, v in joints.items()})
        self._pub.publish(out)

        # Publish gripper servo position + wrist_roll + wrist_flex separately via set_positions_raw
        raw_cmd = {
            str(self._gripper_motor_id): self._gripper_servo_pos,
            "4": motor4_position,
            "5": motor5_position
        }
        raw_servo_cmd = String()
        raw_servo_cmd.data = json.dumps(raw_cmd)
        self._pub_raw.publish(raw_servo_cmd)

    def close(self) -> None:
        self._ik.close()


# ---------------------------------------------------------------------------
# VR teleoperation node
# ---------------------------------------------------------------------------

class VRTeleopNode(Node):
    """Dual-arm VR teleoperation node.
    
    Subscriptions:
        /left_wrist, /right_wrist          geometry_msgs/PoseStamped   VR wrist poses
        /{side}_hand/pinch_distance        std_msgs/Float32            pinch distances
        /{side}_arm/vr/gripper             std_msgs/Float32            gripper triggers

    Publications:
        /{side}_arm/cmd/set_positions      std_msgs/String             IK joint commands
        /{side}_arm/cmd/set_positions_raw  std_msgs/String             raw servo commands
    """

    def __init__(self, args):
        super().__init__("so101_vr_teleop")

        # ---- ROS parameters ----
        self.declare_parameter("urdf_path",           args.urdf_path)
        self.declare_parameter("scale",               args.scale)
        self.declare_parameter("max_delta_m",         args.max_delta_m)
        self.declare_parameter("control_orientation", args.control_orientation)

        urdf_path      = self.get_parameter("urdf_path").value
        scale          = float(self.get_parameter("scale").value)
        max_delta      = float(self.get_parameter("max_delta_m").value)
        ctrl_orient    = bool(self.get_parameter("control_orientation").value)

        # ---- Load configs for both arms ----
        package_share_dir = get_package_share_directory('arm_vr')
        with open(os.path.join(package_share_dir, 'config', 'left_arm.json'), 'r') as f:
            left_config = json.load(f)
        with open(os.path.join(package_share_dir, 'config', 'right_arm.json'), 'r') as f:
            right_config = json.load(f)

        # ---- Resolve URDF ----
        if not urdf_path:
            urdf_path = str(Path(package_share_dir) / "urdf" / "so101_follower_pybullet.urdf")
        self.get_logger().info(f"URDF: {urdf_path}")

        # ---- Create arm controllers ----
        self.left_arm = _ArmController("left", self, left_config, urdf_path, scale, max_delta, ctrl_orient)
        self.right_arm = _ArmController("right", self, right_config, urdf_path, scale, max_delta, ctrl_orient)

        # Send initial zero positions
        self.left_arm.send_zero_position()
        self.right_arm.send_zero_position()

        # ---- Subscriptions ----
        qos = QoSProfile(depth=10)
        self.create_subscription(PoseStamped, "/left_wrist", self._on_left_wrist, qos)
        self.create_subscription(PoseStamped, "/right_wrist", self._on_right_wrist, qos)
        self.create_subscription(Float32, "/left_hand/pinch_distance", self._on_left_pinch_distance, qos)
        self.create_subscription(Float32, "/right_hand/pinch_distance", self._on_right_pinch_distance, qos)
        self.create_subscription(Float32, "/left_arm/vr/gripper", self._on_left_gripper, qos)
        self.create_subscription(Float32, "/right_arm/vr/gripper", self._on_right_gripper, qos)

        self.get_logger().info(
            f"Dual-arm teleop active | scale={scale}  max_step={max_delta} m  orient={ctrl_orient}"
        )

    # ------ Left arm callbacks ------
    def _on_left_wrist(self, msg: PoseStamped) -> None:
        self.left_arm.on_vr_pose(msg)

    def _on_left_gripper(self, msg: Float32) -> None:
        self.left_arm.on_gripper(msg)

    def _on_left_pinch_distance(self, msg: Float32) -> None:
        self.left_arm.on_pinch_distance(msg)

    # ------ Right arm callbacks ------
    def _on_right_wrist(self, msg: PoseStamped) -> None:
        self.right_arm.on_vr_pose(msg)

    def _on_right_gripper(self, msg: Float32) -> None:
        self.right_arm.on_gripper(msg)

    def _on_right_pinch_distance(self, msg: Float32) -> None:
        self.right_arm.on_pinch_distance(msg)

    def destroy_node(self):
        self.left_arm.close()
        self.right_arm.close()
        return super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SO101 Dual-Arm VR Teleoperation Node")
    parser.add_argument("--urdf-path", default="",
                        help="Path to so101_follower.urdf  "
                             "(default: resolved from arm_vr ament share)")
    parser.add_argument("--scale",               type=float, default=1.0)
    parser.add_argument("--max-delta-m",         type=float, default=DEFAULT_MAX_DELTA_M)
    parser.add_argument("--control-orientation", action="store_true",
                        help="Also send wrist orientation to IK (tune position first)")

    ros_args = rclpy.utilities.remove_ros_args(sys.argv[1:])
    args = parser.parse_args(ros_args)

    rclpy.init()
    node = None
    try:
        node = VRTeleopNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()