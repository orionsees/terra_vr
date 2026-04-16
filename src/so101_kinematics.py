#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-file SO101 forward/inverse kinematics helpers.

This module keeps kinematics in one place and uses a URDF as the single source
of geometry (link lengths, joint axes, limits, and frame hierarchy).

Typical usage:

    kin = SO101Kinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    )

    # FK from joint vector (degrees)
    t_ee = kin.forward_kinematics(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

    # IK to reach desired 4x4 pose
    q_target = kin.inverse_kinematics(np.array([0.0, 0.0, 0.0, 0.0, 0.0]), t_ee)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence
from xml.etree import ElementTree as ET

import numpy as np


def _local_name(tag: str) -> str:
    """Return XML local tag name without namespace prefix."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _find_child(elem: ET.Element, tag_name: str) -> ET.Element | None:
    """Find first direct child by local tag name."""
    for child in list(elem):
        if _local_name(child.tag) == tag_name:
            return child
    return None


def _parse_vec3(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    """Parse URDF xyz/rpy/axis vector strings into float numpy arrays."""
    if value is None:
        return np.array(default, dtype=float)
    parts = [float(p) for p in value.replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3-vector, got {value!r}")
    return np.array(parts, dtype=float)


def _skew(v: np.ndarray) -> np.ndarray:
    """Build skew-symmetric matrix from a 3-vector."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert URDF rpy to rotation matrix (Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=float,
    )
    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )
    rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rz @ ry @ rx


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotation vector to rotation matrix."""
    rv = np.asarray(rotvec, dtype=float)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3, dtype=float)
    axis = rv / theta
    k = _skew(axis)
    return np.eye(3, dtype=float) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def _matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle rotation vector."""
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got shape={r.shape}")

    trace = float(np.trace(r))
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    theta = math.acos(cos_theta)

    if theta < 1e-12:
        return np.zeros(3, dtype=float)

    if abs(math.pi - theta) < 1e-6:
        # Robust path close to pi.
        diag = np.diag(r)
        axis = np.array(
            [
                math.sqrt(max(0.0, (diag[0] + 1.0) * 0.5)),
                math.sqrt(max(0.0, (diag[1] + 1.0) * 0.5)),
                math.sqrt(max(0.0, (diag[2] + 1.0) * 0.5)),
            ],
            dtype=float,
        )

        # Recover signs from off-diagonal terms.
        if r[2, 1] - r[1, 2] < 0.0:
            axis[0] = -axis[0]
        if r[0, 2] - r[2, 0] < 0.0:
            axis[1] = -axis[1]
        if r[1, 0] - r[0, 1] < 0.0:
            axis[2] = -axis[2]

        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            axis = axis / n
        return axis * theta

    v = np.array(
        [
            r[2, 1] - r[1, 2],
            r[0, 2] - r[2, 0],
            r[1, 0] - r[0, 1],
        ],
        dtype=float,
    )
    axis = v / (2.0 * math.sin(theta))
    return axis * theta


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build 4x4 transform from 3x3 rotation and 3-vector translation."""
    t = np.eye(4, dtype=float)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix from unit axis and angle."""
    ax = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(ax))
    if n < 1e-12:
        raise ValueError("Joint axis norm is near zero")
    ax = ax / n
    return _rotvec_to_matrix(ax * angle)


@dataclass(frozen=True)
class _URDFJoint:
    """Minimal URDF joint info for FK/IK."""

    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_transform: np.ndarray
    axis: np.ndarray
    lower_limit: float | None
    upper_limit: float | None


@dataclass(frozen=True)
class EndEffectorPose:
    """End-effector pose represented as translation + rotation vector."""

    x: float
    y: float
    z: float
    wx: float
    wy: float
    wz: float

    def as_matrix(self) -> np.ndarray:
        """Build a 4x4 transform matrix from pose fields."""
        rotation = _rotvec_to_matrix(np.array([self.wx, self.wy, self.wz], dtype=float))
        translation = np.array([self.x, self.y, self.z], dtype=float)
        return _make_transform(rotation, translation)

    @classmethod
    def from_matrix(cls, transform: np.ndarray) -> "EndEffectorPose":
        """Create pose fields from a 4x4 transform matrix."""
        t = np.asarray(transform, dtype=float)
        if t.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 transform matrix, got shape={t.shape}")

        tw = _matrix_to_rotvec(t[:3, :3])
        return cls(
            x=float(t[0, 3]),
            y=float(t[1, 3]),
            z=float(t[2, 3]),
            wx=float(tw[0]),
            wy=float(tw[1]),
            wz=float(tw[2]),
        )


class SO101Kinematics:
    """Forward/inverse kinematics for SO101 from URDF (self-contained backend)."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: Sequence[str] | None = None,
    ):
        resolved_urdf_path = Path(urdf_path).expanduser()
        if not resolved_urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {resolved_urdf_path}")

        self.urdf_path = str(resolved_urdf_path)
        self.target_frame_name = target_frame_name

        self._joints_by_name, self._joints_by_child_link = self._parse_urdf(self.urdf_path)
        self._chain_joints = self._build_chain_to_target(self.target_frame_name)
        chain_joint_names = [j.name for j in self._chain_joints if j.joint_type in {"revolute", "continuous", "prismatic"}]

        if joint_names is None:
            self.joint_names = list(chain_joint_names)
        else:
            self.joint_names = list(joint_names)

        if not self.joint_names:
            raise ValueError("joint_names is empty. Provide at least one joint name.")

        missing_joint_names = [name for name in self.joint_names if name not in self._joints_by_name]
        if missing_joint_names:
            raise ValueError(
                "These joints are not present in the URDF: "
                f"{missing_joint_names}."
            )

        not_in_chain = [name for name in self.joint_names if name not in chain_joint_names]
        if not_in_chain:
            raise ValueError(
                "These joints are not on the chain from root to target frame "
                f"'{self.target_frame_name}': {not_in_chain}. Chain joints are: {chain_joint_names}"
            )

        self._joint_limits: dict[str, tuple[float | None, float | None]] = {
            name: (self._joints_by_name[name].lower_limit, self._joints_by_name[name].upper_limit)
            for name in self.joint_names
        }

        # IK tuning defaults (rad units internally).
        self._ik_max_iterations = 120
        self._ik_position_tolerance_m = 1e-4
        self._ik_orientation_tolerance_rad = 2e-3
        self._ik_fd_epsilon_rad = 1e-6
        self._ik_damping = 5e-3
        self._ik_max_step_rad = 0.25
        self._ik_joint_step_tolerance_rad = 1e-7

    @staticmethod
    def _parse_urdf(urdf_path: str) -> tuple[dict[str, _URDFJoint], dict[str, _URDFJoint]]:
        """Parse joints needed for kinematics from a URDF/Xacro XML file."""
        try:
            root = ET.parse(urdf_path).getroot()
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse URDF/XML at '{urdf_path}': {e}") from e

        joints_by_name: dict[str, _URDFJoint] = {}
        joints_by_child_link: dict[str, _URDFJoint] = {}

        for elem in root.iter():
            if _local_name(elem.tag) != "joint":
                continue

            name = elem.attrib.get("name")
            joint_type = elem.attrib.get("type", "fixed")
            if not name:
                continue

            parent_elem = _find_child(elem, "parent")
            child_elem = _find_child(elem, "child")
            if parent_elem is None or child_elem is None:
                continue

            parent_link = parent_elem.attrib.get("link")
            child_link = child_elem.attrib.get("link")
            if not parent_link or not child_link:
                continue

            origin_elem = _find_child(elem, "origin")
            xyz = _parse_vec3(
                origin_elem.attrib.get("xyz") if origin_elem is not None else None,
                (0.0, 0.0, 0.0),
            )
            rpy = _parse_vec3(
                origin_elem.attrib.get("rpy") if origin_elem is not None else None,
                (0.0, 0.0, 0.0),
            )
            origin_transform = _make_transform(_rpy_to_matrix(rpy[0], rpy[1], rpy[2]), xyz)

            axis_elem = _find_child(elem, "axis")
            axis = _parse_vec3(
                axis_elem.attrib.get("xyz") if axis_elem is not None else None,
                (1.0, 0.0, 0.0),
            )
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1e-12:
                axis = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                axis = axis / axis_norm

            lower_limit = None
            upper_limit = None
            limit_elem = _find_child(elem, "limit")
            if limit_elem is not None:
                lower_raw = limit_elem.attrib.get("lower")
                upper_raw = limit_elem.attrib.get("upper")
                if lower_raw is not None:
                    lower_limit = float(lower_raw)
                if upper_raw is not None:
                    upper_limit = float(upper_raw)

            joint = _URDFJoint(
                name=name,
                joint_type=joint_type,
                parent_link=parent_link,
                child_link=child_link,
                origin_transform=origin_transform,
                axis=axis,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            )

            joints_by_name[name] = joint
            joints_by_child_link[child_link] = joint

        if not joints_by_name:
            raise ValueError(f"No joints found in URDF/XML '{urdf_path}'")

        return joints_by_name, joints_by_child_link

    def _build_chain_to_target(self, target_frame_name: str) -> list[_URDFJoint]:
        """Build ordered joint chain from root to target link frame."""
        # In this lightweight implementation, target frame is expected to be a link name.
        if target_frame_name not in self._joints_by_child_link:
            all_child_links = sorted(self._joints_by_child_link.keys())
            raise ValueError(
                f"Target frame '{target_frame_name}' was not found as a link in the URDF chain. "
                f"Available child links include: {all_child_links}"
            )

        chain: list[_URDFJoint] = []
        current_link = target_frame_name
        while current_link in self._joints_by_child_link:
            joint = self._joints_by_child_link[current_link]
            chain.append(joint)
            current_link = joint.parent_link

        chain.reverse()
        return chain

    def _coerce_joint_vector_radians(self, joint_pos_rad: Sequence[float] | np.ndarray) -> np.ndarray:
        q_rad = np.asarray(joint_pos_rad, dtype=float)
        if q_rad.ndim != 1:
            raise ValueError(f"Expected a 1-D joint vector, got shape={q_rad.shape}")
        if q_rad.size < len(self.joint_names):
            raise ValueError(f"Expected at least {len(self.joint_names)} joints, got {q_rad.size}.")
        return q_rad

    def _clip_to_joint_limits(self, q_rad: np.ndarray) -> np.ndarray:
        for i, joint_name in enumerate(self.joint_names):
            lower, upper = self._joint_limits[joint_name]
            if lower is not None:
                q_rad[i] = max(q_rad[i], lower)
            if upper is not None:
                q_rad[i] = min(q_rad[i], upper)
        return q_rad

    def _fk_from_control_joints_radians(self, q_control_rad: np.ndarray) -> np.ndarray:
        """Compute FK from control joints only; non-controlled joints default to zero."""
        q_map = {name: float(q_control_rad[i]) for i, name in enumerate(self.joint_names)}

        t = np.eye(4, dtype=float)
        for joint in self._chain_joints:
            t = t @ joint.origin_transform
            q = q_map.get(joint.name, 0.0)

            if joint.joint_type in {"revolute", "continuous"}:
                r = _rotation_about_axis(joint.axis, q)
                t = t @ _make_transform(r, np.zeros(3, dtype=float))
            elif joint.joint_type == "prismatic":
                t = t @ _make_transform(np.eye(3, dtype=float), joint.axis * q)
            elif joint.joint_type == "fixed":
                pass
            else:
                # Unsupported joint types are treated as fixed to stay robust.
                pass

        return t

    def _coerce_joint_vector(self, joint_pos_deg: Sequence[float] | np.ndarray) -> np.ndarray:
        q_deg = np.asarray(joint_pos_deg, dtype=float)
        if q_deg.ndim != 1:
            raise ValueError(f"Expected a 1-D joint vector, got shape={q_deg.shape}")
        if q_deg.size < len(self.joint_names):
            raise ValueError(
                f"Expected at least {len(self.joint_names)} joints, got {q_deg.size}."
            )
        return q_deg

    def forward_kinematics(self, joint_pos_deg: Sequence[float] | np.ndarray) -> np.ndarray:
        """Compute EE 4x4 pose from joint positions in degrees."""
        q_deg = self._coerce_joint_vector(joint_pos_deg)
        q_rad = np.deg2rad(q_deg[: len(self.joint_names)])
        q_rad = self._clip_to_joint_limits(q_rad)
        return self._fk_from_control_joints_radians(q_rad)

    def inverse_kinematics(
        self,
        current_joint_pos_deg: Sequence[float] | np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        preserve_extra_joints: bool = True,
    ) -> np.ndarray:
        """Compute joint positions in degrees that best match a desired EE 4x4 pose."""
        q_curr_deg = self._coerce_joint_vector(current_joint_pos_deg)
        t_des = np.asarray(desired_ee_pose, dtype=float)
        if t_des.shape != (4, 4):
            raise ValueError(f"Expected desired_ee_pose shape (4, 4), got {t_des.shape}")

        if position_weight < 0.0 or orientation_weight < 0.0:
            raise ValueError("position_weight and orientation_weight must be non-negative")

        n = len(self.joint_names)
        q_solution_rad = np.deg2rad(q_curr_deg[:n]).astype(float)
        q_solution_rad = self._clip_to_joint_limits(q_solution_rad)

        for _ in range(self._ik_max_iterations):
            t_curr = self._fk_from_control_joints_radians(q_solution_rad)

            pos_err = t_des[:3, 3] - t_curr[:3, 3]
            r_err = t_des[:3, :3] @ t_curr[:3, :3].T
            ori_err = _matrix_to_rotvec(r_err)

            if (
                float(np.linalg.norm(pos_err)) < self._ik_position_tolerance_m
                and float(np.linalg.norm(ori_err)) < self._ik_orientation_tolerance_rad
            ):
                break

            jac = np.zeros((6, n), dtype=float)
            eps = self._ik_fd_epsilon_rad
            for i in range(n):
                q_perturbed = np.array(q_solution_rad, copy=True)
                q_perturbed[i] += eps
                t_perturbed = self._fk_from_control_joints_radians(q_perturbed)

                dp = (t_perturbed[:3, 3] - t_curr[:3, 3]) / eps
                dr = t_perturbed[:3, :3] @ t_curr[:3, :3].T
                dw = _matrix_to_rotvec(dr) / eps
                jac[:3, i] = dp
                jac[3:, i] = dw

            weighted_jac = np.array(jac, copy=True)
            weighted_jac[:3, :] *= position_weight
            weighted_jac[3:, :] *= orientation_weight

            weighted_error = np.concatenate(
                [
                    pos_err * position_weight,
                    ori_err * orientation_weight,
                ]
            )

            damping = self._ik_damping
            lhs = weighted_jac.T @ weighted_jac + (damping**2) * np.eye(n, dtype=float)
            rhs = weighted_jac.T @ weighted_error
            try:
                dq = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            step_norm = float(np.linalg.norm(dq))
            if step_norm > self._ik_max_step_rad and step_norm > 0.0:
                dq *= self._ik_max_step_rad / step_norm

            q_solution_rad += dq
            q_solution_rad = self._clip_to_joint_limits(q_solution_rad)

            if step_norm < self._ik_joint_step_tolerance_rad:
                break

        q_solution_deg = np.rad2deg(q_solution_rad)

        if preserve_extra_joints and q_curr_deg.size > len(self.joint_names):
            result = np.array(q_curr_deg, copy=True)
            result[: len(self.joint_names)] = q_solution_deg
            return result

        return q_solution_deg

    def joints_dict_to_vector(
        self,
        joints: Mapping[str, float],
        suffix: str = ".pos",
    ) -> np.ndarray:
        """Extract ordered joint vector from a dict with keys like '<joint>.pos'."""
        try:
            return np.array([float(joints[f"{name}{suffix}"]) for name in self.joint_names], dtype=float)
        except KeyError as e:
            raise KeyError(
                f"Missing joint key '{e.args[0]}'. Expected keys for joint_names={self.joint_names} "
                f"with suffix='{suffix}'."
            ) from e

    def vector_to_joints_dict(
        self,
        joint_pos_deg: Sequence[float] | np.ndarray,
        suffix: str = ".pos",
    ) -> dict[str, float]:
        """Build dict with keys like '<joint>.pos' from ordered joint vector."""
        q = self._coerce_joint_vector(joint_pos_deg)
        return {f"{name}{suffix}": float(q[i]) for i, name in enumerate(self.joint_names)}

    def forward_kinematics_from_joint_dict(
        self,
        joints: Mapping[str, float],
        suffix: str = ".pos",
    ) -> EndEffectorPose:
        """FK helper for dict-based joint inputs."""
        q = self.joints_dict_to_vector(joints, suffix=suffix)
        return EndEffectorPose.from_matrix(self.forward_kinematics(q))

    def inverse_kinematics_to_joint_dict(
        self,
        current_joints: Mapping[str, float],
        desired_pose: EndEffectorPose | np.ndarray,
        suffix: str = ".pos",
        preserve_extra_joint_keys: bool = True,
    ) -> dict[str, float]:
        """IK helper that returns a dict-based joint command."""
        q_curr = self.joints_dict_to_vector(current_joints, suffix=suffix)
        t_des = desired_pose.as_matrix() if isinstance(desired_pose, EndEffectorPose) else desired_pose

        q_target = self.inverse_kinematics(
            current_joint_pos_deg=q_curr,
            desired_ee_pose=t_des,
            preserve_extra_joints=False,
        )

        out = self.vector_to_joints_dict(q_target, suffix=suffix)
        if preserve_extra_joint_keys:
            for key, value in current_joints.items():
                if key not in out:
                    out[key] = float(value)
        return out

    @staticmethod
    def ee_fields_to_transform(ee_fields: Mapping[str, float], prefix: str = "ee.") -> np.ndarray:
        """Convert ee fields (ee.x, ee.y, ..., ee.wz) to a 4x4 transform."""
        pose = EndEffectorPose(
            x=float(ee_fields[f"{prefix}x"]),
            y=float(ee_fields[f"{prefix}y"]),
            z=float(ee_fields[f"{prefix}z"]),
            wx=float(ee_fields[f"{prefix}wx"]),
            wy=float(ee_fields[f"{prefix}wy"]),
            wz=float(ee_fields[f"{prefix}wz"]),
        )
        return pose.as_matrix()

    @staticmethod
    def transform_to_ee_fields(transform: np.ndarray, prefix: str = "ee.") -> dict[str, float]:
        """Convert a 4x4 transform to ee fields (ee.x, ee.y, ..., ee.wz)."""
        pose = EndEffectorPose.from_matrix(transform)
        return {
            f"{prefix}x": pose.x,
            f"{prefix}y": pose.y,
            f"{prefix}z": pose.z,
            f"{prefix}wx": pose.wx,
            f"{prefix}wy": pose.wy,
            f"{prefix}wz": pose.wz,
        }