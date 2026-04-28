#!/usr/bin/env python3
"""
Dual-Arm 2-DOF Forward / Inverse Kinematics for the SO-101 (left + right arms).

Teleoperate both arms simultaneously.  The same planar 2-DOF model is used for
each arm; the two arms share the same URDF geometry but connect to different
serial ports and use separate calibration files.

VR wrist-tracking topics consumed:
    /right_wrist                   → geometry_msgs/PoseStamped
    /right_hand/pinch_distance     → std_msgs/Float32
    /left_wrist                    → geometry_msgs/PoseStamped
    /left_hand/pinch_distance      → std_msgs/Float32

Usage (standalone):
    python fk_ik_2dof_dual.py                                          # headless demo
    python fk_ik_2dof_dual.py --gui                                    # PyBullet GUI
    python fk_ik_2dof_dual.py --viz                                    # matplotlib (no HW)
    python fk_ik_2dof_dual.py --viz \\
        --port-right /dev/ttyACM0 --port-left /dev/ttyACM1            # dual hardware
    python fk_ik_2dof_dual.py --viz --ros \\
        --port-right /dev/ttyACM0 --port-left /dev/ttyACM1            # full teleoperation
"""

from __future__ import annotations

import argparse
import json as _json
import math
import os
import threading
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pybullet as p
import pybullet_data

from so101 import SO101Arm

# Optional ROS2 imports (only needed when --ros is used)
try:
    import rclpy
    from rclpy.node import Node as _RosNode
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float32
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_URDF      = os.path.join(_THIS_DIR, "..", "urdf", "so101_follower_pybullet.urdf")
DEFAULT_CAL_RIGHT = os.path.join(_THIS_DIR, "..", "config", "right_arm.json")
DEFAULT_CAL_LEFT  = os.path.join(_THIS_DIR, "..", "config", "left_arm.json")

# Joint names for link-length extraction and active-joint indexing
_JOINT_LINK1  = "elbow_flex"   # parent-frame offset = length of link-1
_JOINT_LINK2  = "wrist_flex"   # parent-frame offset = length of link-2
_JOINT_THETA1 = "shoulder_lift"
_JOINT_THETA2 = "elbow_flex"

# Ground plane: arm cannot go below this Y value
GROUND_Y = 0.0


# ---------------------------------------------------------------------------
# Teleoperation constants
# ---------------------------------------------------------------------------
WRIST_ROLL_YAW_MIN   = -170.0
WRIST_ROLL_YAW_MAX   =  170.0
WRIST_FLEX_ROLL_MIN  =  -65.0
WRIST_FLEX_ROLL_MAX  =   50.0

SPIKE_THRESHOLD_YAW   = 15.0   # degrees — wrist_roll
SPIKE_THRESHOLD_ROLL  = 15.0   # degrees — wrist_flex
SPIKE_THRESHOLD_PINCH =  2.0   # cm      — gripper

DEADBAND_WRIST_ROLL  = 20  # ticks
DEADBAND_WRIST_FLEX  = 20  # ticks
DEADBAND_GRIPPER     =  3  # ticks

DEADBAND_WRIST_X = 0.008   # metres
DEADBAND_WRIST_Y = 0.008   # metres

PAN_WRIST_SCALE  = 300.0   # degrees of shoulder_pan per metre of wrist X displacement
TELE_SMOOTH_ALPHA = 0.25   # EMA smoothing factor for hardware servo commands

MAX_EE_VEL = 0.3           # m/s — boundary-travel speed when target is out of reach
_SMOOTH_DT = 0.05          # seconds — 20 Hz tick interval


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# PyBullet helpers
# ---------------------------------------------------------------------------

def _get_joint_index(robot_id: int, joint_name: str) -> int:
    n = p.getNumJoints(robot_id)
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        if info[1].decode("utf-8") == joint_name:
            return i
    raise ValueError(f"Joint '{joint_name}' not found in URDF.")


def extract_joint_limits(robot_id: int) -> Tuple[float, float, float, float]:
    """Return (t1_min, t1_max, t2_min, t2_max) in radians from the URDF."""
    idx1  = _get_joint_index(robot_id, _JOINT_THETA1)
    idx2  = _get_joint_index(robot_id, _JOINT_THETA2)
    info1 = p.getJointInfo(robot_id, idx1)
    info2 = p.getJointInfo(robot_id, idx2)
    return float(info1[8]), float(info1[9]), float(info2[8]), float(info2[9])


def extract_link_lengths(robot_id: int) -> Tuple[float, float]:
    """Extract L1 and L2 from joint parent-frame offsets in the URDF."""
    idx1 = _get_joint_index(robot_id, _JOINT_LINK1)
    idx2 = _get_joint_index(robot_id, _JOINT_LINK2)
    L1 = float(np.linalg.norm(np.array(p.getJointInfo(robot_id, idx1)[14])))
    L2 = float(np.linalg.norm(np.array(p.getJointInfo(robot_id, idx2)[14])))
    return L1, L2


# ---------------------------------------------------------------------------
# 2-DOF Forward Kinematics
# ---------------------------------------------------------------------------

def forward_kinematics(theta1: float, theta2: float, L1: float, L2: float) -> Tuple[float, float]:
    """Planar 2-DOF FK.  Angles in radians, returns (x, y) in metres."""
    x = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
    y = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
    return x, y


# ---------------------------------------------------------------------------
# 2-DOF Inverse Kinematics
# ---------------------------------------------------------------------------

def inverse_kinematics(
    x: float,
    y: float,
    L1: float,
    L2: float,
    elbow_up: bool = True,
    limits: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[Tuple[float, float]]:
    """Planar 2-DOF geometric IK (cosine-rule).

    Returns (theta1, theta2) in radians, or None if unreachable / out of limits.
    """
    r_sq = x * x + y * y
    cos_theta2 = (r_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_theta2 = _clamp(cos_theta2, -1.0, 1.0)

    sin_theta2 = math.sqrt(max(0.0, 1.0 - cos_theta2 * cos_theta2))
    if not elbow_up:
        sin_theta2 = -sin_theta2

    theta2 = math.atan2(sin_theta2, cos_theta2)
    k1 = L1 + L2 * cos_theta2
    k2 = L2 * sin_theta2
    theta1 = math.atan2(y, x) - math.atan2(k2, k1)

    if limits is not None:
        t1_min, t1_max, t2_min, t2_max = limits
        if not (t1_min <= theta1 <= t1_max and t2_min <= theta2 <= t2_max):
            return None

    return theta1, theta2


# ---------------------------------------------------------------------------
# Ground constraint helper
# ---------------------------------------------------------------------------

def _above_ground(t1: float, t2: float, L1: float, L2: float) -> bool:
    elbow_y = L1 * math.sin(t1)
    ee_y    = elbow_y + L2 * math.sin(t1 + t2)
    return elbow_y >= GROUND_Y and ee_y >= GROUND_Y


# ---------------------------------------------------------------------------
# Reachable-boundary helper
# ---------------------------------------------------------------------------

def _closest_reachable(x_t: float, y_t: float, L1: float, L2: float) -> Tuple[float, float]:
    """Return the nearest point inside the reachable annulus toward (x_t, y_t)."""
    r = math.hypot(x_t, y_t)
    if r < 1e-9:
        return L1, GROUND_Y
    r_cl = _clamp(r, abs(L1 - L2), L1 + L2)
    x = x_t / r * r_cl
    y = y_t / r * r_cl
    if y < GROUND_Y:
        y = GROUND_Y
    return x, y


# ---------------------------------------------------------------------------
# Quaternion helper
# ---------------------------------------------------------------------------

def quaternion_to_rpy(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    roll  = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw   = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# PyBullet draw helper
# ---------------------------------------------------------------------------

def _draw_arm(robot_id: int, idx_t1: int, idx_t2: int, theta1: float, theta2: float) -> None:
    p.resetJointState(robot_id, idx_t1, theta1)
    p.resetJointState(robot_id, idx_t2, theta2)
    p.stepSimulation()


# ---------------------------------------------------------------------------
# Dual-arm interactive visualisation
# ---------------------------------------------------------------------------

def run_dual_visualization(
    L1: float,
    L2: float,
    limits: Tuple[float, float, float, float],
    arm_right: Optional[SO101Arm] = None,
    arm_left:  Optional[SO101Arm] = None,
    joints_cal_right: Optional[dict] = None,
    joints_cal_left:  Optional[dict] = None,
    use_ros:     bool  = False,
    wrist_scale: float = 1.0,
) -> None:
    """Open an interactive matplotlib window with independent X/Y/Pan sliders for each arm."""

    t1_min, t1_max, t2_min, t2_max = limits
    reach  = L1 + L2
    margin = reach * 0.18

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"SO-101 Dual-Arm 2-DOF  —  L1={L1:.3f} m  L2={L2:.3f} m  "
        f"θ1 [{math.degrees(t1_min):.1f}°, {math.degrees(t1_max):.1f}°]  "
        f"θ2 [{math.degrees(t2_min):.1f}°, {math.degrees(t2_max):.1f}°]",
        fontsize=9,
    )

    # Main arm axes (left panel | right panel)
    ax_l = fig.add_axes([0.04, 0.33, 0.42, 0.60])
    ax_r = fig.add_axes([0.54, 0.33, 0.42, 0.60])

    # Slider axes — left arm
    sl_lpan_ax = fig.add_axes([0.06, 0.24, 0.38, 0.025])
    sl_lx_ax   = fig.add_axes([0.06, 0.17, 0.38, 0.025])
    sl_ly_ax   = fig.add_axes([0.06, 0.10, 0.38, 0.025])

    # Slider axes — right arm
    sl_rpan_ax = fig.add_axes([0.56, 0.24, 0.38, 0.025])
    sl_rx_ax   = fig.add_axes([0.56, 0.17, 0.38, 0.025])
    sl_ry_ax   = fig.add_axes([0.56, 0.10, 0.38, 0.025])

    # -----------------------------------------------------------------------
    # Per-arm mutable state
    # -----------------------------------------------------------------------
    def _make_state(arm: Optional[SO101Arm], joints_cal: Optional[dict]) -> dict:
        return {
            "arm":        arm,
            "joints_cal": joints_cal,
            "last":       {"t1": math.radians(90.0), "t2": math.radians(-90.0)},
            "smooth":     {"x": round(L2, 3), "y": round(L1, 3)},
            "desired":    {"x": round(L2, 3), "y": round(L1, 3), "pan": 0.0},
            "prev_hw_cmd": {"j1": None, "j2": None, "pan": None},
            "smooth_cmd":  {"j1": None, "j2": None, "pan": None},
        }

    states = {
        "left":  _make_state(arm_left,  joints_cal_left),
        "right": _make_state(arm_right, joints_cal_right),
    }

    # -----------------------------------------------------------------------
    # Graphics setup — one pass per arm
    # -----------------------------------------------------------------------
    _COLORS = {
        "left":  ("darkorange", "orange",    "LEFT ARM"),
        "right": ("steelblue",  "royalblue", "RIGHT ARM"),
    }
    gfx = {}

    # Pre-compute reachable workspace scatter (shared — same URDF geometry)
    _ws_x, _ws_y = [], []
    for _t1 in np.linspace(t1_min, t1_max, 200):
        for _t2 in np.linspace(t2_min, t2_max, 100):
            if not _above_ground(_t1, _t2, L1, L2):
                continue
            _ex, _ey = forward_kinematics(_t1, _t2, L1, L2)
            _ws_x.append(_ex)
            _ws_y.append(_ey)

    for side, ax in [("left", ax_l), ("right", ax_r)]:
        col_a, col_b, title = _COLORS[side]

        ax.set_xlim(-(reach + margin), reach + margin)
        ax.set_ylim(-(reach + margin), reach + margin)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title, fontsize=10, fontweight="bold")

        # Ground
        ax.axhline(GROUND_Y, color="saddlebrown", linewidth=2.5, linestyle="-",
                   label=f"Ground (y={GROUND_Y:.2f} m)", zorder=3)
        ax.axhspan(-(reach + margin), GROUND_Y, color="saddlebrown", alpha=0.08, zorder=0)

        # Workspace scatter
        ax.scatter(_ws_x, _ws_y, s=2, color="lightgreen", alpha=0.20,
                   label="Reachable workspace", zorder=1)
        ax.add_patch(
            plt.Circle((0, 0), reach, fill=False, color="green", linestyle="--",
                        linewidth=1.0, alpha=0.4, label=f"Max reach ({reach:.3f} m)")
        )
        inner_r = abs(L1 - L2)
        if inner_r > 1e-4:
            ax.add_patch(
                plt.Circle((0, 0), inner_r, fill=False, color="red", linestyle="--",
                            linewidth=1.0, alpha=0.4, label=f"Min reach ({inner_r:.3f} m)")
            )

        # Arm links / markers
        link1,     = ax.plot([], [], color=col_a, linewidth=6, solid_capstyle="round",
                             label="Arm")
        link2,     = ax.plot([], [], color=col_b, linewidth=6, solid_capstyle="round")
        joints_pt, = ax.plot([], [], "o", color="white", markersize=9,
                             markeredgecolor=col_a, markeredgewidth=2, zorder=5)
        ee_pt,     = ax.plot([], [], "*", color=col_a, markersize=18, zorder=6)
        target_pt, = ax.plot([], [], "rx", markersize=13, markeredgewidth=2.5,
                             zorder=7, label="Target")

        status_text = ax.text(
            0.02, 0.98, "",
            transform=ax.transAxes, verticalalignment="top", fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        ax.legend(loc="upper right", fontsize=7)

        # Top-view pan inset
        ax_top = ax.inset_axes([0.01, 0.60, 0.28, 0.36])
        ax_top.set_aspect("equal")
        ax_top.set_title("Top View\n(shoulder pan)", fontsize=7, pad=2)
        ax_top.set_xlim(-1.45, 1.45)
        ax_top.set_ylim(-0.38, 1.48)
        ax_top.axis("off")

        _pa = np.linspace(-math.pi / 2, math.pi / 2, 60)
        ax_top.fill(np.concatenate([[0.0], np.sin(_pa)]),
                    np.concatenate([[0.0], np.cos(_pa)]),
                    color="lightcyan", alpha=0.4, zorder=0)
        ax_top.plot(np.sin(_pa), np.cos(_pa), color="steelblue", linewidth=1.2, alpha=0.7)
        for _tdeg in [-90, -45, 0, 45, 90]:
            _tr = math.radians(_tdeg)
            _tx, _ty = math.sin(_tr), math.cos(_tr)
            ax_top.plot([0.82 * _tx, _tx], [0.82 * _ty, _ty],
                        color="steelblue", linewidth=1.0, alpha=0.8)
            _lx = 1.25 * _tx if _tdeg != 0 else 0.0
            ax_top.text(_lx, 1.22 * _ty, f"{_tdeg:+d}°",
                        ha="center", va="center", fontsize=6, color="dimgray")
        ax_top.text(0, 1.38, "fwd", ha="center", va="bottom", fontsize=6, color="gray")
        ax_top.plot(0, 0, "o", color="steelblue", markersize=5, zorder=4)

        pan_line, = ax_top.plot([0, 0], [0, 1.0], color="tomato", linewidth=3,
                                solid_capstyle="round", zorder=5)
        pan_dot,  = ax_top.plot([0], [1.0], "o", color="tomato", markersize=7, zorder=6)
        pan_text  = ax_top.text(0, -0.24, "Pan:  0.0°", ha="center", va="top", fontsize=7,
                                family="monospace",
                                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        gfx[side] = {
            "link1":       link1,
            "link2":       link2,
            "joints_pt":   joints_pt,
            "ee_pt":       ee_pt,
            "target_pt":   target_pt,
            "status_text": status_text,
            "pan_line":    pan_line,
            "pan_dot":     pan_dot,
            "pan_text":    pan_text,
        }

    # -----------------------------------------------------------------------
    # Sliders
    # -----------------------------------------------------------------------
    slider_lpan = Slider(sl_lpan_ax, "L Pan (°)", -90.0, 90.0, valinit=0.0, valstep=0.5)
    slider_lx   = Slider(sl_lx_ax,   "L X (m)",  -reach, reach,
                         valinit=round(L2, 3), valstep=0.001)
    slider_ly   = Slider(sl_ly_ax,   "L Y (m)",  -reach, reach,
                         valinit=round(L1, 3), valstep=0.001)

    slider_rpan = Slider(sl_rpan_ax, "R Pan (°)", -90.0, 90.0, valinit=0.0, valstep=0.5)
    slider_rx   = Slider(sl_rx_ax,   "R X (m)",  -reach, reach,
                         valinit=round(L2, 3), valstep=0.001)
    slider_ry   = Slider(sl_ry_ax,   "R Y (m)",  -reach, reach,
                         valinit=round(L1, 3), valstep=0.001)

    sliders = {
        "left":  {"pan": slider_lpan, "x": slider_lx, "y": slider_ly},
        "right": {"pan": slider_rpan, "x": slider_rx, "y": slider_ry},
    }

    # Slider callbacks — one factory per side to avoid closure issues
    def _make_xy_cb(side: str):
        def _cb(_val):
            states[side]["desired"]["x"] = sliders[side]["x"].val
            states[side]["desired"]["y"] = sliders[side]["y"].val
            gfx[side]["target_pt"].set_data(
                [states[side]["desired"]["x"]], [states[side]["desired"]["y"]]
            )
            fig.canvas.draw_idle()
        return _cb

    def _make_pan_cb(side: str):
        def _cb(_val):
            states[side]["desired"]["pan"] = sliders[side]["pan"].val
            fig.canvas.draw_idle()
        return _cb

    for _side in ("left", "right"):
        sliders[_side]["x"].on_changed(_make_xy_cb(_side))
        sliders[_side]["y"].on_changed(_make_xy_cb(_side))
        sliders[_side]["pan"].on_changed(_make_pan_cb(_side))

    # -----------------------------------------------------------------------
    # Hardware feedback helper
    # -----------------------------------------------------------------------
    def _read_arm_model_angles(arm: SO101Arm) -> Optional[Tuple[float, float]]:
        try:
            pos = arm.read_positions()
            s1  = pos.get("shoulder_lift.pos")
            s2  = pos.get("elbow_flex.pos")
            if s1 is None or s2 is None:
                return None
            return math.radians(90.0 - float(s1)), math.radians(-float(s2) - 90.0)
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Per-arm tick (IK → HW send → graphics update)
    # -----------------------------------------------------------------------
    def _tick_arm(side: str) -> None:
        st  = states[side]
        g   = gfx[side]
        arm = st["arm"]

        x_des    = st["desired"]["x"]
        y_des    = st["desired"]["y"]
        y_des_cl = max(y_des, GROUND_Y)

        # Check whether the desired target is directly reachable
        sol_direct = inverse_kinematics(x_des, y_des_cl, L1, L2, elbow_up=False)
        if sol_direct is not None and not _above_ground(sol_direct[0], sol_direct[1], L1, L2):
            sol_direct = None

        if sol_direct is not None:
            st["smooth"]["x"] = x_des
            st["smooth"]["y"] = y_des_cl
            out_of_reach = False
        else:
            cx, cy = _closest_reachable(x_des, y_des, L1, L2)
            cy = max(cy, GROUND_Y)
            dx   = cx - st["smooth"]["x"]
            dy   = cy - st["smooth"]["y"]
            dist = math.hypot(dx, dy)
            max_step = MAX_EE_VEL * _SMOOTH_DT
            if dist <= max_step:
                st["smooth"]["x"] = cx
                st["smooth"]["y"] = cy
            else:
                st["smooth"]["x"] += dx / dist * max_step
                st["smooth"]["y"] += dy / dist * max_step
            out_of_reach = True

        # Solve IK at smooth position
        sol = inverse_kinematics(st["smooth"]["x"], st["smooth"]["y"], L1, L2, elbow_up=False)
        if sol is not None and not _above_ground(sol[0], sol[1], L1, L2):
            sol = None

        if sol is not None:
            t1, t2 = sol
            st["last"]["t1"] = t1
            st["last"]["t2"] = t2
        else:
            if arm is not None:
                fb = _read_arm_model_angles(arm)
                if fb is not None:
                    st["last"]["t1"], st["last"]["t2"] = fb
            t1 = st["last"]["t1"]
            t2 = st["last"]["t2"]

        xj = L1 * math.cos(t1)
        yj = L1 * math.sin(t1)
        xe, ye = forward_kinematics(t1, t2, L1, L2)

        g["link1"].set_data([0, xj],       [0, yj])
        g["link2"].set_data([xj, xe],      [yj, ye])
        g["joints_pt"].set_data([0, xj, xe], [0, yj, ye])
        g["ee_pt"].set_data([xe], [ye])

        t1_d      = math.degrees(t1)
        t2_d      = math.degrees(t2)
        servo_j1  = 90.0 - t1_d
        servo_j2  = -t2_d - 90.0
        servo_pan = st["desired"]["pan"]

        # EMA smoothing for hardware commands
        sc = st["smooth_cmd"]
        if sc["j1"] is None:
            sc["j1"]  = servo_j1
            sc["j2"]  = servo_j2
            sc["pan"] = servo_pan
        else:
            sc["j1"]  = TELE_SMOOTH_ALPHA * servo_j1  + (1.0 - TELE_SMOOTH_ALPHA) * sc["j1"]
            sc["j2"]  = TELE_SMOOTH_ALPHA * servo_j2  + (1.0 - TELE_SMOOTH_ALPHA) * sc["j2"]
            sc["pan"] = TELE_SMOOTH_ALPHA * servo_pan + (1.0 - TELE_SMOOTH_ALPHA) * sc["pan"]

        hw_j1  = sc["j1"]
        hw_j2  = sc["j2"]
        hw_pan = sc["pan"]

        phc = st["prev_hw_cmd"]
        hw_changed = (
            phc["j1"] is None
            or abs(hw_j1  - phc["j1"])  > 0.05
            or abs(hw_j2  - phc["j2"])  > 0.05
            or phc["pan"] is None
            or abs(hw_pan - phc["pan"]) > 0.1
        )

        tag = f"[{side.upper()}]"

        if out_of_reach:
            label = "OUT OF REACH — travelling to boundary"
            if hw_changed:
                print(
                    f"{tag} model J1={t1_d:+7.2f}°  J2={t2_d:+7.2f}°  "
                    f"servo J1={servo_j1:+7.2f}°  J2={servo_j2:+7.2f}°  "
                    f"pan={servo_pan:+.1f}°  EE=({xe:+.4f},{ye:+.4f}) m  "
                    f"[SMOOTH→BOUNDARY target({x_des:+.4f},{y_des:+.4f})]"
                )
        else:
            label = "IK OK"
            if hw_changed:
                print(
                    f"{tag} model J1={t1_d:+7.2f}°  J2={t2_d:+7.2f}°  "
                    f"servo J1={servo_j1:+7.2f}°  J2={servo_j2:+7.2f}°  "
                    f"pan={servo_pan:+.1f}°  EE=({xe:+.4f},{ye:+.4f}) m"
                )

        # Send to hardware
        if arm is not None and hw_changed:
            try:
                arm.set_positions(
                    {"shoulder_lift": hw_j1, "elbow_flex": hw_j2, "shoulder_pan": hw_pan},
                    expect_ack=False,
                )
                hw_status = "HW SMOOTH" if out_of_reach else "HW OK"
            except Exception as exc:
                hw_status = f"HW ERR: {exc}"
        else:
            hw_status = (
                "(no HW)" if arm is None else
                ("HW SMOOTH" if out_of_reach else "HW OK")
            )

        if hw_changed:
            phc["j1"]  = hw_j1
            phc["j2"]  = hw_j2
            phc["pan"] = hw_pan

        # Top-view pan indicator
        _pan_rad = math.radians(servo_pan)
        g["pan_line"].set_data([0, math.sin(_pan_rad)], [0, math.cos(_pan_rad)])
        g["pan_dot"].set_data([math.sin(_pan_rad)], [math.cos(_pan_rad)])
        g["pan_text"].set_text(f"Pan: {servo_pan:+.1f}°")

        g["status_text"].set_text(
            f"{label}  {hw_status}\n"
            f"model  J1={t1_d:+7.2f}°   J2={t2_d:+7.2f}°\n"
            f"servo  J1={servo_j1:+7.2f}°   J2={servo_j2:+7.2f}°\n"
            f"pan={servo_pan:+.1f}°   EE=({xe:+.4f},{ye:+.4f}) m"
        )

    # -----------------------------------------------------------------------
    # Combined tick (both arms)
    # -----------------------------------------------------------------------
    def _tick():
        for _s in ("left", "right"):
            _tick_arm(_s)
        fig.canvas.draw_idle()

    # Initial target markers + first draw
    for _s in ("left", "right"):
        gfx[_s]["target_pt"].set_data(
            [states[_s]["desired"]["x"]], [states[_s]["desired"]["y"]]
        )
    _tick()

    _smooth_timer = fig.canvas.new_timer(interval=int(_SMOOTH_DT * 1000))
    _smooth_timer.add_callback(_tick)
    _smooth_timer.start()

    # -----------------------------------------------------------------------
    # ROS2 dual wrist-tracking
    # -----------------------------------------------------------------------
    if use_ros:
        if not _ROS_AVAILABLE:
            print("[WARN] --ros requested but rclpy is not available. Wrist tracking disabled.")
        else:
            _wrist_deltas: dict = {
                "left":  {"dx": 0.0, "dy": 0.0},
                "right": {"dx": 0.0, "dy": 0.0},
            }
            _pan_shared: dict = {"left": 0.0, "right": 0.0}
            _wrist_lock = threading.Lock()

            class _DualWristNode(_RosNode):
                """Single ROS2 node that tracks both wrists."""

                def __init__(self_node):
                    super().__init__("fk_ik_dual_wrist_ctrl")
                    qos = QoSProfile(
                        depth=10,
                        reliability=ReliabilityPolicy.BEST_EFFORT,
                    )

                    # Per-side tracking state
                    self_node._as: dict = {}
                    for _side in ("left", "right"):
                        self_node._as[_side] = {
                            "prev_z":              None,
                            "prev_y":              None,
                            "prev_x":              None,
                            "ref_wrist_x":         None,
                            "last_yaw":            None,
                            "last_roll":           None,
                            "last_gripper_pinch":  None,
                            "sent_wrist_roll":     None,
                            "sent_wrist_flex":     None,
                            "sent_gripper":        None,
                        }

                    self_node.create_subscription(
                        PoseStamped, "/right_wrist", self_node._cb_right, qos
                    )
                    self_node.create_subscription(
                        PoseStamped, "/left_wrist", self_node._cb_left, qos
                    )
                    self_node.create_subscription(
                        Float32, "/right_hand/pinch_distance",
                        self_node._pinch_cb_right, qos,
                    )
                    self_node.create_subscription(
                        Float32, "/left_hand/pinch_distance",
                        self_node._pinch_cb_left, qos,
                    )
                    self_node.get_logger().info(
                        "Subscribed to /right_wrist, /left_wrist, "
                        "/right_hand/pinch_distance, /left_hand/pinch_distance"
                    )

                # --- Dispatch wrappers ---
                def _cb_right(self_node, msg: "PoseStamped"):
                    self_node._wrist_cb(msg, "right")

                def _cb_left(self_node, msg: "PoseStamped"):
                    self_node._wrist_cb(msg, "left")

                def _pinch_cb_right(self_node, msg: "Float32"):
                    self_node._pinch_cb(msg, "right")

                def _pinch_cb_left(self_node, msg: "Float32"):
                    self_node._pinch_cb(msg, "left")

                # --- Wrist pose callback ---
                def _wrist_cb(self_node, msg: "PoseStamped", side: str):
                    as_     = self_node._as[side]
                    arm     = states[side]["arm"]
                    jcal    = states[side]["joints_cal"]

                    cur_z = msg.pose.position.z
                    cur_y = msg.pose.position.y
                    cur_x = msg.pose.position.x

                    if as_["prev_z"] is not None:
                        dz     = cur_z - as_["prev_z"]
                        dy     = cur_y - as_["prev_y"]
                        dx_pan = cur_x - as_["prev_x"]

                        pan_dominant = abs(dx_pan) > abs(dz) and abs(dx_pan) > abs(dy)
                        if not pan_dominant:
                            with _wrist_lock:
                                _wrist_deltas[side]["dx"] += dz * wrist_scale
                                _wrist_deltas[side]["dy"] += dy * wrist_scale

                    as_["prev_z"] = cur_z
                    as_["prev_y"] = cur_y
                    as_["prev_x"] = cur_x

                    # Shoulder pan via wrist X position
                    if as_["ref_wrist_x"] is None:
                        as_["ref_wrist_x"] = cur_x
                        self_node.get_logger().info(
                            f"[{side.upper()}] Pan reference captured — "
                            f"wrist X = {cur_x:.4f} m"
                        )
                    else:
                        pan_deg = (cur_x - as_["ref_wrist_x"]) * PAN_WRIST_SCALE
                        pan_deg = _clamp(pan_deg, -90.0, 90.0)
                        with _wrist_lock:
                            _pan_shared[side] = pan_deg

                    # Wrist roll / flex require hardware + calibration
                    if arm is None or jcal is None:
                        return

                    q = msg.pose.orientation
                    roll, pitch, yaw = quaternion_to_rpy(q.x, q.y, q.z, q.w)
                    yaw_deg   = math.degrees(yaw)
                    roll_deg  = math.degrees(roll)
                    pitch_deg = math.degrees(pitch)

                    # wrist_roll via yaw
                    if WRIST_ROLL_YAW_MIN <= yaw_deg <= WRIST_ROLL_YAW_MAX:
                        if (as_["last_yaw"] is None
                                or abs(yaw_deg - as_["last_yaw"]) <= SPIKE_THRESHOLD_YAW):
                            as_["last_yaw"] = yaw_deg
                            frac = (
                                (yaw_deg - WRIST_ROLL_YAW_MIN)
                                / (WRIST_ROLL_YAW_MAX - WRIST_ROLL_YAW_MIN)
                            )
                            wr_pos = int(
                                jcal["wrist_roll"]["range_min"]
                                + frac * (jcal["wrist_roll"]["range_max"]
                                          - jcal["wrist_roll"]["range_min"])
                            )
                            if (as_["sent_wrist_roll"] is None
                                    or abs(wr_pos - as_["sent_wrist_roll"]) >= DEADBAND_WRIST_ROLL):
                                as_["sent_wrist_roll"] = wr_pos
                                arm.set_positions_raw(
                                    {5: wr_pos}, acc=254, speed=0, expect_ack=False
                                )
                        else:
                            self_node.get_logger().warn(
                                f"[{side.upper()}] Yaw spike rejected: "
                                f"{as_['last_yaw']:.1f}° → {yaw_deg:.1f}°"
                            )

                    # wrist_flex via roll/pitch blended by current yaw
                    theta = math.radians(
                        as_["last_yaw"] if as_["last_yaw"] is not None else 0.0
                    )
                    effective_flex = roll_deg * math.cos(theta) - pitch_deg * math.sin(theta)

                    if WRIST_FLEX_ROLL_MIN <= effective_flex <= WRIST_FLEX_ROLL_MAX:
                        if (as_["last_roll"] is None
                                or abs(effective_flex - as_["last_roll"]) <= SPIKE_THRESHOLD_ROLL):
                            as_["last_roll"] = effective_flex
                            frac = (
                                (effective_flex - WRIST_FLEX_ROLL_MIN)
                                / (WRIST_FLEX_ROLL_MAX - WRIST_FLEX_ROLL_MIN)
                            )
                            wf_pos = int(
                                jcal["wrist_flex"]["range_min"]
                                + frac * (jcal["wrist_flex"]["range_max"]
                                          - jcal["wrist_flex"]["range_min"])
                            )
                            if (as_["sent_wrist_flex"] is None
                                    or abs(wf_pos - as_["sent_wrist_flex"]) >= DEADBAND_WRIST_FLEX):
                                as_["sent_wrist_flex"] = wf_pos
                                arm.set_positions_raw(
                                    {4: wf_pos}, acc=254, speed=0, expect_ack=False
                                )
                        else:
                            self_node.get_logger().warn(
                                f"[{side.upper()}] Flex spike rejected: "
                                f"{as_['last_roll']:.1f}° → {effective_flex:.1f}°"
                            )

                # --- Gripper callback ---
                def _pinch_cb(self_node, msg: "Float32", side: str):
                    arm  = states[side]["arm"]
                    jcal = states[side]["joints_cal"]
                    if arm is None or jcal is None:
                        return

                    as_   = self_node._as[side]
                    pinch = float(msg.data)
                    pinch = _clamp(pinch, 2.0, 10.0)

                    if (as_["last_gripper_pinch"] is not None
                            and abs(pinch - as_["last_gripper_pinch"]) > SPIKE_THRESHOLD_PINCH):
                        self_node.get_logger().warn(
                            f"[{side.upper()}] Pinch spike rejected: "
                            f"{as_['last_gripper_pinch']:.2f} → {pinch:.2f} cm"
                        )
                        return

                    as_["last_gripper_pinch"] = pinch
                    frac = (pinch - 2.0) / 8.0
                    gripper_pos = int(
                        jcal["gripper"]["range_min"]
                        + frac * (jcal["gripper"]["range_max"]
                                  - jcal["gripper"]["range_min"])
                    )
                    if (as_["sent_gripper"] is None
                            or abs(gripper_pos - as_["sent_gripper"]) >= DEADBAND_GRIPPER):
                        as_["sent_gripper"] = gripper_pos
                        arm.set_positions_raw(
                            {6: gripper_pos}, acc=254, speed=0, expect_ack=False
                        )

            input(
                "\nPosition BOTH arms so that shoulder_pan is at 0° (centre), "
                "then press ENTER..."
            )
            rclpy.init()
            _dual_node = _DualWristNode()

            def _ros_spin():
                rclpy.spin(_dual_node)

            _ros_thread = threading.Thread(
                target=_ros_spin, daemon=True, name="ros-dual-wrist-spin"
            )
            _ros_thread.start()
            print("[ROS] Dual wrist tracking active — wrist Z→X, wrist Y→Y (both arms)")

            for _s in ("left", "right"):
                _arm = states[_s]["arm"]
                _cal = states[_s]["joints_cal"]
                if _arm is not None and _cal is not None:
                    print(f"[ROS] [{_s.upper()}] Teleoperation active — "
                          "wrist_roll, wrist_flex, gripper enabled")
                elif _arm is None:
                    print(f"[ROS] [{_s.upper()}] No hardware — wrist_roll/flex/gripper disabled")
                else:
                    print(f"[ROS] [{_s.upper()}] No calibration — wrist_roll/flex/gripper disabled")

            def _apply_wrist_deltas():
                """Called every 50 ms by matplotlib timer; applies accumulated delta."""
                with _wrist_lock:
                    for _side in ("left", "right"):
                        dx = _wrist_deltas[_side]["dx"]
                        dy = _wrist_deltas[_side]["dy"]

                        apply_x = abs(dx) >= DEADBAND_WRIST_X
                        apply_y = abs(dy) >= DEADBAND_WRIST_Y

                        if apply_x:
                            _wrist_deltas[_side]["dx"] = 0.0
                        if apply_y:
                            _wrist_deltas[_side]["dy"] = 0.0

                        new_pan   = _pan_shared[_side]
                        apply_pan = abs(new_pan - sliders[_side]["pan"].val) >= 0.5

                        if not apply_x and not apply_y and not apply_pan:
                            continue

                        new_x = float(np.clip(
                            sliders[_side]["x"].val + (dx if apply_x else 0.0),
                            -reach, reach,
                        ))
                        new_y = float(np.clip(
                            sliders[_side]["y"].val + (dy if apply_y else 0.0),
                            -reach, reach,
                        ))

                        if apply_x:
                            sliders[_side]["x"].set_val(round(new_x, 3))
                        if apply_y:
                            sliders[_side]["y"].set_val(round(new_y, 3))
                        if apply_pan:
                            sliders[_side]["pan"].set_val(round(new_pan, 1))

            _ros_timer = fig.canvas.new_timer(interval=50)
            _ros_timer.add_callback(_apply_wrist_deltas)
            _ros_timer.start()

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SO-101 Dual-Arm 2-DOF FK/IK teleoperation (left + right)"
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF,
                        help="Path to the URDF file")
    parser.add_argument("--gui", action="store_true",
                        help="Open the PyBullet GUI viewer")
    parser.add_argument("--viz", action="store_true",
                        help="Open interactive matplotlib dual-arm visualisation")

    # Right arm
    parser.add_argument("--port-right", dest="port_right", default=None,
                        help="Serial port for the right arm (e.g. /dev/ttyACM0)")
    parser.add_argument("--cal-right", dest="cal_right", default=DEFAULT_CAL_RIGHT,
                        help="Calibration JSON for the right arm")

    # Left arm
    parser.add_argument("--port-left", dest="port_left", default=None,
                        help="Serial port for the left arm (e.g. /dev/ttyACM1)")
    parser.add_argument("--cal-left", dest="cal_left", default=DEFAULT_CAL_LEFT,
                        help="Calibration JSON for the left arm")

    parser.add_argument("--ros", action="store_true",
                        help="Control sliders from /right_wrist + /left_wrist ROS2 topics")
    parser.add_argument("--wrist-scale", dest="wrist_scale", type=float, default=1.0,
                        help="Scale factor applied to wrist displacement (default: 1.0)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # PyBullet setup (shared geometry for both arms)
    # -----------------------------------------------------------------------
    mode = p.GUI if args.gui else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    robot_id = p.loadURDF(args.urdf, useFixedBase=True)

    L1, L2  = extract_link_lengths(robot_id)
    limits  = extract_joint_limits(robot_id)
    t1_min, t1_max, t2_min, t2_max = limits

    print("=" * 60)
    print("Link lengths extracted from URDF")
    print(f"  L1 (shoulder_lift → elbow_flex) : {L1:.6f} m")
    print(f"  L2 (elbow_flex   → wrist_flex)  : {L2:.6f} m")
    print("Joint limits extracted from URDF")
    print(f"  shoulder_lift (θ1): [{math.degrees(t1_min):.2f}°, {math.degrees(t1_max):.2f}°]")
    print(f"  elbow_flex    (θ2): [{math.degrees(t2_min):.2f}°, {math.degrees(t2_max):.2f}°]")
    print(f"Ground constraint  : y >= {GROUND_Y:.3f} m")
    print("=" * 60)

    # FK demo
    print("\n--- Forward Kinematics ---")
    for theta1, theta2, label in [
        (0.0,           0.0,           "zero pose"),
        (math.pi / 4,   math.pi / 4,   "45° / 45°"),
        (math.pi / 6,   math.pi / 3,   "30° / 60°"),
        (0.0,           math.pi / 2,   "0° / 90°"),
    ]:
        x, y = forward_kinematics(theta1, theta2, L1, L2)
        print(
            f"  [{label}]  θ1={math.degrees(theta1):6.1f}°  "
            f"θ2={math.degrees(theta2):6.1f}°  →  x={x:.4f} m,  y={y:.4f} m"
        )

    # IK round-trip demo
    print("\n--- Inverse Kinematics (round-trip verification) ---")
    idx_t1 = _get_joint_index(robot_id, _JOINT_THETA1)
    idx_t2 = _get_joint_index(robot_id, _JOINT_THETA2)

    for x_t, y_t, label in [
        (L1 + L2, 0.0,  "full reach along x"),
        (0.10,    0.15,  "(0.10, 0.15)"),
        (0.20,    0.05,  "(0.20, 0.05)"),
        (0.50,    0.50,  "out-of-reach target"),
    ]:
        print(f"\n  Target [{label}]: x={x_t:.4f}, y={y_t:.4f}")
        sol = inverse_kinematics(x_t, y_t, L1, L2, elbow_up=False, limits=limits)
        if sol is None:
            print("    elbow-down: UNREACHABLE / outside joint limits")
            continue
        t1, t2 = sol
        if not _above_ground(t1, t2, L1, L2):
            print("    elbow-down: REJECTED — arm would go below ground")
            continue
        x_fk, y_fk = forward_kinematics(t1, t2, L1, L2)
        err = math.hypot(x_fk - x_t, y_fk - y_t)
        print(
            f"    elbow-down: θ1={math.degrees(t1):7.2f}°  θ2={math.degrees(t2):7.2f}°  "
            f"→ FK=({x_fk:.4f},{y_fk:.4f})  err={err:.2e} m"
        )
        if args.gui:
            _draw_arm(robot_id, idx_t1, idx_t2, t1, t2)

    print("\nDone.")
    if args.gui:
        input("PyBullet GUI is open — press Enter to exit.")
    p.disconnect()

    if not args.viz:
        return

    # -----------------------------------------------------------------------
    # Matplotlib dual-arm visualisation
    # -----------------------------------------------------------------------
    print("\nOpening dual-arm matplotlib visualisation...")

    arm_right = arm_left = None
    joints_cal_right = joints_cal_left = None

    # Connect to right arm hardware
    if args.port_right:
        print(f"Connecting to right arm on {args.port_right} (cal: {args.cal_right}) ...")
        arm_right = SO101Arm(args.port_right, calibration_file=args.cal_right)
        arm_right.open()
        print("Right arm connected.")

    # Connect to left arm hardware
    if args.port_left:
        print(f"Connecting to left arm on {args.port_left} (cal: {args.cal_left}) ...")
        arm_left = SO101Arm(args.port_left, calibration_file=args.cal_left)
        arm_left.open()
        print("Left arm connected.")

    # Load calibration files
    for side, cal_path in [("right", args.cal_right), ("left", args.cal_left)]:
        if os.path.isfile(cal_path):
            try:
                with open(cal_path, "r") as _f:
                    cal_data = _json.load(_f)
                if side == "right":
                    joints_cal_right = cal_data
                else:
                    joints_cal_left = cal_data
            except Exception as exc:
                print(f"[WARN] Could not load calibration {cal_path}: {exc}")
        else:
            print(f"[WARN] Calibration file not found: {cal_path} — "
                  f"{side} arm wrist/gripper teleop disabled")

    try:
        run_dual_visualization(
            L1, L2, limits,
            arm_right=arm_right,
            arm_left=arm_left,
            joints_cal_right=joints_cal_right,
            joints_cal_left=joints_cal_left,
            use_ros=args.ros,
            wrist_scale=args.wrist_scale,
        )
    finally:
        if arm_right is not None:
            arm_right.close()
        if arm_left is not None:
            arm_left.close()


if __name__ == "__main__":
    main()
