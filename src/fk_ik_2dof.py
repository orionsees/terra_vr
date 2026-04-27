#!/usr/bin/env python3
"""2-DOF Forward / Inverse Kinematics for the SO-101 arm.

The two active joints are:
    theta1  →  shoulder_lift  (parent: upper_arm_link)
    theta2  →  elbow_flex     (parent: lower_arm_link)

Link lengths are extracted automatically from the URDF via PyBullet's
getJointInfo() so the geometry never goes out of sync with the robot model.

Usage (standalone):
    python fk_ik_2dof.py                     # headless demo
    python fk_ik_2dof.py --gui               # open PyBullet GUI viewer
    python fk_ik_2dof.py --viz               # interactive matplotlib visualisation
"""

from __future__ import annotations

import argparse
import math
import os
import sys
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
# Path to the URDF (resolved relative to this file)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_URDF = os.path.join(_THIS_DIR, "..", "urdf", "so101_follower_pybullet.urdf")
DEFAULT_CAL  = os.path.join(_THIS_DIR, "..", "config", "right_arm.json")

# Joint names that define the two links we care about
_JOINT_LINK1 = "elbow_flex"   # origin in upper_arm_link frame → length of link1
_JOINT_LINK2 = "wrist_flex"   # origin in lower_arm_link frame → length of link2

# Active joints for our 2-DOF model
_JOINT_THETA1 = "shoulder_lift"
_JOINT_THETA2 = "elbow_flex"

# Ground plane: arm cannot go below this Y value (10 units = 0.10 m below origin)
GROUND_Y = 0.0

# ---------------------------------------------------------------------------
# Teleoperation constants (wrist / gripper control via VR)
# ---------------------------------------------------------------------------

# Twist range (degrees) → wrist_roll servo range
WRIST_ROLL_YAW_MIN   = -170.0
WRIST_ROLL_YAW_MAX   =  170.0

# Flex range (degrees) → wrist_flex servo range
WRIST_FLEX_ROLL_MIN  =  -65.0
WRIST_FLEX_ROLL_MAX  =   50.0

# Spike rejection: max allowed change per callback
SPIKE_THRESHOLD_YAW   = 15.0   # degrees — wrist_roll
SPIKE_THRESHOLD_ROLL  = 15.0   # degrees — wrist_flex
SPIKE_THRESHOLD_PINCH =  2.0   # cm      — gripper

# Deadband: min servo-tick change required to send a new command
DEADBAND_WRIST_ROLL  = 20   # ticks
DEADBAND_WRIST_FLEX  = 20   # ticks
DEADBAND_GRIPPER     =  3   # ticks

# Deadband: min wrist position displacement (metres) required to move the
# 2-DOF arm sliders.  Eliminates jitter from small hand tremor / tracking noise.
DEADBAND_WRIST_X = 0.008   # m  (5 mm)
DEADBAND_WRIST_Y = 0.008   # m  (5 mm)

# Scale factor: degrees of shoulder_pan per metre of wrist X displacement
# 0.45 m swing ≈ 90°  →  200 deg/m
PAN_WRIST_SCALE = 300.0   # deg/m


# ---------------------------------------------------------------------------
# PyBullet helpers
# ---------------------------------------------------------------------------

def _get_joint_index(robot_id: int, joint_name: str) -> int:
    """Return the PyBullet joint index for *joint_name*, or raise."""
    n = p.getNumJoints(robot_id)
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        if info[1].decode("utf-8") == joint_name:
            return i
    raise ValueError(f"Joint '{joint_name}' not found in URDF.")


def extract_joint_limits(robot_id: int) -> Tuple[float, float, float, float]:
    """Return (t1_min, t1_max, t2_min, t2_max) in radians from the URDF."""
    idx1 = _get_joint_index(robot_id, _JOINT_THETA1)
    idx2 = _get_joint_index(robot_id, _JOINT_THETA2)
    info1 = p.getJointInfo(robot_id, idx1)
    info2 = p.getJointInfo(robot_id, idx2)
    return float(info1[8]), float(info1[9]), float(info2[8]), float(info2[9])


def extract_link_lengths(robot_id: int) -> Tuple[float, float]:
    """Extract L1 and L2 from joint parent-frame offsets stored in the URDF."""
    idx_link1 = _get_joint_index(robot_id, _JOINT_LINK1)
    idx_link2 = _get_joint_index(robot_id, _JOINT_LINK2)
    L1 = float(np.linalg.norm(np.array(p.getJointInfo(robot_id, idx_link1)[14])))
    L2 = float(np.linalg.norm(np.array(p.getJointInfo(robot_id, idx_link2)[14])))
    return L1, L2


# ---------------------------------------------------------------------------
# 2-DOF Forward Kinematics
# ---------------------------------------------------------------------------

def forward_kinematics(theta1: float, theta2: float,
                       L1: float, L2: float) -> Tuple[float, float]:
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
    r_sq = x ** 2 + y ** 2
    cos_theta2 = (r_sq - L1 ** 2 - L2 ** 2) / (2.0 * L1 * L2)

    if cos_theta2 < -1.0 or cos_theta2 > 1.0:
        return None

    sin_theta2 = math.sqrt(max(0.0, 1.0 - cos_theta2 ** 2))
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
    """Return True if both the elbow joint and end-effector are at or above GROUND_Y."""
    # Elbow (end of link 1)
    elbow_y = L1 * math.sin(t1)
    # End-effector (end of link 2)
    _, ee_y = forward_kinematics(t1, t2, L1, L2)
    return elbow_y >= GROUND_Y and ee_y >= GROUND_Y


# ---------------------------------------------------------------------------
# Teleoperation helpers
# ---------------------------------------------------------------------------

def quaternion_to_rpy(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """Convert quaternion to roll, pitch, yaw (radians)."""
    roll  = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw   = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Matplotlib interactive visualisation
# ---------------------------------------------------------------------------

def run_visualization(
    L1: float,
    L2: float,
    limits: Tuple[float, float, float, float],
    arm: Optional["SO101Arm"] = None,
    use_ros: bool = False,
    wrist_scale: float = 1.0,
    joints_cal: Optional[dict] = None,
) -> None:
    """Open an interactive matplotlib window with X/Y sliders.

    Prints J1/J2 to the terminal on every slider move.
    When *arm* is provided, servo angles are sent to the hardware.
    When *use_ros* is True, the /right_wrist topic drives the sliders:
        wrist Z displacement → X slider
        wrist Y displacement → Y slider
    """
    t1_min, t1_max, t2_min, t2_max = limits
    reach = L1 + L2
    margin = reach * 0.18

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.27)

    ax.set_xlim(-(reach + margin), reach + margin)
    ax.set_ylim(-(reach + margin), reach + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"SO-101 2-DOF Arm  —  drag X / Y sliders\n"
        f"Joint limits  θ1 [{math.degrees(t1_min):.1f}°, {math.degrees(t1_max):.1f}°]  "
        f"θ2 [{math.degrees(t2_min):.1f}°, {math.degrees(t2_max):.1f}°]",
        fontsize=9,
    )

    # -----------------------------------------------------------------------
    # Ground line at y = GROUND_Y
    # -----------------------------------------------------------------------
    ax.axhline(GROUND_Y, color="saddlebrown", linewidth=2.5, linestyle="-",
               label=f"Ground (y = {GROUND_Y:.2f} m)", zorder=3)
    # Shade the region below the ground to make it visually clear
    ax.axhspan(-(reach + margin), GROUND_Y, color="saddlebrown", alpha=0.08, zorder=0)

    # -----------------------------------------------------------------------
    # Precompute reachable workspace (for scatter plot only)
    # -----------------------------------------------------------------------
    _ws_x, _ws_y = [], []
    for _t1 in np.linspace(t1_min, t1_max, 500):
        for _t2 in np.linspace(t2_min, t2_max, 200):
            _ex, _ey = forward_kinematics(_t1, _t2, L1, L2)
            # Skip configurations that violate the ground constraint
            if not _above_ground(_t1, _t2, L1, L2):
                continue
            _ws_x.append(_ex)
            _ws_y.append(_ey)

    ax.scatter(_ws_x, _ws_y, s=2, color="lightgreen", alpha=0.20,
               label="Reachable workspace", zorder=1)
    ax.add_patch(plt.Circle((0, 0), reach,
                             fill=False, color="green", linestyle="--",
                             linewidth=1.0, alpha=0.4, label=f"Max reach ({reach:.3f} m)"))
    inner_r = abs(L1 - L2)
    if inner_r > 1e-4:
        ax.add_patch(plt.Circle((0, 0), inner_r,
                                 fill=False, color="red", linestyle="--",
                                 linewidth=1.0, alpha=0.4,
                                 label=f"Min reach ({inner_r:.3f} m)"))

    # Arm graphics (blue)
    link1,  = ax.plot([], [], color="steelblue", linewidth=6,
                      solid_capstyle="round", label="Arm")
    link2,  = ax.plot([], [], color="royalblue", linewidth=6,
                      solid_capstyle="round")
    joints, = ax.plot([], [], "o", color="white", markersize=9,
                      markeredgecolor="steelblue", markeredgewidth=2, zorder=5)
    ee,     = ax.plot([], [], "*", color="steelblue", markersize=18, zorder=6)
    target_pt, = ax.plot([], [], "rx", markersize=13, markeredgewidth=2.5,
                         zorder=7, label="Target")

    status_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes, verticalalignment="top", fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    ax.legend(loc="upper right", fontsize=8)

    # -----------------------------------------------------------------------
    # Top-view inset  (shoulder pan)
    # -----------------------------------------------------------------------
    ax_top = ax.inset_axes([0.01, 0.60, 0.28, 0.36])
    ax_top.set_aspect("equal")
    ax_top.set_title("Top View\n(shoulder pan)", fontsize=7, pad=2)
    ax_top.set_xlim(-1.45, 1.45)
    ax_top.set_ylim(-0.38, 1.48)
    ax_top.axis("off")
    # Shaded reachable arc (±90°)
    _pa = np.linspace(-math.pi / 2, math.pi / 2, 60)
    _arc_xs = np.concatenate([[0.0], np.sin(_pa)])
    _arc_ys = np.concatenate([[0.0], np.cos(_pa)])
    ax_top.fill(_arc_xs, _arc_ys, color="lightcyan", alpha=0.4, zorder=0)
    ax_top.plot(np.sin(_pa), np.cos(_pa), color="steelblue", linewidth=1.2, alpha=0.7)
    # Tick marks at 0°, ±45°, ±90°
    for _tdeg in [-90, -45, 0, 45, 90]:
        _tr = math.radians(_tdeg)
        _tx, _ty = math.sin(_tr), math.cos(_tr)
        ax_top.plot([0.82 * _tx, _tx], [0.82 * _ty, _ty],
                    color="steelblue", linewidth=1.0, alpha=0.8)
        _lx = 1.25 * _tx if _tdeg != 0 else 0.0
        _ly = 1.22 * _ty
        ax_top.text(_lx, _ly, f"{_tdeg:+d}°",
                    ha="center", va="center", fontsize=6, color="dimgray")
    ax_top.text(0, 1.38, "fwd", ha="center", va="bottom", fontsize=6, color="gray")
    ax_top.plot(0, 0, "o", color="steelblue", markersize=5, zorder=4)
    # Live pan direction indicator
    _pan_line, = ax_top.plot([0, 0], [0, 1.0], color="tomato", linewidth=3,
                              solid_capstyle="round", zorder=5)
    _pan_dot,  = ax_top.plot([0], [1.0], "o", color="tomato", markersize=7, zorder=6)
    _pan_text  = ax_top.text(0, -0.24, "Pan:  0.0°", ha="center", va="top", fontsize=7,
                              family="monospace",
                              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Sliders
    ax_span = plt.axes([0.15, 0.21, 0.72, 0.03])
    ax_sx   = plt.axes([0.15, 0.14, 0.72, 0.03])
    ax_sy   = plt.axes([0.15, 0.07, 0.72, 0.03])
    # Initial position = physical zero (L-shape): link1 along +Y, link2 along +X
    # In model space: theta1=+90deg, theta2=-90deg => EE at (L2, L1)
    slider_pan = Slider(ax_span, "Pan (°)", -90.0, 90.0, valinit=0.0, valstep=0.5)
    slider_x   = Slider(ax_sx, "X (m)", -reach, reach,
                        valinit=round(L2, 3), valstep=0.001)
    slider_y   = Slider(ax_sy, "Y (m)", -reach, reach,
                        valinit=round(L1, 3), valstep=0.001)

    def _arm_points(t1: float, t2: float):
        xj = L1 * math.cos(t1)
        yj = L1 * math.sin(t1)
        xe, ye = forward_kinematics(t1, t2, L1, L2)
        return (xj, yj), (xe, ye)

    # Last valid model angles — initialised to physical zero (L-shape)
    # servo (0°,0°) → model (90°, -90°)
    _last = {"t1": math.radians(90.0), "t2": math.radians(-90.0)}

    def _read_arm_model_angles() -> Optional[Tuple[float, float]]:
        """Read current servo positions and convert to model (theta1, theta2)."""
        try:
            pos = arm.read_positions()
            s1 = pos.get("shoulder_lift.pos")
            s2 = pos.get("elbow_flex.pos")
            if s1 is None or s2 is None:
                return None
            # servo → model: model_θ1 = 90 - servo_J1,  model_θ2 = -servo_J2 - 90
            return math.radians(90.0 - float(s1)), math.radians(-float(s2) - 90.0)
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Smooth-velocity state for out-of-reach targets
    # -----------------------------------------------------------------------
    # Maximum end-effector speed (m/s) used when travelling toward the
    # workspace boundary after an unreachable target is requested.
    MAX_EE_VEL = 0.3      # m/s
    _SMOOTH_DT  = 0.05    # s  — timer tick period (20 Hz)

    # Desired target: updated immediately on every slider move.
    _desired = {"x": round(L2, 3), "y": round(L1, 3), "pan": 0.0}
    # Smooth (commanded) EE position: advances toward _desired at MAX_EE_VEL
    # when out-of-reach, snaps instantly when reachable.
    _smooth  = {"x": round(L2, 3), "y": round(L1, 3)}
    # Last sent servo angles — used to suppress redundant HW/print calls.
    _prev_cmd = {"j1": None, "j2": None, "pan": None}

    def _closest_reachable(x_t: float, y_t: float) -> Tuple[float, float]:
        """Return the nearest point inside the reachable annulus toward (x_t, y_t)."""
        r = math.hypot(x_t, y_t)
        if r < 1e-9:
            return L1, 0.0
        r_cl = max(abs(L1 - L2), min(L1 + L2, r))
        return x_t / r * r_cl, y_t / r * r_cl

    def update(_):
        """Slider callback: record desired target and refresh the target marker only."""
        _desired["x"] = slider_x.val
        _desired["y"] = slider_y.val
        target_pt.set_data([_desired["x"]], [_desired["y"]])
        fig.canvas.draw_idle()

    def update_pan(_):
        """Pan slider callback: record desired pan angle."""
        _desired["pan"] = slider_pan.val
        fig.canvas.draw_idle()

    def _tick():
        """20 Hz timer: step smooth EE, solve IK, update display and hardware.

        Reachable target  → smooth EE snaps directly to slider position.
        Out-of-reach target → smooth EE advances toward the workspace-boundary
                              projection of the slider position at MAX_EE_VEL m/s,
                              giving the arm a smooth, velocity-limited travel.
        """
        x_des = _desired["x"]
        y_des = _desired["y"]
        y_des_cl = max(y_des, GROUND_Y)

        # Check whether the desired target is directly reachable.
        sol_direct = inverse_kinematics(x_des, y_des_cl, L1, L2, elbow_up=False, limits=None)
        if sol_direct is not None and not _above_ground(sol_direct[0], sol_direct[1], L1, L2):
            sol_direct = None

        if sol_direct is not None:
            # Reachable — snap the smooth position directly to the target.
            _smooth["x"] = x_des
            _smooth["y"] = y_des_cl
            out_of_reach = False
        else:
            # Out-of-reach — advance smooth EE toward the nearest workspace
            # boundary point (in the direction of the desired target).
            cx, cy = _closest_reachable(x_des, y_des)
            cy = max(cy, GROUND_Y)
            dx = cx - _smooth["x"]
            dy = cy - _smooth["y"]
            dist = math.hypot(dx, dy)
            max_step = MAX_EE_VEL * _SMOOTH_DT
            if dist <= max_step:
                _smooth["x"] = cx
                _smooth["y"] = cy
            else:
                _smooth["x"] += dx / dist * max_step
                _smooth["y"] += dy / dist * max_step
            out_of_reach = True

        # Solve IK at the current smooth position.
        sol = inverse_kinematics(_smooth["x"], _smooth["y"], L1, L2, elbow_up=False, limits=None)
        if sol is not None and not _above_ground(sol[0], sol[1], L1, L2):
            sol = None

        if sol is not None:
            t1, t2 = sol
            _last["t1"] = t1
            _last["t2"] = t2
        else:
            if arm is not None:
                fb = _read_arm_model_angles()
                if fb is not None:
                    _last["t1"], _last["t2"] = fb
            t1 = _last["t1"]
            t2 = _last["t2"]

        (xj, yj), (xe, ye) = _arm_points(t1, t2)
        link1.set_data([0, xj], [0, yj])
        link2.set_data([xj, xe], [yj, ye])
        joints.set_data([0, xj, xe], [0, yj, ye])
        ee.set_data([xe], [ye])

        t1_d = math.degrees(t1)
        t2_d = math.degrees(t2)
        # Servo angles: servo_J1 = 90 - model_θ1,  servo_J2 = -model_θ2 - 90
        servo_j1  = 90.0 - t1_d
        servo_j2  = -t2_d - 90.0
        servo_pan = _desired["pan"]

        # Only print / send to hardware when the commanded angles actually changed
        # (suppresses 20 Hz noise when the arm is stationary).
        changed = (
            _prev_cmd["j1"] is None
            or abs(servo_j1  - _prev_cmd["j1"])  > 0.05
            or abs(servo_j2  - _prev_cmd["j2"])  > 0.05
            or _prev_cmd["pan"] is None
            or abs(servo_pan - _prev_cmd["pan"]) > 0.1
        )

        if out_of_reach:
            label = "OUT OF REACH — travelling to boundary"
            if changed:
                print(f"model  J1={t1_d:+7.2f}°  J2={t2_d:+7.2f}°  "
                      f"servo  J1={servo_j1:+7.2f}°  J2={servo_j2:+7.2f}°  "
                      f"pan={servo_pan:+.1f}°  "
                      f"EE=({xe:+.4f}, {ye:+.4f}) m  "
                      f"[SMOOTH\u2192BOUNDARY  target ({x_des:+.4f}, {y_des:+.4f})]")
            if arm is not None and changed:
                try:
                    arm.set_positions(
                        {"shoulder_lift": servo_j1, "elbow_flex": servo_j2,
                         "shoulder_pan": servo_pan},
                        expect_ack=False,
                    )
                    hw_status = "HW SMOOTH"
                except Exception as exc:
                    hw_status = f"HW ERR: {exc}"
            else:
                hw_status = "HW SMOOTH" if arm is not None else "(no HW)"
        else:
            label = "IK OK"
            if changed:
                print(f"model  J1={t1_d:+7.2f}°  J2={t2_d:+7.2f}°  "
                      f"servo  J1={servo_j1:+7.2f}°  J2={servo_j2:+7.2f}°  "
                      f"pan={servo_pan:+.1f}°  "
                      f"EE=({xe:+.4f}, {ye:+.4f}) m  "
                      f"target=({x_des:+.4f}, {y_des:+.4f}) m")
            if arm is not None and changed:
                try:
                    arm.set_positions(
                        {"shoulder_lift": servo_j1, "elbow_flex": servo_j2,
                         "shoulder_pan": servo_pan},
                        expect_ack=False,
                    )
                    hw_status = "HW OK"
                except Exception as exc:
                    hw_status = f"HW ERR: {exc}"
            else:
                hw_status = "HW OK" if arm is not None else "(no HW)"

        if changed:
            _prev_cmd["j1"]  = servo_j1
            _prev_cmd["j2"]  = servo_j2
            _prev_cmd["pan"] = servo_pan

        # Update top-view pan indicator
        _pan_rad = math.radians(servo_pan)
        _pan_line.set_data([0, math.sin(_pan_rad)], [0, math.cos(_pan_rad)])
        _pan_dot.set_data([math.sin(_pan_rad)], [math.cos(_pan_rad)])
        _pan_text.set_text(f"Pan: {servo_pan:+.1f}\u00b0")

        status_text.set_text(
            f"{label}  {hw_status}\n"
            f"model  J1={t1_d:+7.2f}°   J2={t2_d:+7.2f}°\n"
            f"servo  J1={servo_j1:+7.2f}°   J2={servo_j2:+7.2f}°\n"
            f"pan={servo_pan:+.1f}°   EE=({xe:+.4f}, {ye:+.4f}) m"
        )
        fig.canvas.draw_idle()

    slider_pan.on_changed(update_pan)
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    _tick()  # initial draw

    # 20 Hz timer drives smooth arm motion independently of slider callbacks.
    _smooth_timer = fig.canvas.new_timer(interval=int(_SMOOTH_DT * 1000))
    _smooth_timer.add_callback(_tick)
    _smooth_timer.start()

    # -------------------------------------------------------------------
    # ROS2 wrist-tracking control
    # -------------------------------------------------------------------
    if use_ros:
        if not _ROS_AVAILABLE:
            print("[WARN] --ros requested but rclpy is not available. Wrist tracking disabled.")
        else:
            # Shared accumulator: ROS thread writes, matplotlib timer drains
            _wrist_delta = {"dx": 0.0, "dy": 0.0}
            _pan_shared  = {"deg": 0.0}   # absolute pan angle (degrees)
            _wrist_lock = threading.Lock()

            class _WristNode(_RosNode):
                def __init__(self_node):
                    super().__init__("fk_ik_wrist_ctrl")
                    qos = QoSProfile(
                        depth=10,
                        reliability=ReliabilityPolicy.BEST_EFFORT,
                    )
                    # Position-delta state (slider control)
                    self_node._prev_z: Optional[float] = None
                    self_node._prev_y: Optional[float] = None
                    self_node._prev_x: Optional[float] = None
                    # Pan reference (None until first message after ENTER)
                    self_node._ref_wrist_x: Optional[float] = None
                    # Spike-rejection state
                    self_node._last_yaw: Optional[float] = None
                    self_node._last_roll: Optional[float] = None
                    self_node._last_gripper_pinch: Optional[float] = None
                    # Deadband state
                    self_node._sent_wrist_roll: Optional[int] = None
                    self_node._sent_wrist_flex: Optional[int] = None
                    self_node._sent_gripper: Optional[int] = None
                    self_node.create_subscription(
                        PoseStamped, "/right_wrist", self_node._cb, qos
                    )
                    self_node.create_subscription(
                        Float32, "/right_hand/pinch_distance",
                        self_node._pinch_cb, qos,
                    )
                    self_node.get_logger().info(
                        "Subscribed to /right_wrist and /right_hand/pinch_distance"
                    )

                def _cb(self_node, msg: PoseStamped):
                    # --- Position → X/Y slider (existing behaviour) ---
                    cur_z = msg.pose.position.z
                    cur_y = msg.pose.position.y
                    cur_x = msg.pose.position.x

                    if self_node._prev_z is not None:
                        dz      = cur_z - self_node._prev_z  # wrist Z → arm X
                        dy      = cur_y - self_node._prev_y  # wrist Y → arm Y
                        dx_pan  = cur_x - self_node._prev_x  # wrist X → pan
                        # Suppress arm reach updates when X (pan) is the dominant
                        # axis of motion — avoids arm drift during horizontal panning.
                        pan_dominant = abs(dx_pan) > abs(dz) and abs(dx_pan) > abs(dy)
                        if not pan_dominant:
                            with _wrist_lock:
                                _wrist_delta["dx"] += dz * wrist_scale
                                _wrist_delta["dy"] += dy * wrist_scale
                    self_node._prev_z = cur_z
                    self_node._prev_y = cur_y
                    self_node._prev_x = cur_x

                    # --- Shoulder pan via wrist X position ---
                    cur_x = msg.pose.position.x
                    if self_node._ref_wrist_x is None:
                        self_node._ref_wrist_x = cur_x
                        print(f"Pan reference captured — wrist X = {cur_x:.4f} m")
                    else:
                        pan_deg = (cur_x - self_node._ref_wrist_x) * PAN_WRIST_SCALE
                        pan_deg = max(-90.0, min(90.0, pan_deg))
                        with _wrist_lock:
                            _pan_shared["deg"] = pan_deg

                    # --- Orientation → wrist_roll / wrist_flex (teleoperation) ---
                    if arm is None or joints_cal is None:
                        return
                    q = msg.pose.orientation
                    roll, pitch, yaw = quaternion_to_rpy(q.x, q.y, q.z, q.w)
                    yaw_deg   = math.degrees(yaw)
                    roll_deg  = math.degrees(roll)
                    pitch_deg = math.degrees(pitch)

                    # wrist_roll via yaw
                    if WRIST_ROLL_YAW_MIN <= yaw_deg <= WRIST_ROLL_YAW_MAX:
                        if (self_node._last_yaw is None or
                                abs(yaw_deg - self_node._last_yaw) <= SPIKE_THRESHOLD_YAW):
                            self_node._last_yaw = yaw_deg
                            frac = ((yaw_deg - WRIST_ROLL_YAW_MIN) /
                                    (WRIST_ROLL_YAW_MAX - WRIST_ROLL_YAW_MIN))
                            wrist_roll_pos = int(
                                joints_cal["wrist_roll"]["range_min"] +
                                frac * (joints_cal["wrist_roll"]["range_max"] -
                                        joints_cal["wrist_roll"]["range_min"])
                            )
                            if (self_node._sent_wrist_roll is None or
                                    abs(wrist_roll_pos - self_node._sent_wrist_roll)
                                    >= DEADBAND_WRIST_ROLL):
                                self_node._sent_wrist_roll = wrist_roll_pos
                                arm.set_positions_raw(
                                    {5: wrist_roll_pos}, acc=254, speed=0, expect_ack=False
                                )
                        else:
                            self_node.get_logger().warn(
                                f"Yaw spike rejected: "
                                f"{self_node._last_yaw:.1f}\u00b0 \u2192 {yaw_deg:.1f}\u00b0"
                            )

                    # wrist_flex via roll/pitch blended by current yaw
                    # (when wrist_roll rotates, the flex axis rotates with it)
                    theta = math.radians(
                        self_node._last_yaw if self_node._last_yaw is not None else 0.0
                    )
                    effective_flex = roll_deg * math.cos(theta) - pitch_deg * math.sin(theta)
                    if WRIST_FLEX_ROLL_MIN <= effective_flex <= WRIST_FLEX_ROLL_MAX:
                        if (self_node._last_roll is None or
                                abs(effective_flex - self_node._last_roll) <= SPIKE_THRESHOLD_ROLL):
                            self_node._last_roll = effective_flex
                            frac = ((effective_flex - WRIST_FLEX_ROLL_MIN) /
                                    (WRIST_FLEX_ROLL_MAX - WRIST_FLEX_ROLL_MIN))
                            wrist_flex_pos = int(
                                joints_cal["wrist_flex"]["range_min"] +
                                frac * (joints_cal["wrist_flex"]["range_max"] -
                                        joints_cal["wrist_flex"]["range_min"])
                            )
                            if (self_node._sent_wrist_flex is None or
                                    abs(wrist_flex_pos - self_node._sent_wrist_flex)
                                    >= DEADBAND_WRIST_FLEX):
                                self_node._sent_wrist_flex = wrist_flex_pos
                                arm.set_positions_raw(
                                    {4: wrist_flex_pos}, acc=254, speed=0, expect_ack=False
                                )
                        else:
                            self_node.get_logger().warn(
                                f"Flex spike rejected: "
                                f"{self_node._last_roll:.1f}\u00b0 \u2192 {effective_flex:.1f}\u00b0"
                            )

                def _pinch_cb(self_node, msg: Float32):
                    """Gripper control from /right_hand/pinch_distance (cm)."""
                    if arm is None or joints_cal is None:
                        return
                    pinch = float(msg.data)
                    pinch = max(2.0, min(10.0, pinch))
                    if (self_node._last_gripper_pinch is not None and
                            abs(pinch - self_node._last_gripper_pinch) > SPIKE_THRESHOLD_PINCH):
                        self_node.get_logger().warn(
                            f"Pinch spike rejected: "
                            f"{self_node._last_gripper_pinch:.2f} \u2192 {pinch:.2f} cm"
                        )
                        return
                    self_node._last_gripper_pinch = pinch
                    frac = (pinch - 2.0) / 8.0
                    gripper_pos = int(
                        joints_cal["gripper"]["range_min"] +
                        frac * (joints_cal["gripper"]["range_max"] -
                                joints_cal["gripper"]["range_min"])
                    )
                    if (self_node._sent_gripper is None or
                            abs(gripper_pos - self_node._sent_gripper) >= DEADBAND_GRIPPER):
                        self_node._sent_gripper = gripper_pos
                        arm.set_positions_raw(
                            {6: gripper_pos}, acc=254, speed=0, expect_ack=False
                        )

            input(
                "\nPosition your arm so that shoulder_pan is at 0° (centre), "
                "then press ENTER..."
            )
            rclpy.init()
            _wrist_node = _WristNode()

            def _ros_spin():
                rclpy.spin(_wrist_node)

            _ros_thread = threading.Thread(target=_ros_spin, daemon=True,
                                           name="ros-wrist-spin")
            _ros_thread.start()
            print("[ROS] Wrist tracking active — wrist Z\u2192X slider, wrist Y\u2192Y slider")
            if arm is not None and joints_cal is not None:
                print("[ROS] Teleoperation active — wrist_roll, wrist_flex, gripper control enabled")
            elif arm is None:
                print("[ROS] No hardware connected — wrist_roll/wrist_flex/gripper control disabled")
            elif joints_cal is None:
                print("[ROS] No calibration loaded — wrist_roll/wrist_flex/gripper control disabled")

            def _apply_wrist_delta():
                """Called every 50 ms by matplotlib timer; applies accumulated delta.

                A deadband (DEADBAND_WRIST_X / _Y) suppresses slider updates
                caused by small hand tremor or VR tracking noise.  The
                accumulator is only drained when the displacement exceeds the
                threshold; otherwise the delta rolls over to the next tick so
                intentional slow motion is never silently discarded.
                """
                with _wrist_lock:
                    dx = _wrist_delta["dx"]
                    dy = _wrist_delta["dy"]
                    # Apply deadband per axis independently.
                    apply_x = abs(dx) >= DEADBAND_WRIST_X
                    apply_y = abs(dy) >= DEADBAND_WRIST_Y
                    if apply_x:
                        _wrist_delta["dx"] = 0.0
                    if apply_y:
                        _wrist_delta["dy"] = 0.0
                    new_pan_deg = _pan_shared["deg"]
                # Pan: apply if change exceeds 0.5° threshold
                apply_pan = abs(new_pan_deg - slider_pan.val) >= 0.5
                if not apply_x and not apply_y and not apply_pan:
                    return
                new_x = slider_x.val + (dx if apply_x else 0.0)
                new_y = slider_y.val + (dy if apply_y else 0.0)
                new_x = float(np.clip(new_x, -reach, reach))
                new_y = float(np.clip(new_y, -reach, reach))
                # set_val triggers on_changed → update()
                if apply_x:
                    slider_x.set_val(round(new_x, 3))
                if apply_y:
                    slider_y.set_val(round(new_y, 3))
                if apply_pan:
                    slider_pan.set_val(round(new_pan_deg, 1))

            _timer = fig.canvas.new_timer(interval=50)  # 20 Hz poll
            _timer.add_callback(_apply_wrist_delta)
            _timer.start()

    plt.show()


# ---------------------------------------------------------------------------
# Optional PyBullet visualisation helpers
# ---------------------------------------------------------------------------

def _draw_arm(robot_id: int,
              idx_t1: int, idx_t2: int,
              theta1: float, theta2: float) -> None:
    p.resetJointState(robot_id, idx_t1, theta1)
    p.resetJointState(robot_id, idx_t2, theta2)
    p.stepSimulation()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SO-101 2-DOF FK/IK demo")
    parser.add_argument("--urdf", default=DEFAULT_URDF,
                        help="Path to the URDF file")
    parser.add_argument("--gui", action="store_true",
                        help="Open the PyBullet GUI viewer")
    parser.add_argument("--viz", action="store_true",
                        help="Open interactive matplotlib visualisation")
    parser.add_argument("--port", default=None,
                        help="Serial port for the physical arm (e.g. /dev/ttyACM0). "
                             "Omit to run simulation-only.")
    parser.add_argument("--cal", default=DEFAULT_CAL,
                        help="Path to calibration JSON (default: right_arm.json)")
    parser.add_argument("--ros", action="store_true",
                        help="Control X/Y sliders from /right_wrist ROS2 topic "
                             "(wrist Z → X, wrist Y → Y)")
    parser.add_argument("--wrist-scale", type=float, default=1.0,
                        help="Scale factor applied to wrist displacement (default: 1.0)")
    args = parser.parse_args()

    mode = p.GUI if args.gui else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    robot_id = p.loadURDF(args.urdf, useFixedBase=True)

    L1, L2 = extract_link_lengths(robot_id)
    limits = extract_joint_limits(robot_id)
    t1_min, t1_max, t2_min, t2_max = limits

    print("=" * 55)
    print("Link lengths extracted from URDF")
    print(f"  L1 (shoulder_lift → elbow_flex) : {L1:.6f} m")
    print(f"  L2 (elbow_flex   → wrist_flex)  : {L2:.6f} m")
    print("Joint limits extracted from URDF")
    print(f"  shoulder_lift (θ1): [{math.degrees(t1_min):.2f}°, {math.degrees(t1_max):.2f}°]")
    print(f"  elbow_flex    (θ2): [{math.degrees(t2_min):.2f}°, {math.degrees(t2_max):.2f}°]")
    print(f"Ground constraint  : y >= {GROUND_Y:.3f} m")
    print("=" * 55)

    # FK demo
    print("\n--- Forward Kinematics ---")
    for theta1, theta2, label in [
        (0.0,       0.0,       "zero pose"),
        (math.pi/4, math.pi/4, "45° / 45°"),
        (math.pi/6, math.pi/3, "30° / 60°"),
        (0.0,       math.pi/2, "0° / 90°"),
    ]:
        x, y = forward_kinematics(theta1, theta2, L1, L2)
        print(f"  [{label}]  θ1={math.degrees(theta1):6.1f}°  "
              f"θ2={math.degrees(theta2):6.1f}°  →  x={x:.4f} m,  y={y:.4f} m")

    # IK round-trip demo
    print("\n--- Inverse Kinematics (round-trip verification) ---")
    idx_t1 = _get_joint_index(robot_id, _JOINT_THETA1)
    idx_t2 = _get_joint_index(robot_id, _JOINT_THETA2)
    for x_t, y_t, label in [
        (L1 + L2, 0.0,  "full reach along x"),
        (0.10,    0.15, "(0.10, 0.15)"),
        (0.20,    0.05, "(0.20, 0.05)"),
        (0.50,    0.50, "out-of-reach target"),
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
        print(f"    elbow-down: θ1={math.degrees(t1):7.2f}°  θ2={math.degrees(t2):7.2f}°  "
              f"→ FK=({x_fk:.4f},{y_fk:.4f})  err={err:.2e} m")
        if args.gui:
            _draw_arm(robot_id, idx_t1, idx_t2, t1, t2)

    print("\nDone.")
    if args.gui:
        input("PyBullet GUI is open — press Enter to exit.")
    p.disconnect()

    if args.viz:
        print("\nOpening matplotlib visualisation  (J1/J2 printed on every slider move)...")
        if args.port:
            print(f"Connecting to arm on {args.port} with cal {args.cal} ...")
            arm = SO101Arm(args.port, calibration_file=args.cal)
            arm.open()
            print("Arm connected.")
        else:
            arm = None
        # Load calibration for teleoperation joint ranges (wrist_roll, wrist_flex, gripper)
        import json as _json
        joints_cal: Optional[dict] = None
        if os.path.isfile(args.cal):
            try:
                with open(args.cal) as _f:
                    joints_cal = _json.load(_f)
            except Exception as _exc:
                print(f"[WARN] Could not load calibration {args.cal}: {_exc}")
        else:
            print(f"[WARN] Calibration file not found: {args.cal} — teleop wrist/gripper disabled")

        try:
            run_visualization(
                L1, L2, limits,
                arm=arm,
                use_ros=args.ros,
                wrist_scale=args.wrist_scale,
                joints_cal=joints_cal,
            )
        finally:
            if arm is not None:
                arm.close()


if __name__ == "__main__":
    main()