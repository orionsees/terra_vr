#!/usr/bin/env python3
"""Teleoperation bridge — VR wrist tracking → SO101 arm commands (no ROS2).

Subscribes to IPC bus port 5555 (VR data):
    left_wrist, right_wrist   → x, y, z, qx, qy, qz, qw
    left_pinch, right_pinch   → distance_cm

Sends to arm bridges via IPC bus:
    port 5556  left arm  — topic "set_positions" or "set_positions_raw"
    port 5557  right arm — topic "set_positions" or "set_positions_raw"

Usage:
    python teleop_bridge_node.py
    python teleop_bridge_node.py --urdf ../urdf/so101_follower_pybullet.urdf
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import traceback
import time
from pathlib import Path
from typing import Optional

import pybullet as p
import pybullet_data

_THIS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_THIS_DIR))

from bus import (
    Subscriber, CommandClient,
    VR_DATA_PORT, LEFT_ARM_CMD_PORT, RIGHT_ARM_CMD_PORT,
)

DEFAULT_URDF      = str(_THIS_DIR / ".." / "urdf" / "so101_follower_pybullet.urdf")
DEFAULT_CAL_RIGHT = str(_THIS_DIR / ".." / "config" / "right_arm.json")
DEFAULT_CAL_LEFT  = str(_THIS_DIR / ".." / "config" / "left_arm.json")


# ---------------------------------------------------------------------------
# Teleoperation constants
# ---------------------------------------------------------------------------

WRIST_ROLL_YAW_MIN   = -170.0
WRIST_ROLL_YAW_MAX   =  170.0
WRIST_FLEX_ROLL_MIN  =  -65.0
WRIST_FLEX_ROLL_MAX  =   50.0

SPIKE_THRESHOLD_YAW   = 15.0
SPIKE_THRESHOLD_ROLL  = 15.0
SPIKE_THRESHOLD_PINCH =  4.0

DEADBAND_WRIST_ROLL  = 20
DEADBAND_WRIST_FLEX  = 20
DEADBAND_GRIPPER     =  3

DEADBAND_WRIST_X = 0.008
DEADBAND_WRIST_Y = 0.008

PAN_WRIST_SCALE   = 300.0
TELE_SMOOTH_ALPHA = 0.25
MAX_EE_VEL        = 0.3
TICK_DT           = 0.05   # 20 Hz

GROUND_Y = 0.0


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def forward_kinematics(t1: float, t2: float, L1: float, L2: float):
    x = L1 * math.cos(t1) + L2 * math.cos(t1 + t2)
    y = L1 * math.sin(t1) + L2 * math.sin(t1 + t2)
    return x, y


def inverse_kinematics(x: float, y: float, L1: float, L2: float) -> Optional[tuple]:
    r_sq = x * x + y * y
    c2   = _clamp((r_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2), -1.0, 1.0)
    s2   = -math.sqrt(max(0.0, 1.0 - c2 * c2))
    t2   = math.atan2(s2, c2)
    k1   = L1 + L2 * c2
    k2   = L2 * s2
    t1   = math.atan2(y, x) - math.atan2(k2, k1)
    return t1, t2


def _above_ground(t1: float, t2: float, L1: float, L2: float) -> bool:
    ey = L1 * math.sin(t1)
    return ey >= GROUND_Y and (ey + L2 * math.sin(t1 + t2)) >= GROUND_Y


def _closest_reachable(x: float, y: float, L1: float, L2: float):
    r = math.hypot(x, y)
    if r < 1e-9:
        return L1, GROUND_Y
    rc = _clamp(r, abs(L1 - L2), L1 + L2)
    cx, cy = x / r * rc, y / r * rc
    return cx, max(cy, GROUND_Y)


def quaternion_to_rpy(x, y, z, w):
    roll  = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(_clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw   = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# PyBullet geometry helpers (unchanged — no ROS2 dependency)
# ---------------------------------------------------------------------------

def _get_joint_index(robot_id: int, name: str) -> int:
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[1].decode() == name:
            return i
    raise ValueError(f"Joint '{name}' not found")


def extract_geometry(urdf_path: str):
    import numpy as np
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(urdf_path, useFixedBase=True)
    i1 = _get_joint_index(robot, "elbow_flex")
    i2 = _get_joint_index(robot, "wrist_flex")
    L1 = float(np.linalg.norm(p.getJointInfo(robot, i1)[14]))
    L2 = float(np.linalg.norm(p.getJointInfo(robot, i2)[14]))
    it1 = _get_joint_index(robot, "shoulder_lift")
    it2 = _get_joint_index(robot, "elbow_flex")
    t1_min, t1_max = float(p.getJointInfo(robot, it1)[8]), float(p.getJointInfo(robot, it1)[9])
    t2_min, t2_max = float(p.getJointInfo(robot, it2)[8]), float(p.getJointInfo(robot, it2)[9])
    p.disconnect()
    return L1, L2, (t1_min, t1_max, t2_min, t2_max)


# ---------------------------------------------------------------------------
# Teleoperation bridge (no ROS2)
# ---------------------------------------------------------------------------

class TeleopBridge:

    def __init__(self, L1: float, L2: float,
                 cal_right: Optional[dict], cal_left: Optional[dict]) -> None:
        self._L1    = L1
        self._L2    = L2
        self._reach = L1 + L2
        self._running = True

        # Command clients (thread-safe)
        self._cmd: dict[str, CommandClient] = {
            "left":  CommandClient(LEFT_ARM_CMD_PORT),
            "right": CommandClient(RIGHT_ARM_CMD_PORT),
        }

        self._cal  = {"right": cal_right, "left": cal_left}
        self._lock = threading.Lock()

        init_x = round(L2, 3)
        init_y = round(L1, 3)
        self._state: dict[str, dict] = {}
        for side in ("left", "right"):
            self._state[side] = {
                "desired":    {"x": init_x, "y": init_y, "pan": 0.0},
                "smooth":     {"x": init_x, "y": init_y},
                "last_t":     {"t1": math.radians(90.0), "t2": math.radians(-90.0)},
                "smooth_cmd": {"j1": None, "j2": None, "pan": None},
                "prev_cmd":   {"j1": None, "j2": None, "pan": None},
                "prev_z":     None, "prev_y_vr": None, "prev_x": None,
                "ref_x":      None,
                "last_yaw":   None, "last_roll":  None,
                "sent_wr":    None, "sent_wf":    None, "sent_grip": None,
                "last_pinch": None,
                "delta_x":    0.0,  "delta_y":   0.0,
                "new_pan":    0.0,
            }

        threading.Thread(target=self._vr_loop,   daemon=True).start()
        threading.Thread(target=self._tick_loop, daemon=True).start()

        print(
            f"[teleop bridge] ready — "
            f"L1={L1:.4f} m  L2={L2:.4f} m  "
            f"vr_port={VR_DATA_PORT}  "
            f"left_cmd={LEFT_ARM_CMD_PORT}  right_cmd={RIGHT_ARM_CMD_PORT}"
        )

    # ------------------------------------------------------------------
    # VR data subscription
    # ------------------------------------------------------------------

    def _vr_loop(self) -> None:
        sub = Subscriber(VR_DATA_PORT)
        for msg in sub.iter_messages():
            if not self._running:
                break
            topic = msg.get("topic", "")
            data  = msg.get("data", {})
            if topic in ("left_wrist", "right_wrist"):
                side = "left" if topic.startswith("left") else "right"
                self._wrist_cb(data, side)
            elif topic in ("left_pinch", "right_pinch"):
                side = "left" if topic.startswith("left") else "right"
                self._pinch_cb(data, side)

    # ------------------------------------------------------------------
    # VR callbacks (data is a plain dict from the bus)
    # ------------------------------------------------------------------

    def _wrist_cb(self, data: dict, side: str) -> None:
        st  = self._state[side]
        cal = self._cal[side]

        cur_z = data["z"]
        cur_y = data["y"]
        cur_x = data["x"]

        if st["prev_z"] is not None:
            dz = cur_z - st["prev_z"]
            dy = cur_y - st["prev_y_vr"]
            with self._lock:
                st["delta_x"] += dz
                st["delta_y"] += dy

        st["prev_z"]    = cur_z
        st["prev_y_vr"] = cur_y
        st["prev_x"]    = cur_x

        if st["ref_x"] is None:
            st["ref_x"] = cur_x
            print(f"[teleop bridge] [{side.upper()}] pan reference set — wrist X = {cur_x:.4f} m")
        else:
            pan = _clamp((cur_x - st["ref_x"]) * PAN_WRIST_SCALE, -90.0, 90.0)
            with self._lock:
                st["new_pan"] = pan

        if cal is None:
            return

        roll_d, pitch_d, yaw_d = (
            math.degrees(v)
            for v in quaternion_to_rpy(data["qx"], data["qy"], data["qz"], data["qw"])
        )

        # wrist_roll ← yaw
        if WRIST_ROLL_YAW_MIN <= yaw_d <= WRIST_ROLL_YAW_MAX:
            if (st["last_yaw"] is None
                    or abs(yaw_d - st["last_yaw"]) <= SPIKE_THRESHOLD_YAW):
                st["last_yaw"] = yaw_d
                frac = (yaw_d - WRIST_ROLL_YAW_MIN) / (WRIST_ROLL_YAW_MAX - WRIST_ROLL_YAW_MIN)
                wr   = int(
                    cal["wrist_roll"]["range_min"]
                    + frac * (cal["wrist_roll"]["range_max"] - cal["wrist_roll"]["range_min"])
                )
                if st["sent_wr"] is None or abs(wr - st["sent_wr"]) >= DEADBAND_WRIST_ROLL:
                    st["sent_wr"] = wr
                    self._cmd[side].send({"topic": "set_positions_raw", "data": {"5": wr}})

        # wrist_flex ← roll/pitch blended by yaw
        theta    = math.radians(st["last_yaw"] if st["last_yaw"] is not None else 0.0)
        eff_flex = roll_d * math.cos(theta) - pitch_d * math.sin(theta)
        if WRIST_FLEX_ROLL_MIN <= eff_flex <= WRIST_FLEX_ROLL_MAX:
            if (st["last_roll"] is None
                    or abs(eff_flex - st["last_roll"]) <= SPIKE_THRESHOLD_ROLL):
                st["last_roll"] = eff_flex
                frac = (eff_flex - WRIST_FLEX_ROLL_MIN) / (WRIST_FLEX_ROLL_MAX - WRIST_FLEX_ROLL_MIN)
                wf   = int(
                    cal["wrist_flex"]["range_min"]
                    + frac * (cal["wrist_flex"]["range_max"] - cal["wrist_flex"]["range_min"])
                )
                if st["sent_wf"] is None or abs(wf - st["sent_wf"]) >= DEADBAND_WRIST_FLEX:
                    st["sent_wf"] = wf
                    self._cmd[side].send({"topic": "set_positions_raw", "data": {"4": wf}})

    def _pinch_cb(self, data: dict, side: str) -> None:
        cal = self._cal[side]
        if cal is None:
            return
        st    = self._state[side]
        pinch = _clamp(float(data["distance_cm"]), 2.0, 10.0)
        if (st["last_pinch"] is not None
                and abs(pinch - st["last_pinch"]) > SPIKE_THRESHOLD_PINCH):
            return
        st["last_pinch"] = pinch
        frac = (pinch - 2.0) / 8.0
        gp   = int(
            cal["gripper"]["range_min"]
            + frac * (cal["gripper"]["range_max"] - cal["gripper"]["range_min"])
        )
        if st["sent_grip"] is None or abs(gp - st["sent_grip"]) >= DEADBAND_GRIPPER:
            st["sent_grip"] = gp
            self._cmd[side].send({"topic": "set_positions_raw", "data": {"6": gp}})

    # ------------------------------------------------------------------
    # 20 Hz IK tick
    # ------------------------------------------------------------------

    def _tick_loop(self) -> None:
        while self._running:
            self._tick()
            time.sleep(TICK_DT)

    def _tick(self) -> None:
        for side in ("left", "right"):
            self._tick_side(side)

    def _tick_side(self, side: str) -> None:
        st    = self._state[side]
        L1    = self._L1
        L2    = self._L2
        reach = self._reach

        with self._lock:
            dx      = st["delta_x"]
            dy      = st["delta_y"]
            new_pan = st["new_pan"]
            apply_x = abs(dx) >= DEADBAND_WRIST_X
            apply_y = abs(dy) >= DEADBAND_WRIST_Y
            if apply_x:
                st["delta_x"] = 0.0
            if apply_y:
                st["delta_y"] = 0.0

        if apply_x:
            st["desired"]["x"] = float(_clamp(st["desired"]["x"] + dx, -reach, reach))
        if apply_y:
            st["desired"]["y"] = float(_clamp(st["desired"]["y"] + dy, -reach, reach))
        if abs(new_pan - st["desired"]["pan"]) >= 0.5:
            st["desired"]["pan"] = new_pan

        x_des = st["desired"]["x"]
        y_des = max(st["desired"]["y"], GROUND_Y)

        sol_direct = inverse_kinematics(x_des, y_des, L1, L2)
        if sol_direct is not None and not _above_ground(sol_direct[0], sol_direct[1], L1, L2):
            sol_direct = None

        if sol_direct is not None:
            st["smooth"]["x"] = x_des
            st["smooth"]["y"] = y_des
        else:
            cx, cy = _closest_reachable(x_des, y_des, L1, L2)
            dx_ = cx - st["smooth"]["x"]
            dy_ = cy - st["smooth"]["y"]
            dist = math.hypot(dx_, dy_)
            step = MAX_EE_VEL * TICK_DT
            if dist <= step:
                st["smooth"]["x"], st["smooth"]["y"] = cx, cy
            else:
                st["smooth"]["x"] += dx_ / dist * step
                st["smooth"]["y"] += dy_ / dist * step

        sol = inverse_kinematics(st["smooth"]["x"], st["smooth"]["y"], L1, L2)
        if sol is not None and not _above_ground(sol[0], sol[1], L1, L2):
            sol = None

        if sol is not None:
            t1, t2 = sol
            st["last_t"]["t1"] = t1
            st["last_t"]["t2"] = t2
        else:
            t1 = st["last_t"]["t1"]
            t2 = st["last_t"]["t2"]

        servo_j1  = 90.0 - math.degrees(t1)
        servo_j2  = -math.degrees(t2) - 90.0
        servo_pan = st["desired"]["pan"]

        sc = st["smooth_cmd"]
        if sc["j1"] is None:
            sc["j1"] = servo_j1; sc["j2"] = servo_j2; sc["pan"] = servo_pan
        else:
            sc["j1"]  = TELE_SMOOTH_ALPHA * servo_j1  + (1 - TELE_SMOOTH_ALPHA) * sc["j1"]
            sc["j2"]  = TELE_SMOOTH_ALPHA * servo_j2  + (1 - TELE_SMOOTH_ALPHA) * sc["j2"]
            sc["pan"] = TELE_SMOOTH_ALPHA * servo_pan + (1 - TELE_SMOOTH_ALPHA) * sc["pan"]

        pc = st["prev_cmd"]
        changed = (
            pc["j1"] is None
            or abs(sc["j1"]  - pc["j1"])  > 0.05
            or abs(sc["j2"]  - pc["j2"])  > 0.05
            or abs(sc["pan"] - pc["pan"]) > 0.1
        )

        if changed:
            self._cmd[side].send({
                "topic": "set_positions",
                "data": {
                    "shoulder_lift": round(sc["j1"],  2),
                    "elbow_flex":    round(sc["j2"],  2),
                    "shoulder_pan":  round(sc["pan"], 2),
                },
            })
            pc["j1"]  = sc["j1"]
            pc["j2"]  = sc["j2"]
            pc["pan"] = sc["pan"]

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._running = False
        for c in self._cmd.values():
            c.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SO101 VR teleoperation bridge (no ROS2)"
    )
    parser.add_argument("--urdf",      default=DEFAULT_URDF)
    parser.add_argument("--cal-right", default=DEFAULT_CAL_RIGHT)
    parser.add_argument("--cal-left",  default=DEFAULT_CAL_LEFT)
    args = parser.parse_args()

    print(f"[teleop bridge] loading URDF: {args.urdf}")
    L1, L2, limits = extract_geometry(args.urdf)
    print(
        f"[teleop bridge] L1={L1:.4f} m  L2={L2:.4f} m  "
        f"θ1=[{math.degrees(limits[0]):.1f}°,{math.degrees(limits[1]):.1f}°]  "
        f"θ2=[{math.degrees(limits[2]):.1f}°,{math.degrees(limits[3]):.1f}°]"
    )

    def _load_cal(path):
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as exc:
                print(f"[WARN] Could not load calibration {path}: {exc}")
        else:
            print(f"[WARN] Calibration not found: {path}")
        return None

    cal_right = _load_cal(args.cal_right)
    cal_left  = _load_cal(args.cal_left)

    bridge = TeleopBridge(L1, L2, cal_right=cal_right, cal_left=cal_left)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[teleop bridge] stopped.")
    except Exception:
        traceback.print_exc()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
