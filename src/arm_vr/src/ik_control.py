#!/usr/bin/env python3
"""
ik_control.py — VR wrist pose → SO-101 full-arm IK → dual-arm teleoperation

IK solver: PyBullet (DIRECT/headless mode).
IK joints: shoulder_pan(1), shoulder_lift(2), elbow_flex(3),
           wrist_flex(4), wrist_roll(5).
Gripper (6) is excluded from IK — held at neutral.

Frame calibration
─────────────────
  VR headsets use different axis conventions, so run calibration once:

      python3 ik_control.py --calibrate

  You will be asked to move your wrist in 3 directions.  The result is saved to
  config/vr_frame.json and loaded automatically on every subsequent run.
  If no calibration file exists, a sensible default is used.

Reference-latching position control
────────────────────────────────────
  On the first VR message the current wrist position is stored as the "home"
  reference.  Every subsequent position is treated as a displacement from that
  reference and mapped to a displacement of the arm's EE from its FK-home
  position.  Hold your wrist at a comfortable neutral pose before starting
  the node so the reference latches correctly.

Subscribes to IPC bus port 5555 (VR data):
    left_wrist / right_wrist   → x, y, z, qx, qy, qz, qw

Sends to arm bridges via IPC bus:
    port 5556  left arm  → topic "set_positions"
    port 5557  right arm → topic "set_positions"

Usage:
    python3 ik_control.py                    # both arms, default scale
    python3 ik_control.py --calibrate        # run frame calibration first
    python3 ik_control.py --side left        # left arm only
    python3 ik_control.py --scale 0.7        # 70 % workspace scale
    python3 ik_control.py --with-orientation # include VR wrist orientation in IK
"""

from __future__ import annotations

import argparse
import json
import math
import re
import select
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as pb
import pybullet_data
from scipy.spatial.transform import Rotation as ScipyR

sys.path.insert(0, str(Path(__file__).parent))
from bus import (
    Subscriber, CommandClient,
    VR_DATA_PORT, LEFT_ARM_CMD_PORT, RIGHT_ARM_CMD_PORT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_SRC_DIR  = Path(__file__).parent.resolve()
URDF_PATH = (_SRC_DIR / ".." / "urdf" / "so101_follower.urdf").resolve()
CAL_FILE  = (_SRC_DIR / ".." / "config" / "vr_frame.json").resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Robot constants  (from URDF)
# ─────────────────────────────────────────────────────────────────────────────

IK_JOINT_NAMES = [
    "shoulder_pan",   # motor 1
    "shoulder_lift",  # motor 2
    "elbow_flex",     # motor 3
    "wrist_flex",     # motor 4
    "wrist_roll",     # motor 5
]

JOINT_LOWER = [-2.055, -2.018, -1.653, -1.786, -3.194]
JOINT_UPPER = [ 2.058,  2.018,  1.654,  1.790,  4.120]

_GRIPPER_LOWER = -0.1745
_GRIPPER_UPPER =  1.7453

_ALL_LOWER  = JOINT_LOWER + [_GRIPPER_LOWER]
_ALL_UPPER  = JOINT_UPPER + [_GRIPPER_UPPER]
_ALL_RANGES = [u - l for l, u in zip(_ALL_LOWER, _ALL_UPPER)]
_ALL_REST   = [0.0] * len(_ALL_LOWER)

EE_LINK = "gripper_link"
DEFAULT_MAX_DELTA_M = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate frame mapping: VR → arm base frame
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_VR_TO_ARM = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0],
    [ 0.0,  1.0,  0.0],
], dtype=float)

_VR_TO_ARM:  np.ndarray              = DEFAULT_VR_TO_ARM.copy()
_VR_BOUNDS:  Optional[tuple[np.ndarray, np.ndarray]] = None  # (min, max) in VR delta space


def _load_frame_cal() -> None:
    global _VR_TO_ARM, _VR_BOUNDS
    if CAL_FILE.exists():
        data = json.loads(CAL_FILE.read_text())
        _VR_TO_ARM = np.array(data["matrix"], dtype=float)
        if "vr_bounds" in data:
            _VR_BOUNDS = (
                np.array(data["vr_bounds"]["min"], dtype=float),
                np.array(data["vr_bounds"]["max"], dtype=float),
            )
            print(f"[ik_control] Workspace bounds loaded — "
                  f"min={np.round(_VR_BOUNDS[0], 3)}  max={np.round(_VR_BOUNDS[1], 3)}")
        print(f"[ik_control] Frame calibration loaded from {CAL_FILE}")
    else:
        _VR_TO_ARM = DEFAULT_VR_TO_ARM.copy()
        _VR_BOUNDS = None
        print("[ik_control] No calibration file — using default frame mapping.")
        print(f"             Run with --calibrate to create one.\n")


def _save_frame_cal(matrix: np.ndarray,
                    vr_min: list[float] | None = None,
                    vr_max: list[float] | None = None) -> None:
    CAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"matrix": matrix.tolist()}
    if vr_min is not None and vr_max is not None:
        payload["vr_bounds"] = {"min": vr_min, "max": vr_max}
    CAL_FILE.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n  Saved → {CAL_FILE}")


def vr_delta_to_arm(delta_vr: np.ndarray) -> np.ndarray:
    return _VR_TO_ARM @ delta_vr


def vr_quat_to_arm(quat_xyzw: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(_VR_TO_ARM)
    R_proper  = U @ Vt
    if np.linalg.det(R_proper) < 0:
        U[:, -1] *= -1
        R_proper  = U @ Vt
    scipy_R = ScipyR.from_matrix(R_proper)
    return (scipy_R * ScipyR.from_quat(quat_xyzw)).as_quat()


# ─────────────────────────────────────────────────────────────────────────────
# URDF sanitiser
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise_urdf(text: str) -> str:
    text = re.sub(r'<xacro:include\s[^>]*/>', '', text)
    text = text.replace(' xmlns:xacro="http://www.ros.org/wiki/xacro"', '')
    text = re.sub(r'<geometry>\s*<mesh\b[^>]*/>\s*</geometry>',
                  '<geometry><box size="0.01 0.01 0.01"/></geometry>',
                  text, flags=re.DOTALL)
    text = re.sub(r'<geometry>\s*<mesh\b[^>]*>[^<]*</mesh>\s*</geometry>',
                  '<geometry><box size="0.01 0.01 0.01"/></geometry>',
                  text, flags=re.DOTALL)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# PyBullet IK solver
# ─────────────────────────────────────────────────────────────────────────────

class SO101IKSolver:

    def __init__(self, urdf_path: Path):
        raw   = urdf_path.read_text(encoding="utf-8")
        clean = _sanitise_urdf(raw)

        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".urdf", mode="w", encoding="utf-8", delete=False
        )
        self._tmp.write(clean)
        self._tmp.flush()

        self._client = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                   physicsClientId=self._client)
        self._robot = pb.loadURDF(
            self._tmp.name, basePosition=[0, 0, 0],
            useFixedBase=True, physicsClientId=self._client,
        )

        n = pb.getNumJoints(self._robot, physicsClientId=self._client)
        self._joint_idx: dict[str, int] = {}
        self._link_idx:  dict[str, int] = {}
        for i in range(n):
            info = pb.getJointInfo(self._robot, i, physicsClientId=self._client)
            self._joint_idx[info[1].decode()] = i
            self._link_idx[info[12].decode()] = i

        if EE_LINK not in self._link_idx:
            raise RuntimeError(f"EE link '{EE_LINK}' not found in URDF.")
        self._ee_idx = self._link_idx[EE_LINK]
        self._ik_pb_indices = [self._joint_idx[n] for n in IK_JOINT_NAMES]

        self.ee_home: np.ndarray = self._fk_ee()

    def solve(
        self,
        target_pos: np.ndarray,
        target_orn: Optional[list[float]] = None,
        max_iters: int = 300,
        residual: float = 1e-5,
    ) -> dict[str, float]:
        kwargs: dict = dict(
            bodyUniqueId=self._robot, endEffectorLinkIndex=self._ee_idx,
            targetPosition=list(target_pos),
            lowerLimits=_ALL_LOWER, upperLimits=_ALL_UPPER,
            jointRanges=_ALL_RANGES, restPoses=_ALL_REST,
            maxNumIterations=max_iters, residualThreshold=residual,
            physicsClientId=self._client,
        )
        if target_orn is not None:
            kwargs["targetOrientation"] = list(target_orn)

        raw = pb.calculateInverseKinematics(**kwargs)

        result: dict[str, float] = {}
        for i, name in enumerate(IK_JOINT_NAMES):
            rad = float(raw[i])
            rad = max(JOINT_LOWER[i], min(JOINT_UPPER[i], rad))
            result[name] = math.degrees(rad)

        for pb_idx, rad in zip(self._ik_pb_indices, raw[:len(IK_JOINT_NAMES)]):
            pb.resetJointState(self._robot, pb_idx, rad,
                               physicsClientId=self._client)
        return result

    def _fk_ee(self) -> np.ndarray:
        state = pb.getLinkState(self._robot, self._ee_idx,
                                physicsClientId=self._client)
        return np.array(state[4])

    def close(self) -> None:
        pb.disconnect(self._client)
        Path(self._tmp.name).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight pose reader  (used during calibration)
# ─────────────────────────────────────────────────────────────────────────────

class _PoseReader:
    """Subscribes to the VR bus and caches the latest wrist position."""

    def __init__(self, side: str) -> None:
        self._side  = side
        self._topic = f"{side}_wrist"
        self._pos: Optional[np.ndarray] = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()

        t = threading.Thread(target=self._run, daemon=True, name="pose-reader")
        t.start()

    def _run(self) -> None:
        sub = Subscriber(VR_DATA_PORT)
        for msg in sub.iter_messages():
            if self._stop.is_set():
                break
            if msg.get("topic") == self._topic:
                d = msg.get("data", {})
                with self._lock:
                    self._pos = np.array([d["x"], d["y"], d["z"]])

    def get_pos(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._pos is None else self._pos.copy()

    def destroy(self) -> None:
        self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# Interactive calibration
# ─────────────────────────────────────────────────────────────────────────────

_AXIS_LABELS = ["X", "Y", "Z"]


def _enter_pressed() -> bool:
    return bool(select.select([sys.stdin], [], [], 0)[0]) and sys.stdin.readline()


def _wait_enter(reader: _PoseReader) -> np.ndarray:
    print("     ↵  Press Enter when ready …", end="", flush=True)
    while not _enter_pressed():
        time.sleep(0.05)
    pos = reader.get_pos()
    print()
    return pos


def _capture_delta(reader: _PoseReader, home: np.ndarray,
                   prompt: str, min_dist: float = 0.02) -> np.ndarray:
    print(f"\n  {prompt}")
    while True:
        pos   = _wait_enter(reader)
        delta = pos - home
        dist  = np.linalg.norm(delta)
        if dist < min_dist:
            print(f"  ⚠  Movement too small ({dist*100:.1f} cm). "
                  f"Move at least {min_dist*100:.0f} cm and try again.")
            continue
        dom = int(np.argmax(np.abs(delta)))
        print(f"     VR delta = [{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}]  "
              f"dominant axis = VR_{_AXIS_LABELS[dom]}  "
              f"({'+' if delta[dom] > 0 else '-'})")
        return delta


def _dominant_axis(d_pos: np.ndarray, d_neg: np.ndarray,
                   label: str) -> tuple[int, float]:
    """Return (dominant VR axis index, sign) using both directions of a movement pair.

    Averaging |d_pos| and |d_neg| makes axis detection robust against drift and
    asymmetric movement distances.  A consistency warning fires if the two
    measurements don't agree on the dominant axis.
    """
    avg  = (np.abs(d_pos) + np.abs(d_neg)) / 2.0
    dom  = int(np.argmax(avg))
    sign = float(np.sign(d_pos[dom]))  # sign taken from the positive direction

    dom_neg = int(np.argmax(np.abs(d_neg)))
    if dom_neg != dom:
        print(f"  ⚠  {label}: positive/negative measurements disagree on dominant "
              f"axis (pos→VR_{_AXIS_LABELS[dom]}, neg→VR_{_AXIS_LABELS[dom_neg]}). "
              f"Try again with more deliberate movements.")
    return dom, sign


def run_calibration(side: str = "left") -> None:
    print("\n" + "═" * 60)
    print("  VR → ARM FRAME CALIBRATION  (full bounding box)")
    print("═" * 60)
    print(f"""
  We will calibrate which VR axes map to which arm directions
  and measure the full workspace bounding box.
  Using the {side.upper()} wrist (bus topic: {side}_wrist).

  Six movements in 3 paired steps — move at least ~5 cm each
  time and keep the other axes as still as possible.
  Return to neutral between steps, then press Enter.
""")

    reader = _PoseReader(side)

    print("  Waiting for VR pose data …")
    while reader.get_pos() is None:
        time.sleep(0.05)
    print("  Got pose data.\n")

    print("  Step 0 — Hold your wrist STILL at a comfortable neutral position.")
    home = _wait_enter(reader)
    print(f"     Home latched: VR pos = [{home[0]:+.4f}, {home[1]:+.4f}, {home[2]:+.4f}]")

    # ── Pair 1: depth axis ───────────────────────────────────────────────────
    delta_fwd = _capture_delta(
        reader, home,
        "Step 1a — Push your wrist FORWARD toward the arm (arm extends out).\n"
        "          Then press Enter.",
        min_dist=0.03,
    )
    delta_bwd = _capture_delta(
        reader, home,
        "Step 1b — Pull your wrist BACKWARD away from the arm (arm retracts).\n"
        "          Then press Enter.",
        min_dist=0.03,
    )
    dom_depth, sign_vr_depth = _dominant_axis(delta_fwd, delta_bwd, "Depth")
    sign_depth = -sign_vr_depth  # forward VR → arm extends (positive arm Y)

    # ── Pair 2: lateral axis ─────────────────────────────────────────────────
    delta_right = _capture_delta(
        reader, home,
        "Step 2a — Move your wrist to the RIGHT (arm EE moves right).\n"
        "          Then press Enter.",
        min_dist=0.03,
    )
    delta_left = _capture_delta(
        reader, home,
        "Step 2b — Move your wrist to the LEFT (arm EE moves left).\n"
        "          Then press Enter.",
        min_dist=0.03,
    )
    dom_horiz, sign_horiz = _dominant_axis(delta_right, delta_left, "Lateral")

    # ── Pair 3: vertical axis ────────────────────────────────────────────────
    delta_up = _capture_delta(
        reader, home,
        "Step 3a — Raise your wrist UP (arm EE moves up).\n"
        "          Then press Enter.",
        min_dist=0.03,
    )
    delta_down = _capture_delta(
        reader, home,
        "Step 3b — Lower your wrist DOWN (arm EE moves down).\n"
        "          Then press Enter.",
        min_dist=0.03,
    )
    dom_vert, sign_vert = _dominant_axis(delta_up, delta_down, "Vertical")

    # ── Build rotation matrix ─────────────────────────────────────────────────
    M = np.zeros((3, 3), dtype=float)
    M[0, dom_horiz] = float(sign_horiz)
    M[1, dom_depth] = float(sign_depth)
    M[2, dom_vert]  = float(sign_vert)

    print("\n  Frame mapping:")
    print(f"    arm_X  ←  {'+' if sign_horiz > 0 else '-'}VR_{_AXIS_LABELS[dom_horiz]}  (lateral)")
    print(f"    arm_Y  ←  {'+' if sign_depth > 0 else '-'}VR_{_AXIS_LABELS[dom_depth]}  (depth)")
    print(f"    arm_Z  ←  {'+' if sign_vert  > 0 else '-'}VR_{_AXIS_LABELS[dom_vert]}   (vertical)")
    print(f"\n  Matrix:\n{M}")

    dom_axes = [dom_horiz, dom_depth, dom_vert]
    if len(set(dom_axes)) < 3:
        print("\n  ⚠ WARNING: two arm axes share the same VR axis — "
              "calibration may be poor.  Movements were not distinct enough.")

    # ── Bounding box in VR delta space ────────────────────────────────────────
    all_deltas = np.stack([
        delta_fwd, delta_bwd,
        delta_right, delta_left,
        delta_up, delta_down,
    ])
    vr_min = all_deltas.min(axis=0).tolist()
    vr_max = all_deltas.max(axis=0).tolist()

    print("\n  Workspace bounds (VR delta from neutral):")
    for i, ax in enumerate(_AXIS_LABELS):
        print(f"    VR_{ax}  [{vr_min[i]:+.4f}, {vr_max[i]:+.4f}] m")

    _save_frame_cal(M, vr_min, vr_max)

    reader.destroy()

    print("\n" + "═" * 60)
    print("  Calibration complete!  Run without --calibrate to start control.")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-arm controller
# ─────────────────────────────────────────────────────────────────────────────

class _ArmController:

    def __init__(self, side: str, urdf_path: Path,
                 scale: float, max_delta: float, with_orientation: bool,
                 vr_bounds: Optional[tuple[np.ndarray, np.ndarray]]):
        self.side        = side
        self._scale      = scale
        self._max_delta  = max_delta
        self._with_orn   = with_orientation
        self._vr_bounds  = vr_bounds

        self._ik             = SO101IKSolver(urdf_path)
        self._ref_pos: Optional[np.ndarray] = None
        self._last_target    = self._ik.ee_home.copy()

        port = LEFT_ARM_CMD_PORT if side == "left" else RIGHT_ARM_CMD_PORT
        self._cmd = CommandClient(port)

        print(f"[ik_control] [{side}] IK ready — EE home: {np.round(self._ik.ee_home, 4)} m")

    def on_pose(self, data: dict) -> None:
        vpos  = np.array([data["x"], data["y"], data["z"]])
        vquat = np.array([data["qx"], data["qy"], data["qz"], data["qw"]])

        if self._ref_pos is None:
            self._ref_pos = vpos.copy()
            print(f"[ik_control] [{self.side}] Reference latched: {np.round(vpos, 4)}")

        delta_vr = vpos - self._ref_pos

        # Clamp to calibrated bounding box so the arm never exceeds the
        # measured workspace limits.
        if self._vr_bounds is not None:
            delta_vr = np.clip(delta_vr, self._vr_bounds[0], self._vr_bounds[1])

        delta_arm  = vr_delta_to_arm(delta_vr * self._scale)
        raw_target = self._ik.ee_home + delta_arm

        step = raw_target - self._last_target
        norm = float(np.linalg.norm(step))
        if norm > self._max_delta:
            raw_target = self._last_target + step * (self._max_delta / norm)
        self._last_target = raw_target

        orn = vr_quat_to_arm(vquat).tolist() if self._with_orn else None

        try:
            joints = self._ik.solve(raw_target, target_orn=orn)
        except Exception as exc:
            print(f"[ik_control] [{self.side}] IK failed: {exc}", file=sys.stderr)
            return

        self._cmd.send({
            "topic": "set_positions",
            "data":  {k: round(v, 3) for k, v in joints.items()},
        })

    def close(self) -> None:
        self._ik.close()
        self._cmd.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main IK control class (no ROS2)
# ─────────────────────────────────────────────────────────────────────────────

class IKControl:

    def __init__(self, args: argparse.Namespace) -> None:
        sides  = ["left", "right"] if args.side == "both" else [args.side]
        urdf   = Path(args.urdf)
        self._running = True

        self._arms: dict[str, _ArmController] = {
            side: _ArmController(
                side=side, urdf_path=urdf,
                scale=args.scale, max_delta=args.max_delta,
                with_orientation=args.with_orientation,
                vr_bounds=_VR_BOUNDS,
            )
            for side in sides
        }

        self._sides = set(sides)
        t = threading.Thread(target=self._vr_loop, daemon=True, name="ik-vr-sub")
        t.start()

        print(
            f"[ik_control] sides={sides}  scale={args.scale}"
            f"  max_delta={args.max_delta} m"
            f"  orientation={'ON' if args.with_orientation else 'OFF'}"
        )
        print(
            "[ik_control] Hold your wrist(s) at a neutral pose — "
            "reference latches on the first message."
        )

    def _vr_loop(self) -> None:
        sub = Subscriber(VR_DATA_PORT)
        for msg in sub.iter_messages():
            if not self._running:
                break
            topic = msg.get("topic", "")
            data  = msg.get("data", {})
            if topic == "left_wrist" and "left" in self._sides:
                self._arms["left"].on_pose(data)
            elif topic == "right_wrist" and "right" in self._sides:
                self._arms["right"].on_pose(data)

    def close(self) -> None:
        self._running = False
        for arm in self._arms.values():
            arm.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ik_control",
        description="VR wrist pose → SO-101 full-arm IK → teleoperation (no ROS2)",
    )
    parser.add_argument("--calibrate",        action="store_true",
                        help="Run interactive frame calibration and exit")
    parser.add_argument("--cal-side",         choices=["left", "right"], default="left",
                        help="Which wrist to use for calibration (default: left)")
    parser.add_argument("--side",             choices=["left", "right", "both"], default="both")
    parser.add_argument("--scale",            type=float, default=1.0,
                        help="Workspace scale (default: 1.0)")
    parser.add_argument("--max-delta",        type=float, default=DEFAULT_MAX_DELTA_M,
                        help=f"Max Cartesian step per callback in metres "
                             f"(default: {DEFAULT_MAX_DELTA_M})")
    parser.add_argument("--with-orientation", action="store_true",
                        help="Include VR wrist orientation as IK target")
    parser.add_argument("--urdf",             default=str(URDF_PATH))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.calibrate:
        run_calibration(side=args.cal_side)
        return

    _load_frame_cal()

    if not Path(args.urdf).exists():
        print(f"[ik_control] ERROR — URDF not found: {args.urdf}")
        sys.exit(1)

    print(f"[ik_control] Frame mapping:\n{_VR_TO_ARM}\n")

    ik: Optional[IKControl] = None
    try:
        ik = IKControl(args)
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[ik_control] Stopped.")
    except Exception:
        traceback.print_exc()
    finally:
        if ik is not None:
            ik.close()


if __name__ == "__main__":
    main()
