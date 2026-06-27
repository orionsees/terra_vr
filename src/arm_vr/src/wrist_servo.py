#!/usr/bin/env python3
"""
wrist_servo.py — Calibrate then control SO-101 wrist servos from VR wrist data.

  Roll  → servo ID 4  (wrist_flex)   on both arms
  Yaw   → servo ID 5  (wrist_roll)   on both arms

Calibration stored in:
    config/left_wrist.json
    config/right_wrist.json

Subscribes:
    /left_wrist   /right_wrist   geometry_msgs/PoseStamped

Publishes:
    /left_arm/cmd/set_positions_raw    std_msgs/String  {"4": <tick>, "5": <tick>}
    /right_arm/cmd/set_positions_raw   std_msgs/String  {"4": <tick>, "5": <tick>}

Usage:
    python3 wrist_servo.py
    python3 wrist_servo.py --recalibrate
"""

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


# ── paths ─────────────────────────────────────────────────────────────────────
_CONFIG_DIR     = (Path(__file__).parent / ".." / "config").resolve()
CAL_LEFT        = _CONFIG_DIR / "left_wrist.json"
CAL_RIGHT       = _CONFIG_DIR / "right_wrist.json"
SERVO_CAL_LEFT  = _CONFIG_DIR / "left_arm.json"
SERVO_CAL_RIGHT = _CONFIG_DIR / "right_arm.json"

# ── control constants ─────────────────────────────────────────────────────────
SPIKE_THRESHOLD  = 20.0   # degrees — reject jumps larger than this
DEADBAND         = 15     # ticks   — skip command if change is smaller than this
ACTIVITY_WINDOW  = 30     # samples — rolling window for activity (std dev) comparison


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_roll(qx, qy, qz, qw) -> float:
    return math.degrees(
        math.atan2(2.0 * (qw * qx + qy * qz),
                   1.0 - 2.0 * (qx * qx + qy * qy))
    )


def quat_to_yaw(qx, qy, qz, qw) -> float:
    return math.degrees(
        math.atan2(2.0 * (qw * qz + qx * qy),
                   1.0 - 2.0 * (qy * qy + qz * qz))
    )


def angle_to_tick(angle: float, a_min: float, a_max: float,
                  t_min: int, t_max: int) -> int:
    frac = max(0.0, min(1.0, (angle - a_min) / (a_max - a_min)))
    return int(t_min + frac * (t_max - t_min))


# ─────────────────────────────────────────────────────────────────────────────
# File helpers  (merge-write so scripts never clobber each other's keys)
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, updates: dict) -> None:
    data = _load(path)
    data.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'═' * 58}")
    print(f"  {title}")
    print(f"{'═' * 58}")


def _cursor_up(n: int = 1) -> None:
    print(f"\033[{n}A", end="", flush=True)


def _enter_pressed() -> bool:
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Reader node  (calibration phase — no publishers needed)
# ─────────────────────────────────────────────────────────────────────────────

class WristReader(Node):

    def __init__(self):
        super().__init__("wrist_reader")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._lock = threading.Lock()
        self._data = {"left": None, "right": None}   # {roll, yaw} per side

        self.create_subscription(PoseStamped, "/left_wrist",
                                 lambda m: self._cb(m, "left"),  qos)
        self.create_subscription(PoseStamped, "/right_wrist",
                                 lambda m: self._cb(m, "right"), qos)

    def _cb(self, msg: PoseStamped, side: str):
        q = msg.pose.orientation
        with self._lock:
            self._data[side] = {
                "roll": quat_to_roll(q.x, q.y, q.z, q.w),
                "yaw":  quat_to_yaw(q.x, q.y, q.z, q.w),
            }

    def get(self, side: str):
        with self._lock:
            return self._data[side]


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def _capture(reader: WristReader, side: str, angle_key: str, label: str) -> float:
    """Live-display one angle and return the value when the user presses Enter."""
    print(f"\n  Waiting for {side.upper()} wrist data…")
    while reader.get(side) is None:
        time.sleep(0.05)

    first = True
    while True:
        d = reader.get(side)
        val = d[angle_key]
        if not first:
            _cursor_up(1)
        first = False
        print(f"  {label:<42}  {angle_key} = {val:+7.1f}°   [Press Enter to capture]",
              flush=True)
        if _enter_pressed():
            val = reader.get(side)[angle_key]
            print(f"\n  ✓  Captured: {val:+.1f}°")
            return val
        time.sleep(0.08)


def run_calibration(reader: WristReader) -> None:
    _banner("WRIST CALIBRATION  —  Roll (servo 4)  +  Yaw (servo 5)")
    print("""
  For each wrist we'll capture four positions:
    Roll MIN / MAX  — tilt sideways (palm-up ↔ palm-down)
    Yaw  MIN / MAX  — rotate like a doorknob (left ↔ right)
""")

    for side, cal_file in [("left", CAL_LEFT), ("right", CAL_RIGHT)]:
        _banner(f"{side.upper()} WRIST")

        # ── roll ──────────────────────────────────────────────────────────────
        print(f"\n  ── ROLL  (servo 4 — wrist_flex) ──")

        print(f"\n  Step 1 — Tilt {side.upper()} wrist to MINIMUM roll")
        print("           (as far as you comfortably tilt in one direction).")
        roll_min = _capture(reader, side, "roll", f"{side.upper()} MIN roll")

        print(f"\n  Step 2 — Tilt {side.upper()} wrist to MAXIMUM roll")
        print("           (the opposite direction).")
        roll_max = _capture(reader, side, "roll", f"{side.upper()} MAX roll")

        if roll_min > roll_max:
            roll_min, roll_max = roll_max, roll_min

        # ── yaw ───────────────────────────────────────────────────────────────
        print(f"\n  ── YAW  (servo 5 — wrist_roll) ──")

        print(f"\n  Step 3 — Rotate {side.upper()} wrist to MINIMUM yaw")
        print("           (as far as you comfortably rotate in one direction).")
        yaw_min = _capture(reader, side, "yaw", f"{side.upper()} MIN yaw")

        print(f"\n  Step 4 — Rotate {side.upper()} wrist to MAXIMUM yaw")
        print("           (the opposite direction).")
        yaw_max = _capture(reader, side, "yaw", f"{side.upper()} MAX yaw")

        if yaw_min > yaw_max:
            yaw_min, yaw_max = yaw_max, yaw_min

        # ── save ──────────────────────────────────────────────────────────────
        _save(cal_file, {
            "roll_min": round(roll_min, 2), "roll_max": round(roll_max, 2),
            "yaw_min":  round(yaw_min,  2), "yaw_max":  round(yaw_max,  2),
        })

        print(f"\n  Saved → {cal_file}")
        print(f"  roll  {roll_min:+.2f}° → {roll_max:+.2f}°   "
              f"span {roll_max - roll_min:.1f}°")
        print(f"  yaw   {yaw_min:+.2f}° → {yaw_max:+.2f}°   "
              f"span {yaw_max - yaw_min:.1f}°")


def _cal_complete() -> bool:
    required = ("roll_min", "roll_max", "yaw_min", "yaw_max")
    for f in (CAL_LEFT, CAL_RIGHT):
        d = _load(f)
        if any(k not in d for k in required):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Control node
# ─────────────────────────────────────────────────────────────────────────────

class WristServoNode(Node):

    def __init__(self):
        super().__init__("wrist_servo")

        left_cal  = _load(CAL_LEFT)
        right_cal = _load(CAL_RIGHT)
        srv_left  = json.loads(SERVO_CAL_LEFT.read_text())
        srv_right = json.loads(SERVO_CAL_RIGHT.read_text())

        # angle ranges from wrist calibration
        self._roll_range = {
            "left":  (left_cal["roll_min"],  left_cal["roll_max"]),
            "right": (right_cal["roll_min"], right_cal["roll_max"]),
        }
        self._yaw_range = {
            "left":  (left_cal["yaw_min"],  left_cal["yaw_max"]),
            "right": (right_cal["yaw_min"], right_cal["yaw_max"]),
        }

        # tick ranges from servo calibration
        self._tick4 = {
            "left":  (srv_left["wrist_flex"]["range_min"],
                      srv_left["wrist_flex"]["range_max"]),
            "right": (srv_right["wrist_flex"]["range_min"],
                      srv_right["wrist_flex"]["range_max"]),
        }
        self._tick5 = {
            "left":  (srv_left["wrist_roll"]["range_min"],
                      srv_left["wrist_roll"]["range_max"]),
            "right": (srv_right["wrist_roll"]["range_min"],
                      srv_right["wrist_roll"]["range_max"]),
        }

        self._prev  = {"left": {"roll": None, "yaw": None},
                       "right": {"roll": None, "yaw": None}}
        self._sent  = {"left": {4: None, 5: None},
                       "right": {4: None, 5: None}}

        # rolling buffers for activity detection (std dev over last ACTIVITY_WINDOW samples)
        self._roll_buf  = {"left": deque(maxlen=ACTIVITY_WINDOW),
                           "right": deque(maxlen=ACTIVITY_WINDOW)}
        self._yaw_buf   = {"left": deque(maxlen=ACTIVITY_WINDOW),
                           "right": deque(maxlen=ACTIVITY_WINDOW)}
        self._dominant  = {"left": None, "right": None}  # tracks last logged dominant

        qos     = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        rel_qos = QoSProfile(depth=10)

        self.create_subscription(PoseStamped, "/left_wrist",
                                 lambda m: self._cb(m, "left"),  qos)
        self.create_subscription(PoseStamped, "/right_wrist",
                                 lambda m: self._cb(m, "right"), qos)

        self._pub = {
            "left":  self.create_publisher(String, "/left_arm/cmd/set_positions_raw",  rel_qos),
            "right": self.create_publisher(String, "/right_arm/cmd/set_positions_raw", rel_qos),
        }

        self.get_logger().info("wrist_servo control active")
        for side in ("left", "right"):
            self.get_logger().info(
                f"  [{side.upper()}]"
                f"  roll {self._roll_range[side]}  →  tick4 {self._tick4[side]}"
                f"  |  yaw {self._yaw_range[side]}  →  tick5 {self._tick5[side]}"
            )

    def _cb(self, msg: PoseStamped, side: str) -> None:
        q    = msg.pose.orientation
        roll = quat_to_roll(q.x, q.y, q.z, q.w)
        yaw  = quat_to_yaw(q.x, q.y, q.z, q.w)

        # Accumulate rolling activity buffers
        self._roll_buf[side].append(roll)
        self._yaw_buf[side].append(yaw)

        # Activity = std dev over recent window (needs ≥3 samples)
        roll_act = (float(np.std(self._roll_buf[side]))
                    if len(self._roll_buf[side]) >= 3 else 0.0)
        yaw_act  = (float(np.std(self._yaw_buf[side]))
                    if len(self._yaw_buf[side])  >= 3 else 0.0)

        dominant = "roll" if roll_act >= yaw_act else "yaw"

        # Log on dominance switch
        if dominant != self._dominant[side]:
            servo = 4 if dominant == "roll" else 5
            self.get_logger().info(
                f"[{side.upper()}] dominant → {dominant.upper()}"
                f"  (σ roll={roll_act:.1f}°  yaw={yaw_act:.1f}°)"
                f"  → servo {servo}"
            )
            self._dominant[side] = dominant

        prev = self._prev[side]
        cmd  = {}

        if dominant == "roll":
            # ── servo 4 (wrist_flex) ──────────────────────────────────────────
            if prev["roll"] is None or abs(roll - prev["roll"]) <= SPIKE_THRESHOLD:
                prev["roll"] = roll
                t4 = angle_to_tick(roll, *self._roll_range[side], *self._tick4[side])
                if self._sent[side][4] is None or abs(t4 - self._sent[side][4]) >= DEADBAND:
                    self._sent[side][4] = t4
                    cmd[4] = t4
            else:
                self.get_logger().warn(
                    f"[{side.upper()}] roll spike rejected  {prev['roll']:+.1f}° → {roll:+.1f}°"
                )
        else:
            # ── servo 5 (wrist_roll) ──────────────────────────────────────────
            if prev["yaw"] is None or abs(yaw - prev["yaw"]) <= SPIKE_THRESHOLD:
                prev["yaw"] = yaw
                t5 = angle_to_tick(yaw, *self._yaw_range[side], *self._tick5[side])
                if self._sent[side][5] is None or abs(t5 - self._sent[side][5]) >= DEADBAND:
                    self._sent[side][5] = t5
                    cmd[5] = t5
            else:
                self.get_logger().warn(
                    f"[{side.upper()}] yaw spike rejected  {prev['yaw']:+.1f}° → {yaw:+.1f}°"
                )

        if not cmd:
            return

        out = String()
        out.data = json.dumps({str(k): v for k, v in cmd.items()})
        self._pub[side].publish(out)

        for k, v in cmd.items():
            angle_str = f"roll={roll:+.1f}°" if k == 4 else f"yaw={yaw:+.1f}°"
            self.get_logger().info(
                f"[{side.upper()}]  servo{k}={v}  ({angle_str})"
                f"  [σ roll={roll_act:.1f}°  yaw={yaw_act:.1f}°]"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wrist_servo",
        description="VR wrist roll/yaw → SO-101 servo 4 (wrist_flex) + servo 5 (wrist_roll)",
    )
    parser.add_argument("--recalibrate", action="store_true",
                        help="Re-run calibration even if saved data exists")
    args = parser.parse_args()

    needs_cal = args.recalibrate or not _cal_complete()

    rclpy.init()

    # ── phase 1: calibration ──────────────────────────────────────────────────
    if needs_cal:
        reader = WristReader()
        stop   = threading.Event()

        def _spin():
            while not stop.is_set():
                rclpy.spin_once(reader, timeout_sec=0.02)

        spin_thread = threading.Thread(target=_spin, daemon=True)
        spin_thread.start()

        try:
            run_calibration(reader)
            _banner("CALIBRATION COMPLETE")
            print(f"  {CAL_LEFT}")
            print(f"  {CAL_RIGHT}\n")
        except KeyboardInterrupt:
            print("\n[wrist_servo] Calibration cancelled.")
            stop.set()
            reader.destroy_node()
            rclpy.shutdown()
            sys.exit(0)

        stop.set()
        spin_thread.join(timeout=2.0)
        reader.destroy_node()

    else:
        _banner("CALIBRATION LOADED")
        for side, f in [("left", CAL_LEFT), ("right", CAL_RIGHT)]:
            d = _load(f)
            print(f"  {side.upper():<6}  "
                  f"roll [{d['roll_min']:+.1f}°, {d['roll_max']:+.1f}°]   "
                  f"yaw  [{d['yaw_min']:+.1f}°, {d['yaw_max']:+.1f}°]")
        print("  (run with --recalibrate to redo)\n")

    # ── phase 2: control ──────────────────────────────────────────────────────
    node = WristServoNode()
    print("[wrist_servo] Running — tilt/rotate your wrists to control servos 4 & 5.")
    print("              Press Ctrl-C to stop.\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[wrist_servo] Stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
