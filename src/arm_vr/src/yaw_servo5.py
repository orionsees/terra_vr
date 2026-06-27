#!/usr/bin/env python3
"""
yaw_servo5.py — Calibrate wrist yaw range then control SO-101 wrist_roll
                (servo ID 5) on both arms from VR wrist yaw data.

Calibration is stored in:
    config/left_wrist.json    (shared with roll_servo4.py)
    config/right_wrist.json   (shared with roll_servo4.py)

Only the yaw_min / yaw_max keys are written — existing keys (e.g. roll_min,
roll_max written by roll_servo4.py) are preserved.

Subscribes:
    /left_wrist   /right_wrist   geometry_msgs/PoseStamped

Publishes:
    /left_arm/cmd/set_positions_raw    std_msgs/String  {"5": <tick>}
    /right_arm/cmd/set_positions_raw   std_msgs/String  {"5": <tick>}

Usage:
    python3 yaw_servo5.py
    python3 yaw_servo5.py --recalibrate
"""

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import threading
import time
from pathlib import Path

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

# ── servo control constants ───────────────────────────────────────────────────
SPIKE_THRESHOLD = 20.0   # degrees
DEADBAND        = 15     # ticks


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_yaw(qx, qy, qz, qw) -> float:
    return math.degrees(
        math.atan2(2.0 * (qw * qz + qx * qy),
                   1.0 - 2.0 * (qy * qy + qz * qz))
    )


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


def _load_wrist_file(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_wrist_file(path: Path, updates: dict) -> None:
    data = _load_wrist_file(path)
    data.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight reader node (calibration phase only)
# ─────────────────────────────────────────────────────────────────────────────

class YawReader(Node):

    def __init__(self):
        super().__init__("yaw_reader")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._lock = threading.Lock()
        self._yaw  = {"left": None, "right": None}
        self.create_subscription(PoseStamped, "/left_wrist",
                                 lambda m: self._cb(m, "left"),  qos)
        self.create_subscription(PoseStamped, "/right_wrist",
                                 lambda m: self._cb(m, "right"), qos)

    def _cb(self, msg: PoseStamped, side: str):
        q = msg.pose.orientation
        with self._lock:
            self._yaw[side] = quat_to_yaw(q.x, q.y, q.z, q.w)

    def get(self, side: str):
        with self._lock:
            return self._yaw[side]


# ─────────────────────────────────────────────────────────────────────────────
# Calibration routine
# ─────────────────────────────────────────────────────────────────────────────

def _capture_yaw(reader: YawReader, side: str, label: str) -> float:
    print(f"\n  Waiting for {side.upper()} wrist data…")
    while reader.get(side) is None:
        time.sleep(0.05)

    first_print = True
    while True:
        val = reader.get(side)
        if not first_print:
            _cursor_up(1)
        first_print = False
        print(f"  {label:<38}  yaw = {val:+7.1f}°   [Press Enter to capture]",
              flush=True)
        if _enter_pressed():
            val = reader.get(side)
            print(f"\n  ✓  Captured: {val:+.1f}°")
            return val
        time.sleep(0.08)


def run_calibration(reader: YawReader) -> None:
    _banner("WRIST YAW CALIBRATION")
    print("""
  We'll record the MIN and MAX yaw angle for each wrist.
  Rotate your wrist left/right (like turning a doorknob).
  Move to the requested position, watch the live reading
  stabilise, then press Enter to lock it in.
""")

    for side, cal_file in [("left", CAL_LEFT), ("right", CAL_RIGHT)]:
        _banner(f"{side.upper()} WRIST")

        print(f"\n  Step 1 — Rotate your {side.upper()} wrist to the MINIMUM yaw")
        print("           (as far as you comfortably rotate in one direction).")
        yaw_min = _capture_yaw(reader, side, f"{side.upper()} wrist MIN yaw")

        print(f"\n  Step 2 — Now rotate your {side.upper()} wrist to the MAXIMUM yaw")
        print("           (the opposite direction, full comfortable range).")
        yaw_max = _capture_yaw(reader, side, f"{side.upper()} wrist MAX yaw")

        if yaw_min > yaw_max:
            yaw_min, yaw_max = yaw_max, yaw_min
            print("  (min/max swapped so that min < max)")

        _save_wrist_file(cal_file, {
            "yaw_min": round(yaw_min, 2),
            "yaw_max": round(yaw_max, 2),
        })

        print(f"\n  Saved → {cal_file}")
        print(f"  {side.upper():<6}  yaw_min = {yaw_min:+.2f}°   "
              f"yaw_max = {yaw_max:+.2f}°   "
              f"span = {yaw_max - yaw_min:.2f}°")


# ─────────────────────────────────────────────────────────────────────────────
# Control node
# ─────────────────────────────────────────────────────────────────────────────

class YawServo5Node(Node):

    def __init__(self):
        super().__init__("yaw_servo5")

        left_cal  = _load_wrist_file(CAL_LEFT)
        right_cal = _load_wrist_file(CAL_RIGHT)
        srv_left  = json.loads(SERVO_CAL_LEFT.read_text())
        srv_right = json.loads(SERVO_CAL_RIGHT.read_text())

        self._yaw_range = {
            "left":  (left_cal["yaw_min"],  left_cal["yaw_max"]),
            "right": (right_cal["yaw_min"], right_cal["yaw_max"]),
        }
        self._tick_range = {
            "left":  (srv_left["wrist_roll"]["range_min"],
                      srv_left["wrist_roll"]["range_max"]),
            "right": (srv_right["wrist_roll"]["range_min"],
                      srv_right["wrist_roll"]["range_max"]),
        }

        self._prev_yaw  = {"left": None, "right": None}
        self._sent_tick = {"left": None, "right": None}

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

        self.get_logger().info("yaw_servo5 control active")
        for side in ("left", "right"):
            ymin, ymax = self._yaw_range[side]
            tmin, tmax = self._tick_range[side]
            self.get_logger().info(
                f"  [{side.upper()}]  yaw [{ymin:+.1f}°, {ymax:+.1f}°]"
                f"  →  tick [{tmin}, {tmax}]"
            )

    def _cb(self, msg: PoseStamped, side: str) -> None:
        q   = msg.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        prev = self._prev_yaw[side]
        if prev is not None and abs(yaw - prev) > SPIKE_THRESHOLD:
            self.get_logger().warn(
                f"[{side.upper()}] spike rejected  {prev:+.1f}° → {yaw:+.1f}°"
            )
            return
        self._prev_yaw[side] = yaw

        y_min, y_max = self._yaw_range[side]
        t_min, t_max = self._tick_range[side]
        frac = max(0.0, min(1.0, (yaw - y_min) / (y_max - y_min)))
        tick = int(t_min + frac * (t_max - t_min))

        last = self._sent_tick[side]
        if last is not None and abs(tick - last) < DEADBAND:
            return
        self._sent_tick[side] = tick

        out = String()
        out.data = json.dumps({"5": tick})
        self._pub[side].publish(out)

        self.get_logger().info(
            f"[{side.upper()}]  yaw = {yaw:+6.1f}°  →  tick = {tick:4d}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _yaw_cal_exists() -> bool:
    for f in (CAL_LEFT, CAL_RIGHT):
        data = _load_wrist_file(f)
        if "yaw_min" not in data or "yaw_max" not in data:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="yaw_servo5",
        description="Map VR wrist yaw → SO-101 wrist_roll (servo 5), both arms",
    )
    parser.add_argument("--recalibrate", action="store_true",
                        help="Re-run the yaw calibration even if saved data exists")
    args = parser.parse_args()

    needs_cal = args.recalibrate or not _yaw_cal_exists()

    rclpy.init()

    if needs_cal:
        reader = YawReader()
        stop   = threading.Event()

        def _spin():
            while not stop.is_set():
                rclpy.spin_once(reader, timeout_sec=0.02)

        spin_thread = threading.Thread(target=_spin, daemon=True)
        spin_thread.start()

        try:
            run_calibration(reader)
            _banner("CALIBRATION COMPLETE")
            print(f"  left_wrist.json  → {CAL_LEFT}")
            print(f"  right_wrist.json → {CAL_RIGHT}\n")
        except KeyboardInterrupt:
            print("\n[yaw_servo5] Calibration cancelled.")
            stop.set()
            reader.destroy_node()
            rclpy.shutdown()
            sys.exit(0)

        stop.set()
        spin_thread.join(timeout=2.0)
        reader.destroy_node()

    else:
        _banner("YAW CALIBRATION LOADED")
        for side, f in [("left", CAL_LEFT), ("right", CAL_RIGHT)]:
            d = _load_wrist_file(f)
            print(f"  {side.upper():<6}  yaw_min = {d['yaw_min']:+.2f}°  "
                  f"yaw_max = {d['yaw_max']:+.2f}°")
        print("  (run with --recalibrate to redo)\n")

    node = YawServo5Node()
    print("[yaw_servo5] Running — rotate your wrists to control servo 5.")
    print("             Press Ctrl-C to stop.\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[yaw_servo5] Stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
