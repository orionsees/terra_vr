#!/usr/bin/env python3
"""
roll_servo4.py — Calibrate wrist roll range then control SO-101 wrist_flex
                 (servo ID 4) on both arms from VR wrist roll data.

Calibration is stored in:
    config/left_wrist.json    (shared with yaw_servo5.py)
    config/right_wrist.json   (shared with yaw_servo5.py)

Only the roll_min / roll_max keys are written — existing keys (e.g. yaw_min,
yaw_max written by yaw_servo5.py) are preserved.

Subscribes:
    /left_wrist   /right_wrist   geometry_msgs/PoseStamped

Publishes:
    /left_arm/cmd/set_positions_raw    std_msgs/String  {"4": <tick>}
    /right_arm/cmd/set_positions_raw   std_msgs/String  {"4": <tick>}

Usage:
    python3 roll_servo4.py
    python3 roll_servo4.py --recalibrate
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

def quat_to_roll(qx, qy, qz, qw) -> float:
    return math.degrees(
        math.atan2(2.0 * (qw * qx + qy * qz),
                   1.0 - 2.0 * (qx * qx + qy * qy))
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
    """Load existing wrist JSON or return empty dict if missing."""
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_wrist_file(path: Path, updates: dict) -> None:
    """Merge updates into the existing wrist JSON and write back."""
    data = _load_wrist_file(path)
    data.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight reader node (calibration phase only)
# ─────────────────────────────────────────────────────────────────────────────

class RollReader(Node):

    def __init__(self):
        super().__init__("roll_reader")
        qos     = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        rel_qos = QoSProfile(depth=10)
        self._lock = threading.Lock()
        self._roll = {"left": None, "right": None}
        self.create_subscription(PoseStamped, "/left_wrist",
                                 lambda m: self._cb(m, "left"),  qos)
        self.create_subscription(PoseStamped, "/right_wrist",
                                 lambda m: self._cb(m, "right"), qos)
        self._pub = {
            "left":  self.create_publisher(String, "/left_arm/cmd/set_positions_raw",  rel_qos),
            "right": self.create_publisher(String, "/right_arm/cmd/set_positions_raw", rel_qos),
        }

    def _cb(self, msg: PoseStamped, side: str):
        q = msg.pose.orientation
        with self._lock:
            self._roll[side] = quat_to_roll(q.x, q.y, q.z, q.w)

    def get(self, side: str):
        with self._lock:
            return self._roll[side]

    def send_tick(self, side: str, tick: int) -> None:
        """Drive servo 4 on one arm to a specific tick (calibration helper)."""
        msg = String()
        msg.data = json.dumps({"4": tick})
        self._pub[side].publish(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration routine
# ─────────────────────────────────────────────────────────────────────────────

def _capture_roll(reader: RollReader, side: str, label: str) -> float:
    print(f"\n  Waiting for {side.upper()} wrist data…")
    while reader.get(side) is None:
        time.sleep(0.05)

    first_print = True
    while True:
        val = reader.get(side)
        if not first_print:
            _cursor_up(1)
        first_print = False
        print(f"  {label:<38}  roll = {val:+7.1f}°   [Press Enter to capture]",
              flush=True)
        if _enter_pressed():
            val = reader.get(side)
            print(f"\n  ✓  Captured: {val:+.1f}°")
            return val
        time.sleep(0.08)


def run_calibration(reader: RollReader) -> None:
    _banner("WRIST ROLL CALIBRATION")

    srv_cal = {
        "left":  json.loads(SERVO_CAL_LEFT.read_text()),
        "right": json.loads(SERVO_CAL_RIGHT.read_text()),
    }
    tick_range = {
        side: (srv_cal[side]["wrist_flex"]["range_min"],
               srv_cal[side]["wrist_flex"]["range_max"])
        for side in ("left", "right")
    }

    print("""
  The physical arm will move servo 4 to each extreme position.
  Match your wrist roll to it, let the reading stabilise,
  then press Enter to lock in the mapping.
""")

    for side, cal_file in [("left", CAL_LEFT), ("right", CAL_RIGHT)]:
        _banner(f"{side.upper()} WRIST")
        t_min, t_max = tick_range[side]

        print(f"\n  Step 1 — Moving {side.upper()} arm servo 4 → MINIMUM ({t_min} ticks)")
        reader.send_tick(side, t_min)
        time.sleep(0.7)          # let the arm reach the position
        print(f"           Tilt your {side.upper()} wrist to MATCH the arm. "
              f"Watch the reading, then press Enter.")
        roll_min = _capture_roll(reader, side, f"{side.upper()} wrist MIN roll")

        print(f"\n  Step 2 — Moving {side.upper()} arm servo 4 → MAXIMUM ({t_max} ticks)")
        reader.send_tick(side, t_max)
        time.sleep(0.7)
        print(f"           Tilt your {side.upper()} wrist to MATCH the arm. "
              f"Watch the reading, then press Enter.")
        roll_max = _capture_roll(reader, side, f"{side.upper()} wrist MAX roll")

        if roll_min > roll_max:
            roll_min, roll_max = roll_max, roll_min
            print("  (min/max swapped so that min < max)")

        _save_wrist_file(cal_file, {
            "roll_min": round(roll_min, 2),
            "roll_max": round(roll_max, 2),
        })

        print(f"\n  Saved → {cal_file}")
        print(f"  {side.upper():<6}  roll_min = {roll_min:+.2f}°   "
              f"roll_max = {roll_max:+.2f}°   "
              f"span = {roll_max - roll_min:.2f}°")


# ─────────────────────────────────────────────────────────────────────────────
# Control node
# ─────────────────────────────────────────────────────────────────────────────

class RollServo4Node(Node):

    def __init__(self):
        super().__init__("roll_servo4")

        left_cal  = _load_wrist_file(CAL_LEFT)
        right_cal = _load_wrist_file(CAL_RIGHT)
        srv_left  = json.loads(SERVO_CAL_LEFT.read_text())
        srv_right = json.loads(SERVO_CAL_RIGHT.read_text())

        self._roll_range = {
            "left":  (left_cal["roll_min"],  left_cal["roll_max"]),
            "right": (right_cal["roll_min"], right_cal["roll_max"]),
        }
        self._tick_range = {
            "left":  (srv_left["wrist_flex"]["range_min"],
                      srv_left["wrist_flex"]["range_max"]),
            "right": (srv_right["wrist_flex"]["range_min"],
                      srv_right["wrist_flex"]["range_max"]),
        }

        self._prev_roll = {"left": None, "right": None}
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

        self.get_logger().info("roll_servo4 control active")
        for side in ("left", "right"):
            rmin, rmax = self._roll_range[side]
            tmin, tmax = self._tick_range[side]
            self.get_logger().info(
                f"  [{side.upper()}]  roll [{rmin:+.1f}°, {rmax:+.1f}°]"
                f"  →  tick [{tmin}, {tmax}]"
            )

    def _cb(self, msg: PoseStamped, side: str) -> None:
        q    = msg.pose.orientation
        roll = quat_to_roll(q.x, q.y, q.z, q.w)

        prev = self._prev_roll[side]
        if prev is not None and abs(roll - prev) > SPIKE_THRESHOLD:
            self.get_logger().warn(
                f"[{side.upper()}] spike rejected  {prev:+.1f}° → {roll:+.1f}°"
            )
            return
        self._prev_roll[side] = roll

        r_min, r_max = self._roll_range[side]
        t_min, t_max = self._tick_range[side]
        frac = max(0.0, min(1.0, (roll - r_min) / (r_max - r_min)))
        tick = int(t_min + frac * (t_max - t_min))

        last = self._sent_tick[side]
        if last is not None and abs(tick - last) < DEADBAND:
            return
        self._sent_tick[side] = tick

        out = String()
        out.data = json.dumps({"4": tick})
        self._pub[side].publish(out)

        self.get_logger().info(
            f"[{side.upper()}]  roll = {roll:+6.1f}°  →  tick = {tick:4d}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _roll_cal_exists() -> bool:
    for f in (CAL_LEFT, CAL_RIGHT):
        data = _load_wrist_file(f)
        if "roll_min" not in data or "roll_max" not in data:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="roll_servo4",
        description="Map VR wrist roll → SO-101 wrist_flex (servo 4), both arms",
    )
    parser.add_argument("--recalibrate", action="store_true",
                        help="Re-run the roll calibration even if saved data exists")
    args = parser.parse_args()

    needs_cal = args.recalibrate or not _roll_cal_exists()

    rclpy.init()

    if needs_cal:
        reader = RollReader()
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
            print("\n[roll_servo4] Calibration cancelled.")
            stop.set()
            reader.destroy_node()
            rclpy.shutdown()
            sys.exit(0)

        stop.set()
        spin_thread.join(timeout=2.0)
        reader.destroy_node()

    else:
        _banner("ROLL CALIBRATION LOADED")
        for side, f in [("left", CAL_LEFT), ("right", CAL_RIGHT)]:
            d = _load_wrist_file(f)
            print(f"  {side.upper():<6}  roll_min = {d['roll_min']:+.2f}°  "
                  f"roll_max = {d['roll_max']:+.2f}°")
        print("  (run with --recalibrate to redo)\n")

    node = RollServo4Node()
    print("[roll_servo4] Running — tilt your wrists to control servo 4.")
    print("              Press Ctrl-C to stop.\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[roll_servo4] Stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
