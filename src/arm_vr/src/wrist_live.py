#!/usr/bin/env python3
"""
wrist_live.py — Real-time wrist dashboard + SO-101 servo control (IPC bus).

  Servo 4 (wrist_flex)  ←  wrist roll
  Servo 5 (wrist_roll)  ←  wrist yaw

Both servos run simultaneously and independently.

Layout
------
  Row 0  Status boxes  — current X/Y/Z + Roll/Pitch/Yaw for each wrist
  Row 1  POSITION  — X lateral | Y up/down | Z fwd/back   (scrolling, metres)
  Row 2  ORIENTATION  — Roll | Pitch | Yaw  (scrolling, degrees)

IPC bus (no ROS2):
    Subscribes:  port 5555  topics left_wrist / right_wrist
    Publishes:   port 5556  left arm cmds
                 port 5557  right arm cmds

Calibration files:
    config/left_wrist.json
    config/right_wrist.json

Usage:
    python3 wrist_live.py               # 10-second window, 20 Hz refresh
    python3 wrist_live.py --window 5    # shorter time window
    python3 wrist_live.py --rate 30     # faster refresh
    python3 wrist_live.py --recalibrate # re-run calibration
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

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bus import (
    Subscriber, CommandClient,
    VR_DATA_PORT, LEFT_ARM_CMD_PORT, RIGHT_ARM_CMD_PORT,
)


# ── paths ─────────────────────────────────────────────────────────────────────
_CONFIG_DIR     = (Path(__file__).parent / ".." / "config").resolve()
CAL_LEFT        = _CONFIG_DIR / "left_wrist.json"
CAL_RIGHT       = _CONFIG_DIR / "right_wrist.json"
SERVO_CAL_LEFT  = _CONFIG_DIR / "left_arm.json"
SERVO_CAL_RIGHT = _CONFIG_DIR / "right_arm.json"

# ── servo control constants ───────────────────────────────────────────────────
SPIKE_THRESHOLD    = 20.0   # degrees — per-servo, independent
DEADBAND           = 15     # ticks  — per-servo, independent
YAW_MODE_THRESHOLD = 30.0   # degrees — both arms must be within ±this for roll mode

# ── palette ───────────────────────────────────────────────────────────────────
C     = {"left": "#E67E22", "right": "#2980B9"}
C_DIM = {"left": "#5D3A0A", "right": "#154360"}
BG    = "#161616"
PANEL = "#1E1E1E"
GRID  = "#2A2A2A"
TICK  = "#666666"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_rpy(qx, qy, qz, qw):
    roll  = math.degrees(math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx)))))
    yaw   = math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
    return roll, pitch, yaw


def _activity(vals: np.ndarray) -> float:
    """Peak-to-peak amplitude over the window."""
    if len(vals) < 2:
        return 0.0
    return float(vals.max() - vals.min())


def _banner(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


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


def _cal_exists() -> bool:
    for f in (CAL_LEFT, CAL_RIGHT):
        data = _load_wrist_file(f)
        for key in ("roll_min", "roll_max", "yaw_min", "yaw_max"):
            if key not in data:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Combined IPC node — dashboard buffer + servo 4 (roll) + servo 5 (yaw)
# ─────────────────────────────────────────────────────────────────────────────

class WristNode:
    """
    Subscribes to IPC bus port 5555 for left_wrist and right_wrist topics.

    * Buffers every sample for the live dashboard.
    * Drives servo 4 (wrist_flex) from roll and servo 5 (wrist_roll) from yaw
      simultaneously, each with independent spike rejection and deadband.
    * send_tick(side, servo_id, tick) lets calibration position the arm.
    """

    def __init__(self, maxlen: int,
                 roll_range:      dict | None,
                 roll_tick_range: dict | None,
                 yaw_range:       dict | None,
                 yaw_tick_range:  dict | None):

        self._lock            = threading.Lock()
        self._buf             = {"left": deque(maxlen=maxlen), "right": deque(maxlen=maxlen)}
        self._t0              = None

        self._roll_range      = roll_range
        self._roll_tick_range = roll_tick_range
        self._yaw_range       = yaw_range
        self._yaw_tick_range  = yaw_tick_range

        self._prev_roll  = {"left": None, "right": None}
        self._prev_yaw   = {"left": None, "right": None}
        self._sent_tick4 = {"left": None, "right": None}
        self._sent_tick5 = {"left": None, "right": None}
        self._last_yaw   = {"left": 0.0,  "right": 0.0}

        self._cmd = {
            "left":  CommandClient(LEFT_ARM_CMD_PORT),
            "right": CommandClient(RIGHT_ARM_CMD_PORT),
        }

        self._stop = threading.Event()
        self._sub_thread = threading.Thread(
            target=self._sub_loop, daemon=True, name="vr-sub"
        )
        self._sub_thread.start()

        print("[wrist_live] listening on IPC bus port 5555 (left_wrist, right_wrist)")
        if roll_range:
            for side in ("left", "right"):
                rmin, rmax = roll_range[side]
                tmin, tmax = roll_tick_range[side]
                print(
                    f"  [{side.upper()}] servo4 roll [{rmin:+.1f}°, {rmax:+.1f}°]"
                    f"  →  tick [{tmin}, {tmax}]"
                )
        if yaw_range:
            for side in ("left", "right"):
                ymin, ymax = yaw_range[side]
                tmin, tmax = yaw_tick_range[side]
                print(
                    f"  [{side.upper()}] servo5 yaw  [{ymin:+.1f}°, {ymax:+.1f}°]"
                    f"  →  tick [{tmin}, {tmax}]"
                )

    def _sub_loop(self) -> None:
        sub = Subscriber(VR_DATA_PORT)
        for msg in sub.iter_messages():
            if self._stop.is_set():
                break
            topic = msg.get("topic", "")
            data  = msg.get("data", {})
            if topic in ("left_wrist", "right_wrist"):
                side = "left" if topic.startswith("left") else "right"
                self._cb(data, side)

    def _cb(self, data: dict, side: str) -> None:
        now = time.monotonic()
        x, y, z = data["x"], data["y"], data["z"]
        roll, pitch, yaw = quat_to_rpy(data["qx"], data["qy"], data["qz"], data["qw"])

        with self._lock:
            if self._t0 is None:
                self._t0 = now
            self._buf[side].append({
                "t":    now - self._t0,
                "x":    x, "y": y, "z": z,
                "roll": roll, "pitch": pitch, "yaw": yaw,
                "qx":   data["qx"], "qy": data["qy"],
                "qz":   data["qz"], "qw": data["qw"],
            })

        # ── mode decision: per-arm, each arm checks its own yaw independently ───
        self._last_yaw[side] = yaw
        roll_mode = abs(yaw) <= YAW_MODE_THRESHOLD

        # ── servo 4: roll → wrist_flex  (roll mode only) ─────────────────────
        if self._roll_range is not None and roll_mode:
            prev = self._prev_roll[side]
            if prev is not None and abs(roll - prev) > SPIKE_THRESHOLD:
                print(f"[{side.upper()}] roll spike rejected  {prev:+.1f}° → {roll:+.1f}°")
            else:
                self._prev_roll[side] = roll
                r_min, r_max = self._roll_range[side]
                t_min, t_max = self._roll_tick_range[side]
                frac = max(0.0, min(1.0, (roll - r_min) / (r_max - r_min)))
                tick = int(t_min + frac * (t_max - t_min))
                last = self._sent_tick4[side]
                if last is None or abs(tick - last) >= DEADBAND:
                    self._sent_tick4[side] = tick
                    self._cmd[side].send({"topic": "set_positions_raw", "data": {"4": tick}})

        # ── servo 5: yaw → wrist_roll  (yaw mode only) ───────────────────────
        if self._yaw_range is not None and not roll_mode:
            prev = self._prev_yaw[side]
            if prev is not None and abs(yaw - prev) > SPIKE_THRESHOLD:
                print(f"[{side.upper()}] yaw spike rejected  {prev:+.1f}° → {yaw:+.1f}°")
            else:
                self._prev_yaw[side] = yaw
                y_min, y_max = self._yaw_range[side]
                t_min, t_max = self._yaw_tick_range[side]
                frac = max(0.0, min(1.0, (yaw - y_min) / (y_max - y_min)))
                tick = int(t_min + frac * (t_max - t_min))
                last = self._sent_tick5[side]
                if last is None or abs(tick - last) >= DEADBAND:
                    self._sent_tick5[side] = tick
                    self._cmd[side].send({"topic": "set_positions_raw", "data": {"5": tick}})

    # ── dashboard helpers ─────────────────────────────────────────────────────

    def arrays(self, side: str, keys: list[str]):
        with self._lock:
            buf = list(self._buf[side])
        if not buf:
            return {k: np.array([]) for k in keys}
        return {k: np.array([r[k] for r in buf]) for k in keys}

    def latest(self, side: str):
        with self._lock:
            b = self._buf[side]
            return b[-1] if b else None

    def get_roll(self, side: str):
        lat = self.latest(side)
        return lat["roll"] if lat else None

    def get_yaw(self, side: str):
        lat = self.latest(side)
        return lat["yaw"] if lat else None

    def get_mode(self, side: str) -> str:
        """'roll' when this arm's yaw is within threshold, 'yaw' otherwise."""
        return "roll" if abs(self._last_yaw[side]) <= YAW_MODE_THRESHOLD else "yaw"

    # ── calibration helper ────────────────────────────────────────────────────

    def send_tick(self, side: str, servo_id: int, tick: int) -> None:
        """Drive one servo on one arm to a specific tick (used during calibration)."""
        self._cmd[side].send({"topic": "set_positions_raw", "data": {str(servo_id): tick}})

    def close(self) -> None:
        self._stop.set()
        for c in self._cmd.values():
            c.close()


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def _capture_roll(node: WristNode, side: str, label: str) -> float:
    print(f"\n  Waiting for {side.upper()} wrist data…")
    while node.get_roll(side) is None:
        time.sleep(0.05)

    first_print = True
    while True:
        val = node.get_roll(side)
        if not first_print:
            _cursor_up(1)
        first_print = False
        print(f"  {label:<42}  roll = {val:+7.1f}°   [Press Enter to capture]",
              flush=True)
        if _enter_pressed():
            val = node.get_roll(side)
            print(f"\n  ✓  Captured roll: {val:+.1f}°")
            return val
        time.sleep(0.08)


def _capture_yaw(node: WristNode, side: str, label: str) -> float:
    print(f"\n  Waiting for {side.upper()} wrist data…")
    while node.get_yaw(side) is None:
        time.sleep(0.05)

    first_print = True
    while True:
        val = node.get_yaw(side)
        if not first_print:
            _cursor_up(1)
        first_print = False
        print(f"  {label:<42}  yaw  = {val:+7.1f}°   [Press Enter to capture]",
              flush=True)
        if _enter_pressed():
            val = node.get_yaw(side)
            print(f"\n  ✓  Captured yaw: {val:+.1f}°")
            return val
        time.sleep(0.08)


def run_calibration(node: WristNode) -> None:
    _banner("WRIST CALIBRATION  —  servo 4 (roll)  +  servo 5 (yaw)")

    srv_cal = {
        "left":  json.loads(SERVO_CAL_LEFT.read_text()),
        "right": json.loads(SERVO_CAL_RIGHT.read_text()),
    }
    roll_tick_range = {
        side: (srv_cal[side]["wrist_flex"]["range_min"],
               srv_cal[side]["wrist_flex"]["range_max"])
        for side in ("left", "right")
    }
    yaw_tick_range = {
        side: (srv_cal[side]["wrist_roll"]["range_min"],
               srv_cal[side]["wrist_roll"]["range_max"])
        for side in ("left", "right")
    }

    print("""
  The arm will move each servo to its physical limits one at a time.
  Match your wrist to the arm position, let the reading stabilise,
  then press Enter to lock in the mapping.
""")

    for side, cal_file in [("left", CAL_LEFT), ("right", CAL_RIGHT)]:

        # ── servo 4: roll calibration ─────────────────────────────────────────
        _banner(f"{side.upper()} WRIST  —  servo 4  (roll → wrist_flex)")
        t4_min, t4_max = roll_tick_range[side]

        print(f"\n  Step 1 — Moving {side.upper()} servo 4 → MINIMUM ({t4_min} ticks)")
        node.send_tick(side, 4, t4_min)
        time.sleep(0.7)
        print(f"           Tilt your {side.upper()} wrist to MATCH the arm position.")
        roll_min = _capture_roll(node, side, f"{side.upper()} servo-4 MIN")

        print(f"\n  Step 2 — Moving {side.upper()} servo 4 → MAXIMUM ({t4_max} ticks)")
        node.send_tick(side, 4, t4_max)
        time.sleep(0.7)
        print(f"           Tilt your {side.upper()} wrist to MATCH the arm position.")
        roll_max = _capture_roll(node, side, f"{side.upper()} servo-4 MAX")

        if roll_min > roll_max:
            roll_min, roll_max = roll_max, roll_min
            print("  (min/max swapped so that min < max)")

        # ── servo 5: yaw calibration ──────────────────────────────────────────
        _banner(f"{side.upper()} WRIST  —  servo 5  (yaw → wrist_roll)")
        t5_min, t5_max = yaw_tick_range[side]

        print(f"\n  Step 3 — Moving {side.upper()} servo 5 → MINIMUM ({t5_min} ticks)")
        node.send_tick(side, 5, t5_min)
        time.sleep(0.7)
        print(f"           Rotate your {side.upper()} wrist to MATCH the arm position.")
        yaw_min = _capture_yaw(node, side, f"{side.upper()} servo-5 MIN")

        print(f"\n  Step 4 — Moving {side.upper()} servo 5 → MAXIMUM ({t5_max} ticks)")
        node.send_tick(side, 5, t5_max)
        time.sleep(0.7)
        print(f"           Rotate your {side.upper()} wrist to MATCH the arm position.")
        yaw_max = _capture_yaw(node, side, f"{side.upper()} servo-5 MAX")

        if yaw_min > yaw_max:
            yaw_min, yaw_max = yaw_max, yaw_min
            print("  (min/max swapped so that min < max)")

        _save_wrist_file(cal_file, {
            "roll_min": round(roll_min, 2),
            "roll_max": round(roll_max, 2),
            "yaw_min":  round(yaw_min,  2),
            "yaw_max":  round(yaw_max,  2),
        })

        print(f"\n  Saved → {cal_file}")
        print(f"  {side.upper()}  roll [{roll_min:+.2f}°, {roll_max:+.2f}°]"
              f"  span {roll_max - roll_min:.2f}°")
        print(f"  {side.upper()}  yaw  [{yaw_min:+.2f}°, {yaw_max:+.2f}°]"
              f"  span {yaw_max - yaw_min:.2f}°")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def _styled_ax(fig, spec, ylabel, ref_lines=()):
    ax = fig.add_subplot(spec)
    ax.set_facecolor(PANEL)
    ax.set_ylabel(ylabel, fontsize=8, color=TICK)
    ax.set_xlabel("Time  (s)", fontsize=7, color=TICK)
    ax.tick_params(colors=TICK, labelsize=7)
    ax.grid(True, color=GRID, linewidth=0.7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    for y in ref_lines:
        ax.axhline(y, color="#444444", lw=0.8, linestyle="--")
    return ax


def build_dashboard(node: WristNode, window_s: float, rate_hz: float):

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle("VR Wrist Live  ·  orange = LEFT   blue = RIGHT",
                 fontsize=11, fontweight="bold", color="white", y=0.99)

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        left=0.07, right=0.97,
        top=0.96, bottom=0.07,
        hspace=0.65, wspace=0.38,
        height_ratios=[0.5, 1.0, 1.0],
    )

    # ── Row 0: status boxes ───────────────────────────────────────────────────
    status_text = {}
    for col, side in [(0, "left"), (2, "right")]:
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(C_DIM[side])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(C[side]); sp.set_linewidth(2)
        txt = ax.text(0.5, 0.5, f"{side.upper()}\n…",
                      transform=ax.transAxes, ha="center", va="center",
                      fontsize=9, family="monospace", fontweight="bold", color="white")
        status_text[side] = txt

    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.set_facecolor(BG); ax_info.axis("off")
    ax_info.text(0.5, 0.6,
                 "POSITION  (metres)\nX = left / right    Y = up / down    Z = fwd / back",
                 transform=ax_info.transAxes, ha="center", va="center",
                 fontsize=7.5, color="#AAAAAA", family="monospace")
    ax_info.text(0.5, 0.2,
                 f"ORIENTATION  (degrees)\nRoll→servo4 when |yaw|≤{YAW_MODE_THRESHOLD:.0f}°  else Yaw→servo5",
                 transform=ax_info.transAxes, ha="center", va="center",
                 fontsize=7.5, color="#AAAAAA", family="monospace")

    # ── Row 1: POSITION time-series ───────────────────────────────────────────
    pos_cfg = [
        (0, "x", "X  lateral  (m)"),
        (1, "y", "Y  up / down  (m)"),
        (2, "z", "Z  fwd / back  (m)"),
    ]
    pos_lines = {}
    pos_axes  = []

    for col, key, ylabel in pos_cfg:
        ax = _styled_ax(fig, gs[1, col], ylabel, ref_lines=(0,))
        ax.set_title(f"POSITION — {ylabel}", fontsize=7.5, color="#CCCCCC", pad=3)
        pos_axes.append((ax, key))
        for side in ("left", "right"):
            line, = ax.plot([], [], color=C[side], lw=1.5, alpha=0.9, label=side.title())
            pos_lines.setdefault(side, {})[col] = line
        ax.legend(fontsize=6, loc="upper right",
                  facecolor="#222222", labelcolor="white", framealpha=0.7)

    # ── Row 2: ORIENTATION time-series ────────────────────────────────────────
    ori_cfg = [
        (0, "roll",  "Roll  (°)",  (-180, 0, 180)),
        (1, "pitch", "Pitch  (°)", (-90, 0, 90)),
        (2, "yaw",   "Yaw  (°)",   (-180, -90, 0, 90, 180)),
    ]
    ori_lines = {}
    ori_axes  = []

    for col, key, ylabel, refs in ori_cfg:
        ax = _styled_ax(fig, gs[2, col], ylabel, ref_lines=refs)
        ax.set_title(f"ORIENTATION — {ylabel}", fontsize=7.5, color="#CCCCCC", pad=3)
        ori_axes.append((ax, key))
        for side in ("left", "right"):
            line, = ax.plot([], [], color=C[side], lw=1.5, alpha=0.9, label=side.title())
            ori_lines.setdefault(side, {})[col] = line
        ax.legend(fontsize=6, loc="upper right",
                  facecolor="#222222", labelcolor="white", framealpha=0.7)

    # ── Animation update ──────────────────────────────────────────────────────

    KEYS = ["t", "x", "y", "z", "roll", "pitch", "yaw"]

    _ORI_LABELS = {"roll": "Roll  (°)", "pitch": "Pitch  (°)", "yaw": "Yaw  (°)"}
    _POS_LABELS = {"x": "X  lateral  (m)", "y": "Y  up / down  (m)", "z": "Z  fwd / back  (m)"}

    _SERVO_LABEL = {"roll": "servo 4", "yaw": "servo 5"}

    ACTIVITY_WINDOW = 1.5  # seconds — short window for snappy detection

    def _update(_frame):
        now_t = 0.0
        for side in ("left", "right"):
            lat = node.latest(side)
            if lat:
                now_t = max(now_t, lat["t"])
        t_lo   = max(0.0, now_t - window_s)
        act_lo = max(0.0, now_t - ACTIVITY_WINDOW)

        activities = {"left":  {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "x": 0.0, "y": 0.0, "z": 0.0},
                      "right": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}}

        for side in ("left", "right"):
            d = node.arrays(side, KEYS)
            t = d["t"]

            if t.size == 0:
                for col, (ax, key) in enumerate(pos_axes):
                    pos_lines[side][col].set_data([], [])
                for col, (ax, key) in enumerate(ori_axes):
                    ori_lines[side][col].set_data([], [])
                continue

            mask     = t >= t_lo
            act_mask = t >= act_lo
            activities[side]["roll"]  = _activity(d["roll"][act_mask])
            activities[side]["pitch"] = _activity(d["pitch"][act_mask])
            activities[side]["yaw"]   = _activity(d["yaw"][act_mask])
            activities[side]["x"]     = _activity(d["x"][act_mask])
            activities[side]["y"]     = _activity(d["y"][act_mask])
            activities[side]["z"]     = _activity(d["z"][act_mask])

            for col, (ax, key) in enumerate(pos_axes):
                pos_lines[side][col].set_data(t[mask], d[key][mask])
            for col, (ax, key) in enumerate(ori_axes):
                ori_lines[side][col].set_data(t[mask], d[key][mask])

        # ── Status boxes (per-arm mode) ───────────────────────────────────────
        for side in ("left", "right"):
            lat = node.latest(side)
            if lat:
                mode = node.get_mode(side)
                mode_line = ("◆ ROLL MODE  (servo 4 active)" if mode == "roll"
                             else "◆ YAW  MODE  (servo 5 active)")
                status_text[side].set_text(
                    f"{side.upper()}\n"
                    f"X {lat['x']:+.3f} m   Y {lat['y']:+.3f} m   Z {lat['z']:+.3f} m\n"
                    f"Roll {lat['roll']:+.1f}°   Yaw {lat['yaw']:+.1f}°\n"
                    f"{mode_line}"
                )
                status_text[side].set_color("#AAFFAA" if mode == "roll" else "#FFDD88")

        # ── Highlight most-active orientation subplot (visual only) ───────────
        combined = {k: max(activities[s][k] for s in ("left", "right"))
                    for k in ("roll", "pitch", "yaw")}
        most_active = max(combined, key=combined.get)

        for col, (ax, key) in enumerate(ori_axes):
            label = _ORI_LABELS[key]
            if key == most_active:
                suffix = f"  ▶  {_SERVO_LABEL[key]}" if key in _SERVO_LABEL else "  ▶  active"
                ax.set_title(f"ORIENTATION — {label}{suffix}",
                             fontsize=7.5, color="#00FF88", pad=3)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#00FF88"); sp.set_linewidth(2.0)
            else:
                dim_color = "#888888" if key == "pitch" else "#555555"
                ax.set_title(f"ORIENTATION — {label}", fontsize=7.5, color=dim_color, pad=3)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#333333"); sp.set_linewidth(1.0)

        for col, (ax, key) in enumerate(ori_axes):
            is_active = (key == most_active)
            for side in ("left", "right"):
                ori_lines[side][col].set_linewidth(2.5 if is_active else 0.8)
                ori_lines[side][col].set_alpha(0.95 if is_active else 0.3)

        # ── Highlight most-active position subplot ────────────────────────────
        pos_combined = {k: max(activities[s][k] for s in ("left", "right"))
                        for k in ("x", "y", "z")}
        pos_active = max(pos_combined, key=pos_combined.get)

        for col, (ax, key) in enumerate(pos_axes):
            label = _POS_LABELS[key]
            if key == pos_active:
                ax.set_title(f"POSITION — {label}  ▶  active",
                             fontsize=7.5, color="#00CCFF", pad=3)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#00CCFF"); sp.set_linewidth(2.0)
            else:
                ax.set_title(f"POSITION — {label}", fontsize=7.5, color="#555555", pad=3)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#333333"); sp.set_linewidth(1.0)

        for col, (ax, key) in enumerate(pos_axes):
            is_active = (key == pos_active)
            for side in ("left", "right"):
                pos_lines[side][col].set_linewidth(2.5 if is_active else 0.8)
                pos_lines[side][col].set_alpha(0.95 if is_active else 0.3)

        # ── Auto-scale all axes ───────────────────────────────────────────────
        for ax_list, line_dict in [(pos_axes, pos_lines), (ori_axes, ori_lines)]:
            for col, (ax, key) in enumerate(ax_list):
                ax.set_xlim(t_lo, max(t_lo + window_s, now_t + 0.5))
                all_v = []
                for side in ("left", "right"):
                    ln = line_dict[side][col]
                    ydata = ln.get_ydata()
                    if len(ydata):
                        all_v.append(ydata)
                if all_v:
                    ys = np.concatenate(all_v)
                    pad = max(0.5, (ys.max() - ys.min()) * 0.12)
                    ax.set_ylim(ys.min() - pad, ys.max() + pad)

    ani = FuncAnimation(fig, _update,
                        interval=1000.0 / rate_hz,
                        blit=False,
                        cache_frame_data=False)
    return fig, ani


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="wrist_live",
        description="Real-time wrist dashboard + SO-101 servo 4 (roll) & servo 5 (yaw) control",
    )
    parser.add_argument("--window", type=float, default=10.0,
                        help="Scrolling time window in seconds (default: 10)")
    parser.add_argument("--rate",   type=float, default=20.0,
                        help="Plot refresh rate in Hz (default: 20)")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Re-run the full calibration (roll + yaw) even if data exists")
    args = parser.parse_args()

    # ── Phase 1: calibration (if needed) ─────────────────────────────────────
    if args.recalibrate or not _cal_exists():
        cal_node = WristNode(maxlen=300,
                             roll_range=None, roll_tick_range=None,
                             yaw_range=None,  yaw_tick_range=None)
        try:
            run_calibration(cal_node)
            _banner("CALIBRATION COMPLETE")
            print(f"  left_wrist.json  → {CAL_LEFT}")
            print(f"  right_wrist.json → {CAL_RIGHT}\n")
        except KeyboardInterrupt:
            print("\n[wrist_live] Calibration cancelled.")
            cal_node.close()
            sys.exit(0)
        cal_node.close()

    else:
        _banner("CALIBRATION LOADED")
        for side, f in [("left", CAL_LEFT), ("right", CAL_RIGHT)]:
            d = _load_wrist_file(f)
            print(f"  {side.upper()}  roll [{d['roll_min']:+.2f}°, {d['roll_max']:+.2f}°]"
                  f"   yaw [{d['yaw_min']:+.2f}°, {d['yaw_max']:+.2f}°]")
        print("  (run with --recalibrate to redo)\n")

    # ── Phase 2: live dashboard + servo control ───────────────────────────────
    left_cal  = _load_wrist_file(CAL_LEFT)
    right_cal = _load_wrist_file(CAL_RIGHT)
    srv_left  = json.loads(SERVO_CAL_LEFT.read_text())
    srv_right = json.loads(SERVO_CAL_RIGHT.read_text())

    roll_range = {
        "left":  (left_cal["roll_min"],  left_cal["roll_max"]),
        "right": (right_cal["roll_min"], right_cal["roll_max"]),
    }
    roll_tick_range = {
        "left":  (srv_left["wrist_flex"]["range_min"],
                  srv_left["wrist_flex"]["range_max"]),
        "right": (srv_right["wrist_flex"]["range_min"],
                  srv_right["wrist_flex"]["range_max"]),
    }
    yaw_range = {
        "left":  (left_cal["yaw_min"],  left_cal["yaw_max"]),
        "right": (right_cal["yaw_min"], right_cal["yaw_max"]),
    }
    yaw_tick_range = {
        "left":  (srv_left["wrist_roll"]["range_min"],
                  srv_left["wrist_roll"]["range_max"]),
        "right": (srv_right["wrist_roll"]["range_min"],
                  srv_right["wrist_roll"]["range_max"]),
    }

    maxlen = int(args.window * 150)
    node   = WristNode(maxlen=maxlen,
                       roll_range=roll_range, roll_tick_range=roll_tick_range,
                       yaw_range=yaw_range,   yaw_tick_range=yaw_tick_range)

    print("[wrist_live] Dashboard open — move your wrists.")
    print("             Servo 4 tracks roll, servo 5 tracks yaw.")
    print("             Close the window or press Ctrl-C to exit.\n")

    try:
        fig, ani = build_dashboard(node, args.window, args.rate)
        plt.show()
    except KeyboardInterrupt:
        print("\n[wrist_live] Stopped.")
    finally:
        node.close()


if __name__ == "__main__":
    main()
