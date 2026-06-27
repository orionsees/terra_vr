#!/usr/bin/env python3
"""
wrist_live.py — Real-time wrist dashboard + SO-101 wrist_flex (servo 4) control.

Layout
------
  Row 0  Status boxes  — current X/Y/Z + Roll/Pitch/Yaw + active servo
  Row 1  POSITION  — X lateral | Y up/down | Z fwd/back   (scrolling, metres)
  Row 2  ORIENTATION  — Roll | Pitch | Yaw  (scrolling, degrees)

Topics subscribed:
    /left_wrist   /right_wrist   (geometry_msgs/PoseStamped)
Topics published:
    /left_arm/cmd/set_positions_raw    std_msgs/String  {"4": <tick>}
    /right_arm/cmd/set_positions_raw   std_msgs/String  {"4": <tick>}

Calibration files (shared with yaw_servo5.py):
    config/left_wrist.json
    config/right_wrist.json

Usage:
    python3 wrist_live.py               # 10-second window, 20 Hz refresh
    python3 wrist_live.py --window 5    # shorter time window
    python3 wrist_live.py --rate 30     # faster refresh
    python3 wrist_live.py --recalibrate # re-run roll calibration
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

# ── servo control constants ───────────────────────────────────────────────────
SPIKE_THRESHOLD = 20.0   # degrees
DEADBAND        = 15     # ticks

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

def quat_to_rotmat(qx, qy, qz, qw) -> np.ndarray:
    """Return 3×3 rotation matrix from unit quaternion (columns = rotated X/Y/Z axes)."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])


def quat_to_rpy(qx, qy, qz, qw):
    roll  = math.degrees(math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx)))))
    yaw   = math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
    return roll, pitch, yaw


def _activity(vals: np.ndarray) -> float:
    """Peak-to-peak amplitude over the window — uses both extremes so a burst
    in either direction registers immediately and switching is fast."""
    if len(vals) < 2:
        return 0.0
    return float(vals.max() - vals.min())


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


def _roll_cal_exists() -> bool:
    for f in (CAL_LEFT, CAL_RIGHT):
        data = _load_wrist_file(f)
        if "roll_min" not in data or "roll_max" not in data:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Combined ROS2 node — dashboard buffer + servo-4 control
# ─────────────────────────────────────────────────────────────────────────────

class WristNode(Node):
    """
    Subscribes to /left_wrist and /right_wrist.

    * Buffers every sample for the live dashboard.
    * When roll_range/tick_range are provided, drives servo 4 on both arms
      with spike rejection and deadband filtering.
    * send_tick() lets the calibration routine move the arm to a reference
      position before the user captures a roll value.
    """

    def __init__(self, maxlen: int,
                 roll_range: dict | None,
                 tick_range: dict | None):
        super().__init__("wrist_live")

        self._lock       = threading.Lock()
        self._buf        = {"left": deque(maxlen=maxlen), "right": deque(maxlen=maxlen)}
        self._t0         = None
        self._roll_range = roll_range   # None → servo control disabled
        self._tick_range = tick_range
        self._prev_roll  = {"left": None, "right": None}
        self._sent_tick  = {"left": None, "right": None}
        self._roll_active = True   # set by dashboard; True until first update

        qos     = QoSProfile(depth=300, reliability=ReliabilityPolicy.BEST_EFFORT)
        rel_qos = QoSProfile(depth=10)

        self.create_subscription(PoseStamped, "/left_wrist",
                                 lambda m: self._cb(m, "left"),  qos)
        self.create_subscription(PoseStamped, "/right_wrist",
                                 lambda m: self._cb(m, "right"), qos)

        self._pub = {
            "left":  self.create_publisher(String, "/left_arm/cmd/set_positions_raw",  rel_qos),
            "right": self.create_publisher(String, "/right_arm/cmd/set_positions_raw", rel_qos),
        }

        self.get_logger().info("wrist_live: listening on /left_wrist and /right_wrist")
        if roll_range:
            for side in ("left", "right"):
                rmin, rmax = roll_range[side]
                tmin, tmax = tick_range[side]
                self.get_logger().info(
                    f"  [{side.upper()}]  roll [{rmin:+.1f}°, {rmax:+.1f}°]"
                    f"  →  tick [{tmin}, {tmax}]"
                )

    def _cb(self, msg: PoseStamped, side: str) -> None:
        now = time.monotonic()
        p, q = msg.pose.position, msg.pose.orientation
        roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)

        with self._lock:
            if self._t0 is None:
                self._t0 = now
            self._buf[side].append({
                "t":    now - self._t0,
                "x":    p.x, "y": p.y, "z": p.z,
                "roll": roll, "pitch": pitch, "yaw": yaw,
                "qx":   q.x, "qy": q.y, "qz": q.z, "qw": q.w,
            })

        if self._roll_range is None or not self._roll_active:
            return

        # ── servo 4 control ───────────────────────────────────────────────────
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
        """Latest roll value — used by the calibration routine."""
        lat = self.latest(side)
        return lat["roll"] if lat else None

    # ── calibration helper ────────────────────────────────────────────────────

    def set_roll_active(self, active: bool) -> None:
        """Called by the dashboard each frame to gate servo-4 output."""
        self._roll_active = active

    def send_tick(self, side: str, tick: int) -> None:
        """Drive servo 4 on one arm to a specific tick."""
        msg = String()
        msg.data = json.dumps({"4": tick})
        self._pub[side].publish(msg)


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
        print(f"  {label:<38}  roll = {val:+7.1f}°   [Press Enter to capture]",
              flush=True)
        if _enter_pressed():
            val = node.get_roll(side)
            print(f"\n  ✓  Captured: {val:+.1f}°")
            return val
        time.sleep(0.08)


def run_calibration(node: WristNode) -> None:
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
        node.send_tick(side, t_min)
        time.sleep(0.7)
        print(f"           Tilt your {side.upper()} wrist to MATCH the arm. "
              f"Watch the reading, then press Enter.")
        roll_min = _capture_roll(node, side, f"{side.upper()} wrist MIN roll")

        print(f"\n  Step 2 — Moving {side.upper()} arm servo 4 → MAXIMUM ({t_max} ticks)")
        node.send_tick(side, t_max)
        time.sleep(0.7)
        print(f"           Tilt your {side.upper()} wrist to MATCH the arm. "
              f"Watch the reading, then press Enter.")
        roll_max = _capture_roll(node, side, f"{side.upper()} wrist MAX roll")

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
        4, 3, figure=fig,
        left=0.07, right=0.97,
        top=0.96, bottom=0.05,
        hspace=0.75, wspace=0.38,
        height_ratios=[0.5, 1.0, 1.0, 1.1],
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
                 "ORIENTATION  (degrees, from quaternion)\nRoll = tilt side    Pitch = tilt fwd    Yaw = rotate up",
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
        (0, "roll", "Roll  (°)", (-180, 0, 180)),
        (2, "yaw",  "Yaw  (°)",  (-180, -90, 0, 90, 180)),
    ]
    ori_lines = {}
    ori_axes  = []

    for idx, (col, key, ylabel, refs) in enumerate(ori_cfg):
        ax = _styled_ax(fig, gs[2, col], ylabel, ref_lines=refs)
        ax.set_title(f"ORIENTATION — {ylabel}", fontsize=7.5, color="#CCCCCC", pad=3)
        ori_axes.append((ax, key))
        for side in ("left", "right"):
            line, = ax.plot([], [], color=C[side], lw=1.5, alpha=0.9, label=side.title())
            ori_lines.setdefault(side, {})[idx] = line
        ax.legend(fontsize=6, loc="upper right",
                  facecolor="#222222", labelcolor="white", framealpha=0.7)

    # ── Row 3: quaternion visualisation ──────────────────────────────────────
    _AXIS_COLORS = ["#FF5555", "#55FF55", "#5599FF"]   # X=red  Y=green  Z=blue
    _AXIS_LABELS = ["X", "Y", "Z"]

    frame_axes  = {}   # side → Axes3D
    frame_lines = {}   # side → list of 3 Line3D objects (X, Y, Z)

    for col, side in [(0, "left"), (2, "right")]:
        ax3 = fig.add_subplot(gs[3, col], projection="3d")
        ax3.set_facecolor(PANEL)
        ax3.set_title(f"{side.upper()} wrist  —  quaternion frame",
                      fontsize=7.5, color=C[side], pad=3)
        ax3.set_xlim(-1.2, 1.2); ax3.set_ylim(-1.2, 1.2); ax3.set_zlim(-1.2, 1.2)
        ax3.set_xlabel("X", fontsize=6, color="#FF5555", labelpad=1)
        ax3.set_ylabel("Y", fontsize=6, color="#55FF55", labelpad=1)
        ax3.set_zlabel("Z", fontsize=6, color="#5599FF", labelpad=1)
        ax3.tick_params(colors=TICK, labelsize=5)
        ax3.xaxis.pane.fill = ax3.yaxis.pane.fill = ax3.zaxis.pane.fill = False
        ax3.set_facecolor(BG)
        ax3.view_init(elev=22, azim=40)
        # Draw reference grid lines at origin
        for v in np.linspace(-1, 1, 5):
            ax3.plot([-1, 1], [v, v], [0, 0], color=GRID, lw=0.4, alpha=0.5)
            ax3.plot([v, v], [-1, 1], [0, 0], color=GRID, lw=0.4, alpha=0.5)
        # Three lines: X Y Z axes of the wrist frame
        lines = []
        for color, label in zip(_AXIS_COLORS, _AXIS_LABELS):
            ln, = ax3.plot([0, 1], [0, 0], [0, 0], color=color, lw=2.5)
            ax3.text(1.1, 0, 0, label, color=color, fontsize=7)
            lines.append(ln)
        frame_axes[side]  = ax3
        frame_lines[side] = lines

    # ── Row 3 centre: quaternion components over time ─────────────────────────
    ax_quat = _styled_ax(fig, gs[3, 1], "quaternion", ref_lines=(0,))
    ax_quat.set_title("QUATERNION components", fontsize=7.5, color="#CCCCCC", pad=3)
    ax_quat.axhline(1,  color="#333333", lw=0.5, linestyle=":")
    ax_quat.axhline(-1, color="#333333", lw=0.5, linestyle=":")
    _QUAT_KEYS   = ["qw", "qx", "qy", "qz"]
    _QUAT_COLORS = ["#FFFFFF", "#FF5555", "#55FF55", "#5599FF"]
    _QUAT_STYLE  = ["-", "--", "--", "--"]
    quat_lines = {}
    for side in ("left", "right"):
        alpha = 0.9 if side == "left" else 0.5
        quat_lines[side] = {}
        for k, col, sty in zip(_QUAT_KEYS, _QUAT_COLORS, _QUAT_STYLE):
            ln, = ax_quat.plot([], [], color=col, lw=1.2, alpha=alpha,
                               linestyle=sty, label=f"{side[0]}.{k}")
            quat_lines[side][k] = ln
    ax_quat.legend(fontsize=5, loc="upper right",
                   facecolor="#222222", labelcolor="white", framealpha=0.7, ncol=2)

    # ── Animation update ──────────────────────────────────────────────────────

    KEYS = ["t", "x", "y", "z", "roll", "yaw", "qx", "qy", "qz", "qw"]

    _ORI_LABELS = {"roll": "Roll  (°)", "yaw": "Yaw  (°)"}
    _POS_LABELS = {"x": "X  lateral  (m)", "y": "Y  up / down  (m)", "z": "Z  fwd / back  (m)"}

    ACTIVITY_WINDOW = 1.5  # seconds — short window for snappy detection

    def _update(_frame):
        now_t = 0.0
        for side in ("left", "right"):
            lat = node.latest(side)
            if lat:
                now_t = max(now_t, lat["t"])
        t_lo   = max(0.0, now_t - window_s)
        act_lo = max(0.0, now_t - ACTIVITY_WINDOW)

        activities = {"left":  {"roll": 0.0, "yaw": 0.0, "x": 0.0, "y": 0.0, "z": 0.0},
                      "right": {"roll": 0.0, "yaw": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}}

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
            activities[side]["roll"] = _activity(d["roll"][act_mask])
            activities[side]["yaw"]  = _activity(d["yaw"][act_mask])
            activities[side]["x"]     = _activity(d["x"][act_mask])
            activities[side]["y"]     = _activity(d["y"][act_mask])
            activities[side]["z"]     = _activity(d["z"][act_mask])

            for col, (ax, key) in enumerate(pos_axes):
                pos_lines[side][col].set_data(t[mask], d[key][mask])
            for col, (ax, key) in enumerate(ori_axes):
                ori_lines[side][col].set_data(t[mask], d[key][mask])

            # quaternion time series
            for k in _QUAT_KEYS:
                quat_lines[side][k].set_data(t[mask], d[k][mask])

        # ── Status boxes ──────────────────────────────────────────────────────
        for side in ("left", "right"):
            lat = node.latest(side)
            if lat:
                act = activities[side]
                roll_act, yaw_act = act["roll"], act["yaw"]
                dom   = "roll" if roll_act >= yaw_act else "yaw"
                servo = "4"    if dom == "roll"       else "5"
                status_text[side].set_text(
                    f"{side.upper()}\n"
                    f"X {lat['x']:+.3f} m    Y {lat['y']:+.3f} m    Z {lat['z']:+.3f} m\n"
                    f"Roll {lat['roll']:+.1f}°   Yaw {lat['yaw']:+.1f}°\n"
                    f"► {dom.upper()} active  (σ {roll_act:.1f}° vs {yaw_act:.1f}°)  → servo {servo}"
                )

        # ── Highlight dominant orientation subplot ────────────────────────────
        combined_roll = max(activities[s]["roll"] for s in ("left", "right"))
        combined_yaw  = max(activities[s]["yaw"]  for s in ("left", "right"))
        overall_dom   = "roll" if combined_roll >= combined_yaw else "yaw"
        node.set_roll_active(overall_dom == "roll")

        for col, (ax, key) in enumerate(ori_axes):
            label = _ORI_LABELS[key]
            if key == overall_dom:
                servo = "4" if key == "roll" else "5"
                ax.set_title(f"ORIENTATION — {label}  ▶  servo {servo} ACTIVE",
                             fontsize=7.5, color="#00FF88", pad=3)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#00FF88"); sp.set_linewidth(2.0)
            else:
                dim_color = "#555555"
                ax.set_title(f"ORIENTATION — {label}", fontsize=7.5, color=dim_color, pad=3)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#333333"); sp.set_linewidth(1.0)

        for col, (ax, key) in enumerate(ori_axes):
            is_dom = (key == overall_dom)
            for side in ("left", "right"):
                ori_lines[side][col].set_linewidth(2.5 if is_dom else 0.8)
                ori_lines[side][col].set_alpha(0.95 if is_dom else 0.3)

        # ── Highlight dominant position subplot ───────────────────────────────
        pos_act = {k: max(activities[s][k] for s in ("left", "right"))
                   for k in ("x", "y", "z")}
        pos_dom = max(pos_act, key=pos_act.get)

        for col, (ax, key) in enumerate(pos_axes):
            label = _POS_LABELS[key]
            if key == pos_dom:
                ax.set_title(f"POSITION — {label}  ▶  ACTIVE",
                             fontsize=7.5, color="#00CCFF", pad=3)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#00CCFF"); sp.set_linewidth(2.0)
            else:
                ax.set_title(f"POSITION — {label}", fontsize=7.5, color="#555555", pad=3)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#333333"); sp.set_linewidth(1.0)

        for col, (ax, key) in enumerate(pos_axes):
            is_dom = (key == pos_dom)
            for side in ("left", "right"):
                pos_lines[side][col].set_linewidth(2.5 if is_dom else 0.8)
                pos_lines[side][col].set_alpha(0.95 if is_dom else 0.3)

        # ── 3D orientation frames ─────────────────────────────────────────────
        for side in ("left", "right"):
            lat = node.latest(side)
            if lat:
                R = quat_to_rotmat(lat["qx"], lat["qy"], lat["qz"], lat["qw"])
                for i, line in enumerate(frame_lines[side]):
                    col = R[:, i]   # rotated ith unit axis
                    line.set_data_3d([0, col[0]], [0, col[1]], [0, col[2]])

        # ── Quaternion time series auto-scale ─────────────────────────────────
        ax_quat.set_xlim(t_lo, max(t_lo + window_s, now_t + 0.5))
        ax_quat.set_ylim(-1.15, 1.15)

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
        description="Real-time wrist dashboard + SO-101 wrist_flex (servo 4) control",
    )
    parser.add_argument("--window", type=float, default=10.0,
                        help="Scrolling time window in seconds (default: 10)")
    parser.add_argument("--rate",   type=float, default=20.0,
                        help="Plot refresh rate in Hz (default: 20)")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Re-run the roll calibration even if saved data exists")
    args = parser.parse_args()

    rclpy.init()

    # ── Phase 1: calibration (if needed) ─────────────────────────────────────
    if args.recalibrate or not _roll_cal_exists():
        cal_node = WristNode(maxlen=300, roll_range=None, tick_range=None)
        stop     = threading.Event()

        def _spin_cal():
            while not stop.is_set():
                rclpy.spin_once(cal_node, timeout_sec=0.02)

        spin_thread = threading.Thread(target=_spin_cal, daemon=True)
        spin_thread.start()

        try:
            run_calibration(cal_node)
            _banner("CALIBRATION COMPLETE")
            print(f"  left_wrist.json  → {CAL_LEFT}")
            print(f"  right_wrist.json → {CAL_RIGHT}\n")
        except KeyboardInterrupt:
            print("\n[wrist_live] Calibration cancelled.")
            stop.set()
            cal_node.destroy_node()
            rclpy.shutdown()
            sys.exit(0)

        stop.set()
        spin_thread.join(timeout=2.0)
        cal_node.destroy_node()

    else:
        _banner("ROLL CALIBRATION LOADED")
        for side, f in [("left", CAL_LEFT), ("right", CAL_RIGHT)]:
            d = _load_wrist_file(f)
            print(f"  {side.upper():<6}  roll_min = {d['roll_min']:+.2f}°  "
                  f"roll_max = {d['roll_max']:+.2f}°")
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
    tick_range = {
        "left":  (srv_left["wrist_flex"]["range_min"],
                  srv_left["wrist_flex"]["range_max"]),
        "right": (srv_right["wrist_flex"]["range_min"],
                  srv_right["wrist_flex"]["range_max"]),
    }

    maxlen = int(args.window * 150)
    node   = WristNode(maxlen=maxlen, roll_range=roll_range, tick_range=tick_range)
    stop   = threading.Event()

    def _spin():
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.02)

    spin_thread = threading.Thread(target=_spin, daemon=True, name="ros-spin")
    spin_thread.start()

    print("[wrist_live] Dashboard open — move your wrists.")
    print("             Close the window or press Ctrl-C to exit.\n")

    try:
        fig, ani = build_dashboard(node, args.window, args.rate)
        plt.show()
    except KeyboardInterrupt:
        print("\n[wrist_live] Stopped.")
    finally:
        stop.set()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
