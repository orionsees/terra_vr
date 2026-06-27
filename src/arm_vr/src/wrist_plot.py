#!/usr/bin/env python3
"""Live wrist data visualiser — plots only, no servo control.

Subscribes to IPC bus port 5555 (left_wrist / right_wrist topics) and shows
a scrolling dashboard of position, orientation, and quaternion components.

Usage:
    python3 wrist_plot.py
    python3 wrist_plot.py --window 5   # shorter time window
    python3 wrist_plot.py --rate 30    # faster refresh
"""

from __future__ import annotations

import argparse
import math
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
from bus import Subscriber, VR_DATA_PORT


# ── filtering ────────────────────────────────────────────────────────────────
SPIKE_ROT_DEG = 20.0   # max quaternion angular jump between consecutive samples
SPIKE_POS_M   = 0.05   # max per-axis position jump between consecutive samples

# ── palette ───────────────────────────────────────────────────────────────────
C     = {"left": "#E67E22", "right": "#2980B9"}
C_DIM = {"left": "#5D3A0A", "right": "#154360"}
BG    = "#161616"
PANEL = "#1E1E1E"
GRID  = "#2A2A2A"
TICK  = "#666666"

# quaternion component colours (shared across both sides; side = linestyle)
QUAT_COLORS = {"qx": "#FF6B6B", "qy": "#6BCB77", "qz": "#FFD93D", "qw": "#C77DFF"}


def _quat_to_rpy(qx, qy, qz, qw):
    roll  = math.degrees(math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2*(qw*qy - qz*qx)))))
    yaw   = math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
    return roll, pitch, yaw


class WristBuffer:
    """Subscribes to IPC bus and buffers wrist data for both sides."""

    def __init__(self, maxlen: int) -> None:
        self._lock = threading.Lock()
        self._buf  = {"left": deque(maxlen=maxlen), "right": deque(maxlen=maxlen)}
        self._t0   = None

        self._stop = threading.Event()
        threading.Thread(target=self._sub_loop, daemon=True, name="vr-sub").start()

    def _sub_loop(self) -> None:
        sub = Subscriber(VR_DATA_PORT)
        for msg in sub.iter_messages():
            if self._stop.is_set():
                break
            topic = msg.get("topic", "")
            data  = msg.get("data", {})
            if topic in ("left_wrist", "right_wrist"):
                side = "left" if topic.startswith("left") else "right"
                self._ingest(data, side)

    def _ingest(self, data: dict, side: str) -> None:
        now = time.monotonic()
        qx, qy, qz, qw = data["qx"], data["qy"], data["qz"], data["qw"]

        with self._lock:
            if self._buf[side]:
                prev = self._buf[side][-1]

                # Quaternion angular distance — catches orientation spikes without
                # RPY wrap-around issues. dot product gives cos(θ/2); full angle = 2*acos.
                dot = abs(prev["qx"]*qx + prev["qy"]*qy + prev["qz"]*qz + prev["qw"]*qw)
                if math.degrees(2 * math.acos(min(1.0, dot))) > SPIKE_ROT_DEG:
                    return

                # Per-axis position spike check
                if (abs(data["x"] - prev["x"]) > SPIKE_POS_M or
                        abs(data["y"] - prev["y"]) > SPIKE_POS_M or
                        abs(data["z"] - prev["z"]) > SPIKE_POS_M):
                    return

            if self._t0 is None:
                self._t0 = now

            roll, pitch, yaw = _quat_to_rpy(qx, qy, qz, qw)
            self._buf[side].append({
                "t":    now - self._t0,
                "x":    data["x"], "y": data["y"], "z": data["z"],
                "roll": roll, "pitch": pitch, "yaw": yaw,
                "qx":   qx, "qy": qy, "qz": qz, "qw": qw,
            })

    def arrays(self, side: str, keys: list[str]) -> dict:
        with self._lock:
            buf = list(self._buf[side])
        if not buf:
            return {k: np.array([]) for k in keys}
        return {k: np.array([r[k] for r in buf]) for k in keys}

    def latest(self, side: str) -> dict | None:
        with self._lock:
            b = self._buf[side]
            return b[-1] if b else None

    def close(self) -> None:
        self._stop.set()


def _styled_ax(fig, spec, ylabel, ref_lines=(), ylim=None):
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
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


def build_dashboard(buf: WristBuffer, window_s: float, rate_hz: float):
    KEYS = ["t", "x", "y", "z", "roll", "pitch", "yaw", "qx", "qy", "qz", "qw"]

    fig = plt.figure(figsize=(15, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle("VR Wrist Live  ·  orange = LEFT   blue = RIGHT",
                 fontsize=11, fontweight="bold", color="white", y=0.995)

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        left=0.07, right=0.97, top=0.97, bottom=0.05,
        hspace=0.70, wspace=0.38,
        height_ratios=[0.4, 1.0, 1.0, 1.0],
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
    ax_info.text(0.5, 0.5,
                 "POSITION (m)  ·  ORIENTATION (°)  ·  QUATERNION\n"
                 "solid = LEFT  ·  dashed = RIGHT\n"
                 "qx  qy  qz  qw",
                 transform=ax_info.transAxes, ha="center", va="center",
                 fontsize=8, color="#AAAAAA", family="monospace")

    # ── Row 1: position (auto-scale, position data varies widely) ─────────────
    pos_cfg = [
        (0, "x", "X  lateral  (m)"),
        (1, "y", "Y  up / down  (m)"),
        (2, "z", "Z  fwd / back  (m)"),
    ]
    pos_lines: dict[str, dict] = {}
    pos_axes = []
    for col, key, ylabel in pos_cfg:
        ax = _styled_ax(fig, gs[1, col], ylabel, ref_lines=(0,))
        ax.set_title(f"POSITION — {ylabel}", fontsize=7.5, color="#CCCCCC", pad=3)
        pos_axes.append((ax, key))
        for side in ("left", "right"):
            ls = "-" if side == "left" else "--"
            line, = ax.plot([], [], color=C[side], lw=1.5, alpha=0.9,
                            linestyle=ls, label=side.title())
            pos_lines.setdefault(side, {})[col] = line
        ax.legend(fontsize=6, loc="upper right",
                  facecolor="#222222", labelcolor="white", framealpha=0.7)

    # ── Row 2: orientation — fixed Y limits (no wrap-around blowup) ───────────
    ori_cfg = [
        (0, "roll",  "Roll  (°)",  (-180, 0, 180), (-185, 185)),
        (1, "pitch", "Pitch  (°)", (-90,  0,  90),  (-95,  95)),
        (2, "yaw",   "Yaw  (°)",   (-180, -90, 0, 90, 180), (-185, 185)),
    ]
    ori_lines: dict[str, dict] = {}
    ori_axes = []
    for col, key, ylabel, refs, ylim in ori_cfg:
        ax = _styled_ax(fig, gs[2, col], ylabel, ref_lines=refs, ylim=ylim)
        ax.set_title(f"ORIENTATION — {ylabel}", fontsize=7.5, color="#CCCCCC", pad=3)
        ori_axes.append((ax, key))
        for side in ("left", "right"):
            ls = "-" if side == "left" else "--"
            line, = ax.plot([], [], color=C[side], lw=1.5, alpha=0.9,
                            linestyle=ls, label=side.title())
            ori_lines.setdefault(side, {})[col] = line
        ax.legend(fontsize=6, loc="upper right",
                  facecolor="#222222", labelcolor="white", framealpha=0.7)

    # ── Row 3: quaternion components — fixed Y limits [-1.1, 1.1] ─────────────
    # Layout: qx | qy | qz + qw
    # solid line = LEFT arm, dashed = RIGHT arm
    # colour encodes component (shared legend on qz+qw plot)
    quat_lines: dict[str, dict[str, dict]] = {"left": {}, "right": {}}

    # qx
    ax_qx = _styled_ax(fig, gs[3, 0], "qx", ref_lines=(0,), ylim=(-1.15, 1.15))
    ax_qx.set_title("QUATERNION — qx", fontsize=7.5, color="#CCCCCC", pad=3)
    for side in ("left", "right"):
        ls = "-" if side == "left" else "--"
        line, = ax_qx.plot([], [], color=QUAT_COLORS["qx"], lw=1.5,
                           linestyle=ls, alpha=0.9, label=side.title())
        quat_lines[side]["qx"] = line
    ax_qx.legend(fontsize=6, loc="upper right",
                 facecolor="#222222", labelcolor="white", framealpha=0.7)

    # qy
    ax_qy = _styled_ax(fig, gs[3, 1], "qy", ref_lines=(0,), ylim=(-1.15, 1.15))
    ax_qy.set_title("QUATERNION — qy", fontsize=7.5, color="#CCCCCC", pad=3)
    for side in ("left", "right"):
        ls = "-" if side == "left" else "--"
        line, = ax_qy.plot([], [], color=QUAT_COLORS["qy"], lw=1.5,
                           linestyle=ls, alpha=0.9, label=side.title())
        quat_lines[side]["qy"] = line
    ax_qy.legend(fontsize=6, loc="upper right",
                 facecolor="#222222", labelcolor="white", framealpha=0.7)

    # qz + qw on the same axes (4 lines total)
    ax_qzw = _styled_ax(fig, gs[3, 2], "qz / qw", ref_lines=(0,), ylim=(-1.15, 1.15))
    ax_qzw.set_title("QUATERNION — qz / qw", fontsize=7.5, color="#CCCCCC", pad=3)
    for comp in ("qz", "qw"):
        for side in ("left", "right"):
            ls = "-" if side == "left" else "--"
            line, = ax_qzw.plot([], [], color=QUAT_COLORS[comp], lw=1.5,
                                linestyle=ls, alpha=0.9,
                                label=f"{comp} {side[0].upper()}")
            quat_lines[side][comp] = line
    ax_qzw.legend(fontsize=6, loc="upper right",
                  facecolor="#222222", labelcolor="white", framealpha=0.7)

    quat_axes = [
        (ax_qx,  ["qx"]),
        (ax_qy,  ["qy"]),
        (ax_qzw, ["qz", "qw"]),
    ]

    # ── Animation update ──────────────────────────────────────────────────────

    def _update(_frame):
        now_t = 0.0
        for side in ("left", "right"):
            lat = buf.latest(side)
            if lat:
                now_t = max(now_t, lat["t"])
        t_lo = max(0.0, now_t - window_s)
        x_lim = (t_lo, max(t_lo + window_s, now_t + 0.5))

        for side in ("left", "right"):
            d = buf.arrays(side, KEYS)
            t = d["t"]

            if t.size == 0:
                for col, _ in enumerate(pos_axes):
                    pos_lines[side][col].set_data([], [])
                for col, _ in enumerate(ori_axes):
                    ori_lines[side][col].set_data([], [])
                for comp in ("qx", "qy", "qz", "qw"):
                    quat_lines[side][comp].set_data([], [])
                continue

            mask = t >= t_lo

            for col, (_, key) in enumerate(pos_axes):
                pos_lines[side][col].set_data(t[mask], d[key][mask])
            for col, (_, key) in enumerate(ori_axes):
                ori_lines[side][col].set_data(t[mask], d[key][mask])
            for comp in ("qx", "qy", "qz", "qw"):
                quat_lines[side][comp].set_data(t[mask], d[comp][mask])

            lat = buf.latest(side)
            if lat:
                status_text[side].set_text(
                    f"{side.upper()}\n"
                    f"X {lat['x']:+.3f}   Y {lat['y']:+.3f}   Z {lat['z']:+.3f} m\n"
                    f"Roll {lat['roll']:+.1f}°  Pitch {lat['pitch']:+.1f}°  Yaw {lat['yaw']:+.1f}°\n"
                    f"qx {lat['qx']:+.3f}  qy {lat['qy']:+.3f}  "
                    f"qz {lat['qz']:+.3f}  qw {lat['qw']:+.3f}"
                )

        # position: auto-scale (position range varies)
        for col, (ax, _) in enumerate(pos_axes):
            ax.set_xlim(*x_lim)
            all_v = [pos_lines[s][col].get_ydata() for s in ("left", "right")]
            all_v = [v for v in all_v if len(v)]
            if all_v:
                ys  = np.concatenate(all_v)
                pad = max(0.05, (ys.max() - ys.min()) * 0.12)
                ax.set_ylim(ys.min() - pad, ys.max() + pad)

        # orientation and quaternion: fixed Y, only update X
        for col, (ax, _) in enumerate(ori_axes):
            ax.set_xlim(*x_lim)
        for ax, _ in quat_axes:
            ax.set_xlim(*x_lim)

    ani = FuncAnimation(fig, _update, interval=1000.0 / rate_hz,
                        blit=False, cache_frame_data=False)
    return fig, ani


def main():
    parser = argparse.ArgumentParser(
        prog="wrist_plot",
        description="Live wrist dashboard — position, orientation, and quaternion",
    )
    parser.add_argument("--window", type=float, default=10.0,
                        help="Scrolling time window in seconds (default: 10)")
    parser.add_argument("--rate",   type=float, default=20.0,
                        help="Plot refresh rate in Hz (default: 20)")
    args = parser.parse_args()

    maxlen = int(args.window * 150)
    buf    = WristBuffer(maxlen=maxlen)

    print("[wrist_plot] Listening on IPC bus port 5555 — move your wrists.")
    print("             Close the window or press Ctrl-C to exit.\n")

    try:
        fig, ani = build_dashboard(buf, args.window, args.rate)
        plt.show()
    except KeyboardInterrupt:
        print("\n[wrist_plot] Stopped.")
    finally:
        buf.close()


if __name__ == "__main__":
    main()
