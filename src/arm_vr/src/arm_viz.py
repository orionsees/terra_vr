#!/usr/bin/env python3
"""Standalone matplotlib visualizer for the 2-DOF arm (shoulder_lift + elbow_flex).

Shows the 2-link arm starting fully extended along the x-axis.
No ROS subscription — runs standalone.

    python3 arm_viz.py
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("TkAgg")   # change to "Qt5Agg" if TkAgg is not available
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── Same IK constants as teleoperation.py ────────────────────────────────────
ARM_L1 = 0.116   # shoulder_lift → elbow_flex  (m)
ARM_L2 = 0.135   # elbow_flex   → wrist_flex   (m)

ARM_NEUTRAL_FORWARD = ARM_L1 + ARM_L2 - 0.005   # nearly fully extended along x-axis
ARM_NEUTRAL_HEIGHT  = 0.0
ARM_POSITION_SCALE  = 1.5

TRACE_LEN = 120   # number of past end-effector positions to show


def ik(forward: float, height: float) -> tuple[float, float] | None:
    """Return (theta_shoulder, theta_elbow) in radians, or None if unreachable."""
    d_sq = forward ** 2 + height ** 2
    d = math.sqrt(d_sq) if d_sq > 1e-9 else 1e-9
    max_reach = ARM_L1 + ARM_L2 - 1e-4
    min_reach = abs(ARM_L1 - ARM_L2) + 1e-4
    d = max(min_reach, min(max_reach, d))
    # rescale forward/height to lie on the clamped circle
    scale = d / math.sqrt(forward ** 2 + height ** 2) if (forward ** 2 + height ** 2) > 1e-9 else 1.0
    forward *= scale
    height  *= scale

    cos_e = (d ** 2 - ARM_L1 ** 2 - ARM_L2 ** 2) / (2.0 * ARM_L1 * ARM_L2)
    cos_e = max(-1.0, min(1.0, cos_e))
    theta_e = -math.acos(cos_e)
    alpha   = math.atan2(height, forward)
    beta    = math.atan2(ARM_L2 * math.sin(theta_e), ARM_L1 + ARM_L2 * math.cos(theta_e))
    theta_s = alpha - beta
    return theta_s, theta_e


def fk(theta_s: float, theta_e: float) -> tuple[tuple, tuple, tuple]:
    """Forward kinematics: returns (origin, elbow, end_effector) as (x, y) tuples."""
    origin = (0.0, 0.0)
    elbow  = (ARM_L1 * math.cos(theta_s),
              ARM_L1 * math.sin(theta_s))
    end    = (elbow[0] + ARM_L2 * math.cos(theta_s + theta_e),
              elbow[1] + ARM_L2 * math.sin(theta_s + theta_e))
    return origin, elbow, end


class ArmState:
    """Simple state container — no ROS, updated directly."""
    def __init__(self):
        self.arm_fwd    = ARM_NEUTRAL_FORWARD
        self.arm_height = ARM_NEUTRAL_HEIGHT
        self.trace_x: list[float] = []
        self.trace_y: list[float] = []

    def get_state(self):
        return self.arm_fwd, self.arm_height, list(self.trace_x), list(self.trace_y)


def main() -> None:
    state = ArmState()

    # ── Matplotlib setup ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    reach = ARM_L1 + ARM_L2 + 0.02

    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_xlabel("Forward (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("2-DOF Arm — live IK")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)

    # Reachable workspace circle (annulus)
    max_c = plt.Circle((0, 0), ARM_L1 + ARM_L2, color="lightblue", fill=True, alpha=0.15, label="reachable")
    min_c = plt.Circle((0, 0), abs(ARM_L1 - ARM_L2), color="white", fill=True)
    ax.add_patch(max_c)
    ax.add_patch(min_c)

    # Arm links
    link1_line, = ax.plot([], [], "o-", color="steelblue", linewidth=4, markersize=8, label="L1")
    link2_line, = ax.plot([], [], "o-", color="darkorange", linewidth=4, markersize=8, label="L2")

    # End-effector target marker
    target_dot, = ax.plot([], [], "r*", markersize=14, label="target")

    # Trace
    trace_line, = ax.plot([], [], "-", color="red", linewidth=1, alpha=0.5, label="trace")

    # Info text
    info_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                        verticalalignment="top", fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.legend(loc="lower right", fontsize=8)

    def update(_frame):
        fwd, h, tx, ty = state.get_state()

        target_dot.set_data([fwd], [h])
        trace_line.set_data(tx, ty)

        result = ik(fwd, h)
        if result:
            theta_s, theta_e = result
            origin, elbow, end = fk(theta_s, theta_e)
            link1_line.set_data([origin[0], elbow[0]], [origin[1], elbow[1]])
            link2_line.set_data([elbow[0],  end[0]],  [elbow[1],  end[1]])
            info_text.set_text(
                f"target  fwd={fwd:.3f} m  h={h:.3f} m\n"
                f"θ_shoulder={math.degrees(theta_s):+.1f}°\n"
                f"θ_elbow   ={math.degrees(theta_e):+.1f}°\n"
                f"EE  ({end[0]:.3f}, {end[1]:.3f}) m"
            )
        else:
            info_text.set_text("IK: unreachable")

        return link1_line, link2_line, target_dot, trace_line, info_text

    ani = FuncAnimation(fig, update, interval=50, blit=True)  # 20 Hz refresh

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
