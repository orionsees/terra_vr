#!/usr/bin/env python3
"""check_torque.py — snapshot torque status for both arms.

Reads one feedback frame from each arm bridge and reports whether each
joint appears to have torque enabled, based on the load/effort value.

The STS3215 reports near-zero load when torque is disabled (servo not
driving).  Joints with |effort| < EFFORT_THRESHOLD are flagged as OFF.

Usage:
    python3 check_torque.py          # check both arms once
    python3 check_torque.py --watch  # refresh every 1 s (Ctrl-C to stop)
"""

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bus import LEFT_ARM_FEEDBACK_PORT, RIGHT_ARM_FEEDBACK_PORT

EFFORT_OFF_THRESHOLD      = 1.0   # % below this → torque likely OFF
EFFORT_OVERLOAD_THRESHOLD = 90.0  # % above this → overload protection risk


def _read_one(port: int, timeout: float = 2.0) -> dict | None:
    """Open a raw TCP connection to the feedback publisher and read one frame."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.settimeout(timeout)
    try:
        s.connect(("127.0.0.1", port))
        buf = b""
        while True:
            chunk = s.recv(8192)
            if not chunk:
                return None
            buf += chunk
            if b"\n" in buf:
                line, _ = buf.split(b"\n", 1)
                if line:
                    return json.loads(line)
    except (socket.timeout, OSError, json.JSONDecodeError):
        return None
    finally:
        s.close()


def _print_arm(label: str, msg: dict) -> bool:
    """Print per-joint torque status. Returns True if all joints appear ON."""
    names    = msg.get("names", [])
    efforts  = msg.get("efforts", [])
    positions = msg.get("positions", [])

    all_on = True
    rows = []
    for name, eff, pos in zip(names, efforts, positions):
        if math.isnan(eff):
            state   = "OFF  (no reply)"
            flag    = " !"
            all_on  = False
        elif abs(eff) < EFFORT_OFF_THRESHOLD:
            at_zero = abs(pos) < 2.0   # within 2° of target → probably fine
            if at_zero:
                state = f"OK   (at target, {eff:+.1f} %)"
                flag  = ""
            else:
                state  = f"OFF  (zero load at {pos:.1f}°)"
                flag   = " !"
                all_on = False
        elif abs(eff) > EFFORT_OVERLOAD_THRESHOLD:
            state  = f"RISK (overload {eff:+.1f} %)"
            flag   = " !"
            all_on = False
        else:
            state = f"ON   ({eff:+.1f} %)"
            flag  = ""

        rows.append((name, pos, state, flag))

    print(f"\n  {label}")
    print(f"  {'Joint':<16} {'Pos (°)':>8}   {'Torque'}")
    print(f"  {'-'*16} {'-'*8}   {'-'*30}")
    for name, pos, state, flag in rows:
        print(f"  {name:<16} {pos:>8.2f}   {state}{flag}")

    return all_on


def check_once() -> None:
    ok = True
    for label, port in [
        ("Left arm  (port 5558)", LEFT_ARM_FEEDBACK_PORT),
        ("Right arm (port 5559)", RIGHT_ARM_FEEDBACK_PORT),
    ]:
        msg = _read_one(port)
        if msg is None:
            print(f"\n  {label}: no data — is the bridge running?")
            ok = False
        else:
            arm_ok = _print_arm(label, msg)
            ok = ok and arm_ok

    if ok:
        print("\n  All joints OK.")
    else:
        print("\n  ! = joint needs attention (torque off or overload risk).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true",
                        help="Refresh every second until Ctrl-C")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")   # clear terminal
                check_once()
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    else:
        check_once()


if __name__ == "__main__":
    main()
