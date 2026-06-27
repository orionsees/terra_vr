#!/usr/bin/env python3
"""Interactive setup and launch script for SO101 dual-arm VR teleoperation.

Detects arm ports, optionally calibrates each arm with lerobot, then launches
the full pipeline (arm bridges, teleoperation bridge) using the IPC bus (bus.py)
for inter-process communication — no ROS2 required.

VR data is parsed in-process using the hand-tracking-sdk (pip install hand-tracking-sdk)
and published to the IPC bus on port 5555.

IPC bus port map (all localhost TCP):
  5555 – VR data        (this process → teleop_bridge_node.py)
  5556 – Left arm cmds  (teleop/zero_arms → so101_bridge_left.py)
  5557 – Right arm cmds (teleop/zero_arms → so101_bridge_right.py)
  5558 – Left arm feedback  (so101_bridge_left.py → consumers)
  5559 – Right arm feedback (so101_bridge_right.py → consumers)

Usage:
    python3 src/setup_and_launch.py
"""

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from bus import Publisher, VR_DATA_PORT

from hand_tracking_sdk import (
    HTSClient,
    HTSClientConfig,
    JointName,
    HandSide,
    StreamOutput,
    TransportMode,
    convert_hand_frame_unity_left_to_right,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_DIR = SCRIPT_DIR.parent / "config"
PORTS_FILE = CONFIG_DIR / "ports.json"


# ---------------------------------------------------------------------------
# VR data publisher (runs as a daemon thread in this process)
# ---------------------------------------------------------------------------

class _VRPublisher:
    """Receives HTS frames from the Quest and publishes them to IPC bus port 5555.

    Topics published:
        left_wrist / right_wrist  → x, y, z, qx, qy, qz, qw, timestamp
        left_pinch / right_pinch  → distance_cm, timestamp
        head_pose                 → x, y, z, qx, qy, qz, qw, timestamp
    """

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._pub  = Publisher(VR_DATA_PORT)
        self._stop = threading.Event()

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="hts-client"
        )
        self._thread.start()
        print(f"[VR] HTS listener starting on {host}:{port} (TCP)")

    def _run(self) -> None:
        try:
            client = HTSClient(
                HTSClientConfig(
                    transport_mode=TransportMode.TCP_SERVER,
                    host=self._host,
                    port=self._port,
                    timeout_s=1.0,
                    output=StreamOutput.FRAMES,
                    include_wall_time=True,
                )
            )
            for frame in client.iter_events():
                if self._stop.is_set():
                    break
                try:
                    self._process_frame(frame)
                except Exception as exc:
                    print(f"[VR] frame error: {exc}")
        except Exception as exc:
            print(f"[VR] HTS client error: {exc}")

    def _process_frame(self, frame) -> None:
        if frame.recv_time_unix_ns is not None:
            ts = frame.recv_time_unix_ns / 1_000_000_000.0
        else:
            ts = time.time()

        # Convert from Unity left-handed to right-handed coordinate system.
        # Without this the right wrist quaternion is mirrored and its RPY values
        # appear incorrect compared to the left wrist.
        frame = convert_hand_frame_unity_left_to_right(frame)

        if frame.side == HandSide.HEAD:
            h = frame.head
            self._pub.publish({
                "topic": "head_pose",
                "data": {
                    "x": h.x, "y": h.y, "z": h.z,
                    "qx": h.qx, "qy": h.qy, "qz": h.qz, "qw": h.qw,
                },
                "timestamp": ts,
            })
            return

        side_str = "left" if frame.side == HandSide.LEFT else "right"
        w = frame.wrist
        self._pub.publish({
            "topic": f"{side_str}_wrist",
            "data": {
                "x": w.x, "y": w.y, "z": w.z,
                "qx": w.qx, "qy": w.qy, "qz": w.qz, "qw": w.qw,
            },
            "timestamp": ts,
        })

        try:
            tx, ty, tz = frame.get_joint(JointName.THUMB_TIP)
            ix, iy, iz = frame.get_joint(JointName.INDEX_TIP)
            dist_cm = math.sqrt(
                (tx - ix) ** 2 + (ty - iy) ** 2 + (tz - iz) ** 2
            ) * 100.0
            self._pub.publish({
                "topic": f"{side_str}_pinch",
                "data": {"distance_cm": dist_cm},
                "timestamp": ts,
            })
        except Exception:
            pass

    def close(self) -> None:
        self._stop.set()
        self._pub.close()


# ---------------------------------------------------------------------------
# Port detection
# ---------------------------------------------------------------------------

def _list_serial_ports() -> set[Path]:
    return set(Path("/dev").glob("ttyACM*")) | set(Path("/dev").glob("ttyUSB*"))


def _wait_for_new_port(before: set[Path], timeout_s: float = 60.0) -> Optional[str]:
    """Poll until a new serial port appears; return its path or None on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        new = _list_serial_ports() - before
        if new:
            return str(sorted(new)[0])
        time.sleep(0.4)
    return None


def _detect_arm_port(arm_label: str) -> Optional[str]:
    """Prompt the user to connect an arm, then auto-detect the new port."""
    before = _list_serial_ports()
    existing = sorted(str(p) for p in before)
    if existing:
        print(f"  Ports already connected: {', '.join(existing)}")

    input(f"  Connect the {arm_label} USB cable, then press ENTER ...")
    print("  Waiting for port ...", end="", flush=True)

    port = _wait_for_new_port(before)
    if port:
        print(f" found {port}")
        return port

    print(" timed out.")
    manual = input(f"  Enter {arm_label} port manually (e.g. /dev/ttyACM0): ").strip()
    return manual or None


# ---------------------------------------------------------------------------
# Bridge process management
# ---------------------------------------------------------------------------

def _kill_bridges() -> None:
    """Terminate any running arm bridge processes and wait for ports to free."""
    result = subprocess.run(["pkill", "-f", "so101_bridge_"], capture_output=True)
    if result.returncode == 0:
        print("  Stopped existing bridge processes.")
        time.sleep(1.0)  # wait for serial ports to be released


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _run_calibration(port: str, arm_id: str) -> bool:
    """Run lerobot-calibrate for the given arm.

    Saves calibration to CONFIG_DIR/<arm_id>.json (e.g. config/left_arm.json)
    via --robot.calibration_dir so the bridge scripts can find it.
    Returns True if the process exited successfully.
    """
    cmd = [
        "lerobot-calibrate",
        "--robot.type=so101_follower",
        f"--robot.port={port}",
        f"--robot.id={arm_id}",
        f"--robot.calibration_dir={CONFIG_DIR}",
    ]
    print(f"\n  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Per-arm setup
# ---------------------------------------------------------------------------

def _setup_arm(arm_label: str, arm_id: str, saved_port: Optional[str]) -> str:
    """Guide the user through port detection and optional calibration for one arm.

    Returns the confirmed port string.
    """
    _banner(f"Setup: {arm_label}")

    if saved_port:
        print(f"  Previously saved port: {saved_port}")
        reuse = input("  Use this port? [Y/n]: ").strip().lower()
        if reuse != "n":
            port = saved_port
        else:
            port = _detect_arm_port(arm_label)
    else:
        port = _detect_arm_port(arm_label)

    if not port:
        sys.exit(f"[ERROR] Could not determine {arm_label} port. Aborting.")

    print(f"  {arm_label} port: {port}")

    cal_file = CONFIG_DIR / f"{arm_id}.json"
    if cal_file.exists():
        print(f"  Existing calibration: {cal_file}")
        answer = input("  Re-calibrate? [y/N]: ").strip().lower()
    else:
        print(f"  No calibration file found at {cal_file}")
        answer = input("  Run calibration now? [Y/n]: ").strip().lower()
        if answer != "n":
            answer = "y"

    if answer == "y":
        print(f"\n  Starting lerobot calibration for {arm_label} ...")
        _kill_bridges()  # release serial ports before calibration opens them
        ports_before = _list_serial_ports()
        ok = _run_calibration(port, arm_id)
        if ok:
            print(f"  Calibration saved to {cal_file}")
        else:
            print("  [WARNING] Calibration exited with an error.")
            proceed = input("  Continue with launch anyway? [y/N]: ").strip().lower()
            if proceed != "y":
                sys.exit("Aborting due to calibration failure.")

        # lerobot-calibrate can cause USB re-enumeration — the arm may come
        # back on a different port (e.g. /dev/ttyACM1 → /dev/ttyACM2).
        time.sleep(1.0)  # give the kernel time to re-create the device node
        if not Path(port).exists():
            print(f"  Port {port} gone after calibration — waiting for re-enumeration ...")
            new_port = _wait_for_new_port(ports_before - {Path(port)}, timeout_s=15.0)
            if new_port:
                print(f"  Arm re-enumerated on {new_port}")
                port = new_port
            else:
                print(f"  [WARNING] Could not detect new port; keeping {port}")

    return port


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def _build_bridge_cmd(side: str, port: str) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / f"so101_bridge_{side}.py"),
        "--module-dir", str(SCRIPT_DIR),
        "--port", port,
        "--calibration-file", str(CONFIG_DIR / f"{side}_arm.json"),
        "--baudrate", "1000000",
        "--serial-timeout", "0.02",
        "--reply-timeout", "0.05",
        "--feedback-rate-hz", "20.0",
    ]


def _launch_all(
    left_port: str, right_port: str, tcp_host: str
) -> tuple[list[subprocess.Popen], list[str], _VRPublisher]:
    _kill_bridges()  # ensure no stale bridges are holding the serial ports

    teleop_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "teleop_bridge_node.py"),
        "--cal-right", str(CONFIG_DIR / "right_arm.json"),
        "--cal-left",  str(CONFIG_DIR / "left_arm.json"),
    ]

    procs: list[subprocess.Popen] = []
    labels: list[str] = []

    _banner("Launching processes")

    for side, port in [("left", left_port), ("right", right_port)]:
        cmd = _build_bridge_cmd(side, port)
        print(f"  [{side} arm bridge] {' '.join(cmd)}")
        procs.append(subprocess.Popen(cmd))
        labels.append(f"{side} arm bridge")

    vr_pub = _VRPublisher(host=tcp_host, port=8000)
    print(f"  [VR publisher] hand-tracking-sdk → IPC bus port {VR_DATA_PORT}")

    print("\n  Bridges and VR publisher are running.")
    input("  Position BOTH arms with shoulder_pan at 0° (centre), then press ENTER to start teleoperation... ")

    print(f"  [teleop bridge] {' '.join(teleop_cmd)}")
    procs.append(subprocess.Popen(teleop_cmd))
    labels.append("teleop bridge")

    return procs, labels, vr_pub


def _monitor(
    procs: list[subprocess.Popen], labels: list[str], vr_pub: _VRPublisher
) -> None:
    """Wait until all processes exit, forwarding SIGINT/SIGTERM to children."""

    def _shutdown(signum, frame):
        print("\nShutting down ...")
        vr_pub.close()
        for p in procs:
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("\nAll processes running. Press Ctrl+C to stop.\n")

    while True:
        for p, label in zip(procs, labels):
            if p.poll() is not None:
                print(f"[WARNING] '{label}' (pid {p.pid}) exited with code {p.returncode}")
        if all(p.poll() is not None for p in procs):
            break
        time.sleep(1.0)

    vr_pub.close()

    # Give processes a moment to clean up after SIGTERM
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _load_saved_ports() -> dict:
    if PORTS_FILE.exists():
        try:
            return json.loads(PORTS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_ports(left_port: str, right_port: str, tcp_host: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PORTS_FILE.write_text(json.dumps(
        {"left_arm_port": left_port, "right_arm_port": right_port, "tcp_host": tcp_host},
        indent=4,
    ))
    print(f"\n  Ports saved to {PORTS_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _banner("SO101 Dual-Arm + VR Teleoperation Setup")

    saved = _load_saved_ports()

    if saved:
        print(f"  Saved config: {saved}")
        do_setup = input("\n  Run full setup? [y/N]: ").strip().lower() == "y"
    else:
        print("  No saved config found — running full setup.")
        do_setup = True

    if not do_setup:
        # Quick launch using saved config
        left_port  = saved.get("left_arm_port")
        right_port = saved.get("right_arm_port")
        tcp_host   = saved.get("tcp_host", "0.0.0.0")

        if not left_port or not right_port:
            sys.exit("[ERROR] Saved config is missing port entries. Run full setup first.")

        print(f"\n  Left arm port : {left_port}")
        print(f"  Right arm port: {right_port}")
        print(f"  TCP host      : {tcp_host}")
    else:
        print("  This script will:")
        print("    1. Detect / confirm serial ports for each arm")
        print("    2. Optionally run lerobot calibration per arm")
        print("    3. Launch both arm bridges and start VR data publishing")

        # --- Left arm ---
        left_port = _setup_arm(
            arm_label="Left Arm",
            arm_id="left_arm",
            saved_port=saved.get("left_arm_port"),
        )

        # --- Right arm ---
        right_port = _setup_arm(
            arm_label="Right Arm",
            arm_id="right_arm",
            saved_port=saved.get("right_arm_port"),
        )

        # VR host
        default_host = saved.get("tcp_host", "0.0.0.0")
        tcp_host = input(f"\n  VR headset TCP host [{default_host}]: ").strip() or default_host

        # Persist for next run
        _save_ports(left_port, right_port, tcp_host)

    # Launch everything
    procs, labels, vr_pub = _launch_all(left_port, right_port, tcp_host)
    _monitor(procs, labels, vr_pub)


if __name__ == "__main__":
    main()
