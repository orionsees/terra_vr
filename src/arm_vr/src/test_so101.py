"""test_so101.py — Interactive test script for the SO101Arm library.

Usage
-----
    python test_so101.py                        # defaults to /dev/ttyACM0
    python test_so101.py --port /dev/ttyUSB0
    python test_so101.py --port COM3 --cal path/to/cal.json

Each section can be run individually via the numbered menu, or you can
run all non-destructive tests at once with option [A].
"""

import argparse
import sys
import time

# Make sure we can import so101 from the same directory
sys.path.insert(0, __file__.rsplit("/", 1)[0])

from so101 import (
    SO101Arm,
    MotorCalibration,
    load_calibration,
    raw_to_degrees,
    degrees_to_raw,
    DEFAULT_ID_TO_NAME,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "-" * 60


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def fail(msg: str) -> None:
    print(f"  [!!]  {msg}")


def info(msg: str) -> None:
    print(f"  ...   {msg}")


# ---------------------------------------------------------------------------
# Individual test routines
# ---------------------------------------------------------------------------

def test_scan(arm: SO101Arm) -> list[int]:
    """Ping all servo IDs 1-6 and report which ones respond."""
    section("SCAN — discover responding servos")
    responding = arm.scan()
    if responding:
        for sid in responding:
            name = arm.joint_names.get(sid, f"id_{sid}")
            ok(f"Servo {sid} ({name}) is alive")
    else:
        fail("No servos responded — check cable and power")
    return responding


def test_ping(arm: SO101Arm, servo_ids: list[int]) -> None:
    """Ping each servo individually."""
    section("PING — individual servo ping")
    for sid in servo_ids:
        result = arm.ping(sid)
        name = arm.joint_names.get(sid, f"id_{sid}")
        if result:
            ok(f"ping({sid})  [{name}]  → True")
        else:
            fail(f"ping({sid})  [{name}]  → no reply")


def test_read_positions_raw(arm: SO101Arm, servo_ids: list[int]) -> None:
    """Read raw encoder ticks from all responding servos."""
    section("READ POSITIONS RAW — encoder ticks")
    raw_map = arm.read_positions_raw(servo_ids)
    for sid, raw in raw_map.items():
        name = arm.joint_names.get(sid, f"id_{sid}")
        if raw is not None:
            ok(f"Servo {sid} ({name})  raw={raw}")
        else:
            fail(f"Servo {sid} ({name})  no reply")


def test_read_loads(arm: SO101Arm, servo_ids: list[int]) -> None:
    """Read raw load/torque values."""
    section("READ LOADS — raw torque values")
    load_map = arm.read_loads_raw(servo_ids)
    for sid, load in load_map.items():
        name = arm.joint_names.get(sid, f"id_{sid}")
        if load is not None:
            ok(f"Servo {sid} ({name})  load={load}")
        else:
            fail(f"Servo {sid} ({name})  no reply")


def test_read_positions(arm: SO101Arm) -> None:
    """Read positions — degrees if calibrated, raw ticks otherwise."""
    section("READ POSITIONS — degrees (or raw if uncalibrated)")
    positions = arm.read_positions()
    for key, value in positions.items():
        if value is not None:
            ok(f"{key} = {value}")
        else:
            fail(f"{key} = None (no reply)")


def test_calibration_helpers() -> None:
    """Test the standalone calibration conversion helpers."""
    section("CALIBRATION HELPERS — raw_to_degrees / degrees_to_raw")
    cal = MotorCalibration(
        id=1,
        drive_mode=0,
        homing_offset=0,
        range_min=1024,
        range_max=3072,
    )
    mid = (cal.range_min + cal.range_max) / 2  # 2048
    test_cases = [
        (mid, 0.0),
        (cal.range_min, raw_to_degrees(cal.range_min, cal)),
        (cal.range_max, raw_to_degrees(cal.range_max, cal)),
    ]
    for raw_in, expected_deg in test_cases:
        deg = raw_to_degrees(int(raw_in), cal)
        raw_back = degrees_to_raw(deg, cal)
        ok(f"raw={int(raw_in)} → {deg:.3f}° → raw_back={raw_back}")

    info(f"DEFAULT_ID_TO_NAME = {DEFAULT_ID_TO_NAME}")
    ok("Calibration helper tests passed (no hardware required)")


def test_joint_names_and_calibrated_ids(arm: SO101Arm) -> None:
    """Inspect the joint name / calibration metadata."""
    section("ARM METADATA — joint names & calibrated IDs")
    info(f"joint_names     = {arm.joint_names}")
    info(f"calibrated_ids  = {arm.calibrated_ids}")
    if arm.calibrated_ids:
        ok("Calibration data is loaded")
    else:
        info("No calibration loaded — positions will be in raw ticks")


def test_set_position_raw(arm: SO101Arm, servo_id: int, target_raw: int) -> None:
    """Write a raw encoder target to one servo (with ACK)."""
    section(f"SET POSITION RAW — servo {servo_id} → raw={target_raw}")
    name = arm.joint_names.get(servo_id, f"id_{servo_id}")
    info(f"Reading current position of servo {servo_id} ({name}) ...")
    before = arm.read_position_raw(servo_id)
    info(f"Before: raw={before}")

    info(f"Sending goal position raw={target_raw} ...")
    success = arm.set_position_raw(servo_id, target_raw, acc=10, speed=100, goal_time=0)
    if success:
        ok(f"ACK received for servo {servo_id}")
    else:
        fail(f"No ACK from servo {servo_id}")

    time.sleep(1.0)
    after = arm.read_position_raw(servo_id)
    ok(f"After:  raw={after}  (target was {target_raw})")


def test_set_positions_raw(arm: SO101Arm, targets: dict[int, int]) -> None:
    """Write raw positions to multiple servos at once."""
    section("SET POSITIONS RAW — multiple servos")
    info(f"Targets: {targets}")
    results = arm.set_positions_raw(targets, acc=10, speed=100, settle_seconds=1.0, expect_ack=True)
    for sid, success in results.items():
        name = arm.joint_names.get(sid, f"id_{sid}")
        if success:
            ok(f"Servo {sid} ({name}) accepted goal")
        else:
            fail(f"Servo {sid} ({name}) did not ACK")
    positions_after = arm.read_positions_raw(list(targets.keys()))
    for sid, raw in positions_after.items():
        name = arm.joint_names.get(sid, f"id_{sid}")
        info(f"  final position servo {sid} ({name}): raw={raw}")


def test_set_position_degrees(arm: SO101Arm, joint: str, degrees: float) -> None:
    """Move one joint by name in degrees (requires calibration)."""
    section(f"SET POSITION (DEGREES) — joint='{joint}' target={degrees}°")
    if not arm.calibrated_ids:
        fail("No calibration loaded — skipping degree-based move")
        return
    try:
        success = arm.set_position(joint, degrees, acc=10, speed=100, goal_time=0)
        if success:
            ok(f"ACK received for '{joint}'")
        else:
            fail(f"No ACK for '{joint}'")
        time.sleep(1.0)
        positions = arm.read_positions()
        key = f"{joint}.pos"
        info(f"Position after move: {key} = {positions.get(key)}")
    except (KeyError, ValueError) as exc:
        fail(str(exc))


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SO101Arm library test script")
    p.add_argument("--port", default="/dev/ttyACM0", help="Serial port (default: /dev/ttyACM0)")
    p.add_argument("--cal", default="/home/ubunt2/Documents/arm_vr/config/right_arm.json", help="Path to calibration JSON file (optional)")
    p.add_argument("--baud", type=int, default=1_000_000, help="Baudrate (default: 1000000)")
    return p


MENU = """\

  SO101Arm Test Menu
  ==================
  [1]  Scan for servos
  [2]  Ping individual servos (1-6)
  [3]  Read raw positions (encoder ticks)
  [4]  Read raw loads (torque)
  [5]  Read positions (degrees if calibrated)
  [6]  Inspect joint names & calibrated IDs
  [7]  Test calibration math helpers (no HW)
  [8]  Set position RAW — one servo (MOVES SERVO)
  [9]  Set positions RAW — all servos to mid (MOVES SERVOS)
  [D]  Set position in DEGREES — shoulder_pan → 0° (MOVES SERVO)
  [A]  Run all read-only tests (1-7)
  [Q]  Quit
"""


def run_menu(arm: SO101Arm) -> None:
    responding: list[int] = []

    while True:
        print(MENU)
        choice = input("  Choice: ").strip().upper()

        if choice == "Q":
            print("  Bye!")
            break

        elif choice == "1":
            responding = test_scan(arm)

        elif choice == "2":
            ids = responding or list(range(1, 7))
            test_ping(arm, ids)

        elif choice == "3":
            ids = responding or list(range(1, 7))
            test_read_positions_raw(arm, ids)

        elif choice == "4":
            ids = responding or list(range(1, 7))
            test_read_loads(arm, ids)

        elif choice == "5":
            test_read_positions(arm)

        elif choice == "6":
            test_joint_names_and_calibrated_ids(arm)

        elif choice == "7":
            test_calibration_helpers()

        elif choice == "8":
            if not responding:
                responding = arm.scan()
            if not responding:
                fail("No responding servos found")
                continue
            sid = responding[0]
            cur = arm.read_position_raw(sid)
            target = (cur or 2048) + 100
            print(f"  Will move servo {sid} by +100 ticks (to ~{target})")
            confirm = input("  Confirm? [y/N]: ").strip().lower()
            if confirm == "y":
                test_set_position_raw(arm, sid, target)

        elif choice == "9":
            if not responding:
                responding = arm.scan()
            if not responding:
                fail("No responding servos found")
                continue
            targets = {sid: 2048 for sid in responding}
            print(f"  Will move all servos to raw=2048 (mid): {targets}")
            confirm = input("  Confirm? [y/N]: ").strip().lower()
            if confirm == "y":
                test_set_positions_raw(arm, targets)

        elif choice == "D":
            test_set_position_degrees(arm, "shoulder_pan", 0.0)

        else:
            print("  Unknown option — try again")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    print(f"\nConnecting to SO101Arm on {args.port} @ {args.baud} baud ...")
    print(f"Calibration: {args.cal or '(none)'}")

    try:
        arm_kwargs: dict = dict(port=args.port, baudrate=args.baud)
        if args.cal:
            arm_kwargs["calibration_file"] = args.cal

        with SO101Arm(**arm_kwargs) as arm:
            print("  Serial port opened successfully.")
            run_menu(arm)

    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
