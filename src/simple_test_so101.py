"""test_so101.py — Quick SO101Arm control command reference.

Usage:
    python test_so101.py                    # right arm default
    python test_so101.py --port /dev/ttyUSB0 --cal config/left_arm.json
"""

import sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])

from so101 import SO101Arm

PORT = "/dev/ttyACM0"
CAL  = "/home/ubunt2/ros_ws/src/arm_vr/config/right_arm.json"

with SO101Arm(PORT, calibration_file=CAL) as arm:

    # --- discover ---
    print("scan()         :", arm.scan())               # list of responding servo IDs
    print("ping(1)        :", arm.ping(1))               # True / False

    # --- read ---
    print("positions (deg):", arm.read_positions())      # {name.pos: degrees, ...}
    print("raw ticks      :", arm.read_positions_raw([1, 2, 3, 4, 5, 6]))
    print("loads          :", arm.read_loads_raw([1, 2, 3, 4, 5, 6]))

    # --- write by joint name (degrees, needs calibration) ---
    arm.set_position("shoulder_pan",  90, acc=10, speed=100)
    # arm.set_positions({"elbow_flex": -20.0, "wrist_flex": 10.0}, acc=10, speed=100)

    # --- write by servo ID (raw encoder ticks 0-4095, mid=2048) ---
    # arm.set_position_raw(3, 3069, acc=10, speed=100)
    # arm.set_positions_raw({1: 2048, 2: 2048, 3: 2048}, acc=10, speed=100, settle_seconds=1.0)

    # --- joint name → servo ID mapping ---
    # shoulder_pan=1  shoulder_lift=2  elbow_flex=3
    # wrist_flex=4    wrist_roll=5     gripper=6
