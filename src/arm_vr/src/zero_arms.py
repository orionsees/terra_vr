#!/usr/bin/env python3
"""Zero Arms — set both SO101 arms to zero joint angles.

Sends an enable_torque command followed by a set_positions command (all
joints at 0.0°) to both arm bridges via the IPC bus (ports 5556 and 5557).
Requires the arm bridges to be running.

wrist_roll zero positions may differ between left and right arms due to
how each arm was physically positioned during calibration.  Adjust
LEFT_WRIST_ROLL_ZERO and RIGHT_WRIST_ROLL_ZERO if the wrist is not in
the correct neutral position after zeroing.

Usage:
    python3 zero_arms.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bus import CommandClient, LEFT_ARM_CMD_PORT, RIGHT_ARM_CMD_PORT

# Wrist-roll zero position in degrees for each arm.
# 0.0° maps to raw tick 2048 (calibration midpoint).
# If check_torque.py shows wrist_roll at >90 % load at this angle, rotate
# the wrist manually while running `check_torque.py --watch` until load
# drops to <20 %, then set that displayed angle here.
LEFT_WRIST_ROLL_ZERO  =  0.0   # degrees — tune if overload at zero
RIGHT_WRIST_ROLL_ZERO =  0.0   # degrees — tune if overload at zero

LEFT_ZERO_TARGETS = {
    "shoulder_pan":  0.0,
    "shoulder_lift": 0.0,
    "elbow_flex":    0.0,
    "wrist_flex":    0.0,
    "wrist_roll":    LEFT_WRIST_ROLL_ZERO,
    "gripper":       -90.0,
}

RIGHT_ZERO_TARGETS = {
    "shoulder_pan":  0.0,
    "shoulder_lift": 0.0,
    "elbow_flex":    0.0,
    "wrist_flex":    0.0,
    "wrist_roll":    RIGHT_WRIST_ROLL_ZERO,
    "gripper":       -90.0,
}

ENABLE_TORQUE_CMD = {"topic": "enable_torque"}

# Speed cap for zeroing (STS3215 units; 0 = max speed, ~300 ≈ slow/smooth).
# Lower = slower motion.  Raise if the arm is too sluggish.
ZERO_SPEED = 300

# Acceleration ramp (STS3215 units; 0 = instant, 20–50 = smooth ramp-up).
# Prevents shoulder joints from jerking at the start of the move.
ZERO_ACC = 20

# How long to wait after sending the position command.
# At ZERO_SPEED=300 the arm may take up to ~5 s for a full-range move.
ZERO_SETTLE_S = 6.0


TORQUE_HEARTBEAT_S = 1.0   # re-enable torque at this interval during the move


def main() -> int:
    errors = 0
    for side, port, targets in [
        ("left",  LEFT_ARM_CMD_PORT,  LEFT_ZERO_TARGETS),
        ("right", RIGHT_ARM_CMD_PORT, RIGHT_ZERO_TARGETS),
    ]:
        try:
            client = CommandClient(port, max_tries=8)

            # Enable torque and let loaded joints settle before motion starts
            client.send(ENABLE_TORQUE_CMD)
            time.sleep(0.5)

            # Send the goal position with speed and acceleration limits
            client.send({
                "topic": "set_positions",
                "data":  targets,
                "speed": ZERO_SPEED,
                "acc":   ZERO_ACC,
            })

            # While the arm moves, keep pinging torque-enable so that any
            # overload-protection event (servo cuts torque under load) is
            # immediately cleared and the servo resumes moving to its goal.
            deadline = time.monotonic() + ZERO_SETTLE_S
            while time.monotonic() < deadline:
                time.sleep(TORQUE_HEARTBEAT_S)
                client.send(ENABLE_TORQUE_CMD)

            client.close()
            print(f"  {side} arm → zeroed")
        except Exception as exc:
            print(f"  [ERROR] {side} arm: {exc}", file=sys.stderr)
            errors += 1

    if errors == 0:
        print("Both arms set to zero joint angles.")
    return errors


if __name__ == "__main__":
    sys.exit(main())
