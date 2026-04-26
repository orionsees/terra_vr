#!/usr/bin/env python3
"""Subscribes to /right_wrist (geometry_msgs/PoseStamped) and prints XYZ + RPY."""

import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])

from so101 import SO101Arm

PORT = "/dev/ttyACM0"
CAL  = "/home/ubunt2/Documents/arm_vr/config/right_arm.json"
with open(CAL) as f:
    JOINTS = json.load(f)

# Yaw range (degrees) that maps to wrist_roll [range_min, range_max]
WRIST_ROLL_YAW_MIN = -170.0   # yaw at range_min
WRIST_ROLL_YAW_MAX =  170.0   # yaw at range_max

# Roll range (degrees) that maps to wrist_flex [range_min, range_max]
WRIST_FLEX_ROLL_MIN = -65.0   # roll at range_min
WRIST_FLEX_ROLL_MAX =  50.0   # roll at range_max

# Spike rejection: maximum allowed change per callback (degrees / cm)
# Increase if legitimate fast motion is rejected; decrease to filter more
SPIKE_THRESHOLD_YAW   = 15.0  # degrees — wrist_roll
SPIKE_THRESHOLD_ROLL  = 15.0  # degrees — wrist_flex
SPIKE_THRESHOLD_PINCH =  5.0  # cm      — gripper

# Deadband: minimum servo tick change required to send a new command
# Suppresses jitter when wrist is held still (1 tick ≈ 0.088°)
DEADBAND_WRIST_ROLL  = 20   # ticks
DEADBAND_WRIST_FLEX  = 20   # ticks
DEADBAND_GRIPPER     = 5   # ticks

def quaternion_to_rpy(x, y, z, w):
    """Convert quaternion to roll, pitch, yaw (radians)."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class RightWristSubscriber(Node):
    def __init__(self):
        super().__init__("right_wrist_subscriber")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PoseStamped, "/right_wrist", self._callback, qos)
        self.create_subscription(Float32, "/right_hand/pinch_distance", self.right_hand_pinch_callback, qos)
        self.get_logger().info("Subscribed to /right_wrist")

        # Keep arm open for the lifetime of the node
        self._arm = SO101Arm(PORT, calibration_file=CAL, reply_timeout=0.2)
        self._arm.open()

        self._last_pinch = None
        self._last_pinch_time = None

        # Last accepted values for spike filtering
        self._last_yaw   = None
        self._last_roll  = None
        self._last_gripper_pinch = None

        # Last sent servo positions for deadband filtering
        self._sent_wrist_roll = None
        self._sent_wrist_flex = None
        self._sent_gripper    = None

        self.control_flex()

    def _callback(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        roll, pitch, yaw = quaternion_to_rpy(q.x, q.y, q.z, q.w)
        yaw_deg = math.degrees(yaw)

        print(
            f"XYZ: ({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) | "
            f"RPY: ({math.degrees(roll):.2f}°, {math.degrees(pitch):.2f}°, {yaw_deg:.2f}°)"
        )

        # Map yaw [WRIST_ROLL_YAW_MIN, WRIST_ROLL_YAW_MAX] → wrist_roll [range_min, range_max]
        # Ignore values outside the defined range — motor holds last position
        if WRIST_ROLL_YAW_MIN <= yaw_deg <= WRIST_ROLL_YAW_MAX:
            if self._last_yaw is None or abs(yaw_deg - self._last_yaw) <= SPIKE_THRESHOLD_YAW:
                self._last_yaw = yaw_deg
                fraction = (yaw_deg - WRIST_ROLL_YAW_MIN) / (WRIST_ROLL_YAW_MAX - WRIST_ROLL_YAW_MIN)
                wrist_roll_pos = int(
                    JOINTS["wrist_roll"]["range_min"] + fraction * (JOINTS["wrist_roll"]["range_max"] - JOINTS["wrist_roll"]["range_min"])
                )
                if self._sent_wrist_roll is None or abs(wrist_roll_pos - self._sent_wrist_roll) >= DEADBAND_WRIST_ROLL:
                    self._sent_wrist_roll = wrist_roll_pos
                    self._arm.set_positions_raw({5: wrist_roll_pos}, acc=254, speed=0, expect_ack=False)
            else:
                self.get_logger().warn(f"Yaw spike rejected: {self._last_yaw:.1f}° → {yaw_deg:.1f}°")

        # Map roll [WRIST_FLEX_ROLL_MIN, WRIST_FLEX_ROLL_MAX] → wrist_flex [range_min, range_max]
        # When wrist_roll rotates, the flex axis rotates with it — blend roll and pitch accordingly:
        #   theta=0°  → pure roll (wrist upright)
        #   theta=-90° → pure pitch (wrist rotated -90°)
        roll_deg  = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        theta = math.radians(self._last_yaw if self._last_yaw is not None else 0.0)
        effective_flex = roll_deg * math.cos(theta) - pitch_deg * math.sin(theta)

        if WRIST_FLEX_ROLL_MIN <= effective_flex <= WRIST_FLEX_ROLL_MAX:
            if self._last_roll is None or abs(effective_flex - self._last_roll) <= SPIKE_THRESHOLD_ROLL:
                self._last_roll = effective_flex
                fraction = (effective_flex - WRIST_FLEX_ROLL_MIN) / (WRIST_FLEX_ROLL_MAX - WRIST_FLEX_ROLL_MIN)
                wrist_flex_pos = int(
                    JOINTS["wrist_flex"]["range_min"] + fraction * (JOINTS["wrist_flex"]["range_max"] - JOINTS["wrist_flex"]["range_min"])
                )
                if self._sent_wrist_flex is None or abs(wrist_flex_pos - self._sent_wrist_flex) >= DEADBAND_WRIST_FLEX:
                    self._sent_wrist_flex = wrist_flex_pos
                    self._arm.set_positions_raw({4: wrist_flex_pos}, acc=254, speed=0, expect_ack=False)
            else:
                self.get_logger().warn(f"Flex spike rejected: {self._last_roll:.1f}° → {effective_flex:.1f}°")

    def right_hand_pinch_callback(self, msg: Float32):
        pinch_distance = msg.data
        now = time.monotonic()

        self._last_pinch = pinch_distance
        self._last_pinch_time = now

        pinch_distance = max(2.0, min(10.0, pinch_distance))

        if self._last_gripper_pinch is not None and abs(pinch_distance - self._last_gripper_pinch) > SPIKE_THRESHOLD_PINCH:
            self.get_logger().warn(f"Pinch spike rejected: {self._last_gripper_pinch:.2f} → {pinch_distance:.2f} cm")
            return
        self._last_gripper_pinch = pinch_distance

        fraction = (pinch_distance - 2.0) / 8.0

        gripper_servo_pos = int(
            JOINTS["gripper"]["range_min"] + fraction * (JOINTS["gripper"]["range_max"] - JOINTS["gripper"]["range_min"])
        )
        if self._sent_gripper is None or abs(gripper_servo_pos - self._sent_gripper) >= DEADBAND_GRIPPER:
            self._sent_gripper = gripper_servo_pos
            self._arm.set_positions_raw({6: gripper_servo_pos}, acc=254, speed=0, expect_ack=False)

    def control_flex(self):
        # self._arm.set_positions_raw({4: 2048}, acc=10, speed=100, settle_seconds=1.0)
        middle_1 = int((JOINTS["shoulder_pan"]["range_min"] + JOINTS["shoulder_pan"]["range_max"]) / 2)
        middle_2 = int((JOINTS["shoulder_lift"]["range_min"] + JOINTS["shoulder_lift"]["range_max"]) / 2)
        middle_3 = int((JOINTS["elbow_flex"]["range_min"] + JOINTS["elbow_flex"]["range_max"]) / 2)
        middle_4 = int((JOINTS["wrist_flex"]["range_min"] + JOINTS["wrist_flex"]["range_max"]) / 2)
        middle_5 = int((JOINTS["wrist_roll"]["range_min"] + JOINTS["wrist_roll"]["range_max"]) / 2)
        middle_6 = int((JOINTS["gripper"]["range_min"] + JOINTS["gripper"]["range_max"]) / 2)
        # self._arm.set_positions_raw({4: JOINTS["wrist_flex"]["range_min"]}, acc=254, speed=0, expect_ack=False)
        # home = self._arm.set_positions_raw({1: middle_1, 2: 3020, 3: 955, 4: middle_4, 5: middle_5, 6: int(JOINTS["gripper"]["range_min"])}, acc=10, speed=100, settle_seconds=1.0)



def main():
    rclpy.init()
    node = RightWristSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
