#!/usr/bin/env python3
"""SO101 ROS2 Bridge — Serial-to-ROS2 gateway for the SO101/ST3215 robotic arm.

This bridge connects to the arm via serial and publishes joint states while
subscribing to command topics for motion control.

Launch with:
    ros2 launch arm_package so101_bridge.launch.py
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from example_interfaces.srv import AddTwoInts
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Int32MultiArray, String
from std_srvs.srv import Trigger


def _coerce_joint_key(key):
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        stripped = key.strip()
        if stripped.isdigit():
            return int(stripped)
    return key


def _clock_sec(node):
    stamp = node.get_clock().now().to_msg()
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class SO101Bridge(Node):
    def __init__(self, args):
        super().__init__("so101_bridge")

        module_dir = Path(args.module_dir).expanduser().resolve()
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))

        from so101 import SO101Arm

        calibration_file = args.calibration_file.strip() if args.calibration_file else ""
        calibration_file = calibration_file if calibration_file else None

        self.arm = SO101Arm(
            args.port,
            baudrate=args.baudrate,
            serial_timeout=args.serial_timeout,
            reply_timeout=args.reply_timeout,
            calibration_file=calibration_file,
        )
        self.arm.open()

        self.servo_ids = sorted(self.arm.joint_names.keys())
        self.joint_names = [self.arm.joint_names[sid] for sid in self.servo_ids]

        self.last_deg_positions = {}
        self.last_feedback_stamp = None

        qos = QoSProfile(depth=10)

        self.pub_joint_states = self.create_publisher(JointState, "joint_states", qos)
        self.pub_raw_positions = self.create_publisher(
            Int32MultiArray, "raw_positions", qos
        )
        self.pub_temp = self.create_publisher(
            Float64MultiArray, "temp", qos
        )

        self.create_subscription(
            String,
            "cmd/set_positions",
            self._on_set_positions_topic,
            qos,
        )
        self.create_subscription(
            String,
            "cmd/set_positions_raw",
            self._on_set_positions_raw_topic,
            qos,
        )

        self.create_service(AddTwoInts, "ping", self._on_ping)
        self.create_service(Trigger, "scan", self._on_scan)
        self.create_service(Trigger, "set_positions", self._on_set_positions_service)
        self.create_service(
            Trigger,
            "set_positions_raw",
            self._on_set_positions_raw_service,
        )

        self.declare_parameter("set_positions_json", "{}")
        self.declare_parameter("set_positions_raw_json", "{}")

        hz = max(0.5, float(args.feedback_rate_hz))
        self.create_timer(1.0 / hz, self._publish_feedback)

        self.get_logger().info(
            f"SO101 bridge ready on port={args.port}, feedback_rate_hz={hz:.2f}"
        )

    def destroy_node(self):
        try:
            self.arm.close()
        except Exception as exc:
            self.get_logger().error(f"Failed to close arm cleanly: {exc}")
        return super().destroy_node()

    def _on_set_positions_topic(self, msg):
        try:
            targets_raw = json.loads(msg.data)
            if not isinstance(targets_raw, dict):
                raise ValueError("set_positions topic payload must be a JSON object")
            targets = {_coerce_joint_key(k): float(v) for k, v in targets_raw.items()}
            result = self.arm.set_positions(targets)
            self.get_logger().info(f"set_positions executed: {result}")
        except Exception as exc:
            self.get_logger().error(f"set_positions failed: {exc}")

    def _on_set_positions_raw_topic(self, msg):
        try:
            targets_raw = json.loads(msg.data)
            if not isinstance(targets_raw, dict):
                raise ValueError("set_positions_raw topic payload must be a JSON object")
            targets = {int(k): int(v) for k, v in targets_raw.items()}
            result = self.arm.set_positions_raw(targets)
            self.get_logger().info(f"set_positions_raw executed: {result}")
        except Exception as exc:
            self.get_logger().error(f"set_positions_raw failed: {exc}")

    def _on_ping(self, request, response):
        try:
            servo_id = int(request.a)
            alive = bool(self.arm.ping(servo_id))
            response.sum = 1 if alive else 0
        except Exception as exc:
            response.sum = 0
        return response

    def _on_scan(self, request, response):
        del request
        try:
            ids = self.arm.scan(self.servo_ids)
            response.success = True
            response.message = json.dumps({"ids": ids})
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _on_set_positions_service(self, request, response):
        del request
        try:
            targets_raw = json.loads(self.get_parameter("set_positions_json").value)
            if not isinstance(targets_raw, dict):
                raise ValueError("set_positions_json parameter must be a JSON object")
            targets = {_coerce_joint_key(k): float(v) for k, v in targets_raw.items()}
            result = self.arm.set_positions(targets)
            response.success = True
            response.message = json.dumps(result)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _on_set_positions_raw_service(self, request, response):
        del request
        try:
            targets_raw = json.loads(self.get_parameter("set_positions_raw_json").value)
            if not isinstance(targets_raw, dict):
                raise ValueError("set_positions_raw_json parameter must be a JSON object")
            targets = {int(k): int(v) for k, v in targets_raw.items()}
            result = self.arm.set_positions_raw(targets)
            response.success = True
            response.message = json.dumps(result)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _publish_feedback(self):
        try:
            now_s = _clock_sec(self)
            dt = None
            if self.last_feedback_stamp is not None:
                dt = max(1e-6, now_s - self.last_feedback_stamp)

            raw_map = self.arm.read_positions_raw(self.servo_ids)
            named_map = self.arm.read_positions(self.servo_ids)
            loads_map = self.arm.read_loads_raw(self.servo_ids)

            angles_deg = []
            speeds_deg_s = []
            raw_positions = []
            temperatures = []
            
            # Temperature register address for ST3215 servo (typical register 62)
            TEMP_REGISTER = 62
            TEMP_REGISTER_LEN = 1

            for sid, jname in zip(self.servo_ids, self.joint_names):
                key_deg = f"{jname}.pos"
                key_raw = f"{jname}.pos_raw"

                deg_val = named_map.get(key_deg)
                has_calibrated_degrees = deg_val is not None

                if deg_val is None and key_raw in named_map:
                    deg_val = float(named_map[key_raw])

                if deg_val is None:
                    angles_deg.append(float("nan"))
                    speeds_deg_s.append(float("nan"))
                else:
                    deg_val = float(deg_val)
                    angles_deg.append(deg_val)

                    if not has_calibrated_degrees:
                        speeds_deg_s.append(float("nan"))
                    else:
                        prev = self.last_deg_positions.get(jname)
                        if dt is None or prev is None:
                            speeds_deg_s.append(0.0)
                        else:
                            speeds_deg_s.append((deg_val - prev) / dt)
                        self.last_deg_positions[jname] = deg_val

                raw_val = raw_map.get(sid)
                raw_positions.append(int(raw_val) if raw_val is not None else -2147483648)

                # Read temperature from servo (register 62)
                try:
                    temp_data = self.arm.bus.read_data(
                        sid, TEMP_REGISTER, TEMP_REGISTER_LEN, timeout_s=self.arm._reply_timeout
                    )
                    if temp_data is not None and len(temp_data) > 0:
                        temperatures.append(float(temp_data[0]))
                    else:
                        temperatures.append(float("nan"))
                except Exception:
                    temperatures.append(float("nan"))

            # Publish /so101/joint_states (includes angles, velocities, and torques)
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = list(self.joint_names)
            js.position = list(angles_deg)
            js.velocity = list(speeds_deg_s)
            # Include torque feedback as effort
            efforts = []
            for load_val in loads_map.values():
                if load_val is None:
                    efforts.append(float("nan"))
                else:
                    efforts.append(float(load_val) / 10.24)
            js.effort = list(efforts)
            self.pub_joint_states.publish(js)

            # Publish /so101/raw_positions
            raw_msg = Int32MultiArray()
            raw_msg.data = list(raw_positions)
            self.pub_raw_positions.publish(raw_msg)

            # Publish /so101/temp
            temp_msg = Float64MultiArray()
            temp_msg.data = list(temperatures)
            self.pub_temp.publish(temp_msg)

            self.last_feedback_stamp = now_s
        except Exception as exc:
            self.get_logger().error(f"Feedback publish failed: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration-file", default="")
    parser.add_argument("--baudrate", type=int, default=1000000)
    parser.add_argument("--serial-timeout", type=float, default=0.02)
    parser.add_argument("--reply-timeout", type=float, default=0.05)
    parser.add_argument("--feedback-rate-hz", type=float, default=20.0)
    args = parser.parse_args()

    rclpy.init()
    node = None
    try:
        node = SO101Bridge(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
