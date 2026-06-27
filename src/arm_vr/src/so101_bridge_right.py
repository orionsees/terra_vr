#!/usr/bin/env python3
"""SO101 Bridge (Right Arm) — serial gateway using IPC bus instead of ROS2.

Commands received on bus port 5557 (JSON with "topic" and "data"):
  topic="set_positions"     → arm.set_positions (degrees, keyed by name or servo id)
  topic="set_positions_raw" → arm.set_positions_raw (raw ticks, keyed by servo id int)

Feedback published on bus port 5559:
  topic="right_arm/joint_states" → positions, velocities, efforts, raw_positions, temperatures

Launch:
    python3 so101_bridge_right.py --module-dir <src_dir> --port /dev/ttyACM1
"""

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bus import CommandServer, Publisher, RIGHT_ARM_CMD_PORT, RIGHT_ARM_FEEDBACK_PORT


def _coerce_key(key):
    if isinstance(key, int):
        return key
    stripped = str(key).strip()
    return int(stripped) if stripped.isdigit() else key


class SO101BridgeRight:
    def __init__(self, args) -> None:
        module_dir = Path(args.module_dir).expanduser().resolve()
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))

        from so101 import SO101Arm

        cal_file = (args.calibration_file or "").strip() or None

        self.arm = SO101Arm(
            args.port,
            baudrate=args.baudrate,
            serial_timeout=args.serial_timeout,
            reply_timeout=args.reply_timeout,
            calibration_file=cal_file,
        )
        self.arm.open()

        self.servo_ids = sorted(self.arm.joint_names.keys())
        self.joint_names = [self.arm.joint_names[sid] for sid in self.servo_ids]

        self._hz = max(0.5, float(args.feedback_rate_hz))
        self._running = True
        self._last_deg: dict[str, float] = {}
        self._last_stamp: float | None = None

        self._cmd_srv = CommandServer(RIGHT_ARM_CMD_PORT)
        self._fb_pub  = Publisher(RIGHT_ARM_FEEDBACK_PORT)
        self._bus_lock = threading.Lock()

        threading.Thread(target=self._cmd_loop,      daemon=True).start()
        threading.Thread(target=self._feedback_loop, daemon=True).start()

        print(
            f"[right bridge] ready — port={args.port}  "
            f"cmd_port={RIGHT_ARM_CMD_PORT}  "
            f"feedback_port={RIGHT_ARM_FEEDBACK_PORT}  "
            f"hz={self._hz:.1f}"
        )

    # ------------------------------------------------------------------
    # Command loop
    # ------------------------------------------------------------------

    def _cmd_loop(self) -> None:
        for msg in self._cmd_srv.iter_commands():
            if not self._running:
                break
            try:
                topic     = msg.get("topic", "")
                data      = msg.get("data", {})
                acc       = int(msg.get("acc", 0))
                speed     = int(msg.get("speed", 0))
                goal_time = int(msg.get("goal_time", 0))
                with self._bus_lock:
                    if topic == "enable_torque":
                        self.arm.enable_torque()
                    elif "raw" in topic:
                        targets = {int(k): int(v) for k, v in data.items()}
                        self.arm.set_positions_raw(targets, acc=acc, speed=speed, goal_time=goal_time, expect_ack=False)
                    else:
                        targets = {_coerce_key(k): float(v) for k, v in data.items()}
                        self.arm.set_positions(targets, acc=acc, speed=speed, goal_time=goal_time, expect_ack=False)
            except Exception as exc:
                print(f"[right bridge] cmd error: {exc}")

    # ------------------------------------------------------------------
    # Feedback loop
    # ------------------------------------------------------------------

    def _feedback_loop(self) -> None:
        dt = 1.0 / self._hz
        while self._running:
            if self._bus_lock.acquire(blocking=False):
                try:
                    self._publish_feedback()
                except Exception as exc:
                    print(f"[right bridge] feedback error: {exc}")
                finally:
                    self._bus_lock.release()
            time.sleep(dt)

    def _publish_feedback(self) -> None:
        now = time.time()
        elapsed = (
            None if self._last_stamp is None
            else max(1e-6, now - self._last_stamp)
        )

        named_map, raw_map = self.arm.read_positions_with_raw(self.servo_ids)
        loads_map = self.arm.read_loads_raw(self.servo_ids)

        positions, velocities, efforts, raw_positions, temperatures = [], [], [], [], []
        TEMP_REG, TEMP_LEN = 62, 1

        for sid, jname in zip(self.servo_ids, self.joint_names):
            deg = named_map.get(f"{jname}.pos")
            has_cal = deg is not None

            if deg is None:
                raw_deg = named_map.get(f"{jname}.pos_raw")
                deg = float(raw_deg) if raw_deg is not None else float("nan")
            else:
                deg = float(deg)

            positions.append(deg)

            if has_cal and deg == deg:  # not NaN
                prev = self._last_deg.get(jname)
                if elapsed is None or prev is None:
                    velocities.append(0.0)
                else:
                    velocities.append((deg - prev) / elapsed)
                self._last_deg[jname] = deg
            else:
                velocities.append(float("nan"))

            raw_val = raw_map.get(sid)
            raw_positions.append(int(raw_val) if raw_val is not None else -2147483648)

            try:
                tmp = self.arm.bus.read_data(
                    sid, TEMP_REG, TEMP_LEN, timeout_s=self.arm._reply_timeout
                )
                temperatures.append(float(tmp[0]) if tmp else float("nan"))
            except Exception:
                temperatures.append(float("nan"))

            load_val = loads_map.get(sid)
            efforts.append(float("nan") if load_val is None else float(load_val) / 10.24)

        self._fb_pub.publish({
            "topic":         "right_arm/joint_states",
            "names":         list(self.joint_names),
            "positions":     positions,
            "velocities":    velocities,
            "efforts":       efforts,
            "raw_positions": raw_positions,
            "temperatures":  temperatures,
            "timestamp":     now,
        })
        self._last_stamp = now

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._running = False
        try:
            self.arm.close()
        except Exception:
            pass
        self._cmd_srv.close()
        self._fb_pub.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir",        required=True)
    parser.add_argument("--port",              required=True)
    parser.add_argument("--calibration-file",  default="")
    parser.add_argument("--baudrate",          type=int,   default=1_000_000)
    parser.add_argument("--serial-timeout",    type=float, default=0.02)
    parser.add_argument("--reply-timeout",     type=float, default=0.05)
    parser.add_argument("--feedback-rate-hz",  type=float, default=20.0)
    args = parser.parse_args()

    bridge = SO101BridgeRight(args)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
