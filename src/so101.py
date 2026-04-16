"""so101.py — Importable Python library for controlling the SO101/ST3215 robotic arm.

Wraps the low-level ST3215 serial bus and exposes a clean high-level API through
the ``SO101Arm`` class.  All original functionality is preserved; nothing has been
removed or altered — only reorganised for library use.

Quick-start
-----------
    from so101 import SO101Arm

    with SO101Arm("/dev/ttyACM0") as arm:
        # Read all joint positions (degrees if calibrated, raw ticks otherwise)
        print(arm.read_positions())

        # Move joints by name (degrees, requires calibration)
        arm.set_positions({"shoulder_pan": 0.0, "elbow_flex": -20.0})

        # Move joints by ID using raw encoder ticks
        arm.set_positions_raw({1: 2048})

        # Ping a servo to check it's alive
        print(arm.ping(1))

        # Scan which servos respond
        print(arm.scan())

Calibration
-----------
    arm = SO101Arm("/dev/ttyACM0", calibration_file="so101_follower.json")
    # — or —
    arm = SO101Arm("/dev/ttyACM0", calibration_dir="~/.lerobot/calibration/", calibration_id="so101_follower")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import serial  # type: ignore[reportMissingModuleSource]


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

HEADER = b"\xFF\xFF"

INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03

REG_ACCELERATION = 41
REG_GOAL_POSITION = 42
REG_PRESENT_POSITION = 56
REG_PRESENT_LOAD = 60

MODEL_RESOLUTION = 4096

DEFAULT_ID_TO_NAME: dict[int, str] = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _le_u16(low_byte: int, high_byte: int) -> int:
    return ((high_byte & 0xFF) << 8) | (low_byte & 0xFF)


def _decode_sign_magnitude(value: int, sign_bit: int) -> int:
    if value & (1 << sign_bit):
        return -(value & ~(1 << sign_bit))
    return value


def _encode_sign_magnitude(value: int, sign_bit: int) -> int:
    max_mag = (1 << sign_bit) - 1
    if value < 0:
        return (min(-value, max_mag) & max_mag) | (1 << sign_bit)
    return min(value, max_mag) & max_mag


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StatusPacket:
    servo_id: int
    error: int
    params: bytes


@dataclass
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


# ---------------------------------------------------------------------------
# ST3215 serial bus  (low-level, unchanged)
# ---------------------------------------------------------------------------

class ST3215Bus:
    """Thin wrapper around a pyserial port that speaks the ST3215 protocol."""

    def __init__(
        self,
        port: str,
        baudrate: int,
        serial_timeout: float,
        reply_timeout: float,
    ) -> None:
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=serial_timeout,
        )
        self.reply_timeout = reply_timeout

    def __enter__(self) -> "ST3215Bus":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()

    # ------------------------------------------------------------------
    # Internal packet helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _checksum(payload: bytes) -> int:
        return (~sum(payload)) & 0xFF

    def _send_packet(self, servo_id: int, instruction: int, params: bytes = b"") -> None:
        length = len(params) + 2
        payload = bytes((servo_id & 0xFF, length & 0xFF, instruction & 0xFF)) + params
        packet = HEADER + payload + bytes((self._checksum(payload),))
        self.ser.reset_input_buffer()
        self.ser.write(packet)
        self.ser.flush()

    def _read_exact(self, size: int, deadline: float) -> Optional[bytes]:
        out = bytearray()
        while len(out) < size:
            if time.monotonic() >= deadline:
                return None
            chunk = self.ser.read(size - len(out))
            if not chunk:
                continue
            out.extend(chunk)
        return bytes(out)

    def _wait_for_header(self, deadline: float) -> bool:
        prev = -1
        while time.monotonic() < deadline:
            b = self.ser.read(1)
            if not b:
                continue
            curr = b[0]
            if prev == 0xFF and curr == 0xFF:
                return True
            prev = curr
        return False

    # ------------------------------------------------------------------
    # Public bus operations
    # ------------------------------------------------------------------

    def read_status(
        self,
        expected_id: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Optional[StatusPacket]:
        if timeout_s is None:
            timeout_s = self.reply_timeout
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            if not self._wait_for_header(deadline):
                return None

            meta = self._read_exact(3, deadline)
            if meta is None:
                return None

            servo_id, length, error = meta
            if length < 2:
                continue

            params_len = length - 2
            params = self._read_exact(params_len, deadline)
            checksum = self._read_exact(1, deadline)
            if params is None or checksum is None:
                return None

            calc = (~((servo_id + length + error + sum(params)) & 0xFF)) & 0xFF
            if checksum[0] != calc:
                continue

            if expected_id is not None and servo_id != expected_id:
                continue

            return StatusPacket(servo_id=servo_id, error=error, params=params)

        return None

    def ping(self, servo_id: int, timeout_s: Optional[float] = None) -> bool:
        self._send_packet(servo_id, INST_PING)
        status = self.read_status(expected_id=servo_id, timeout_s=timeout_s)
        return status is not None and status.error == 0 and len(status.params) == 0

    def read_data(
        self,
        servo_id: int,
        address: int,
        read_len: int,
        timeout_s: Optional[float] = None,
    ) -> Optional[bytes]:
        self._send_packet(servo_id, INST_READ, bytes((address & 0xFF, read_len & 0xFF)))
        status = self.read_status(expected_id=servo_id, timeout_s=timeout_s)
        if status is None or status.error != 0:
            return None
        if len(status.params) != read_len:
            return None
        return status.params

    def write_data(
        self,
        servo_id: int,
        address: int,
        payload: bytes,
        *,
        expect_ack: bool,
        timeout_s: Optional[float] = None,
    ) -> bool:
        self._send_packet(servo_id, INST_WRITE, bytes((address & 0xFF,)) + payload)
        if not expect_ack:
            return True
        status = self.read_status(expected_id=servo_id, timeout_s=timeout_s)
        return status is not None and status.error == 0

    def read_position_raw(
        self, servo_id: int, timeout_s: Optional[float] = None
    ) -> Optional[int]:
        data = self.read_data(servo_id, REG_PRESENT_POSITION, 2, timeout_s=timeout_s)
        if data is None:
            return None
        return _decode_sign_magnitude(_le_u16(data[0], data[1]), sign_bit=15)

    def read_load_raw(
        self, servo_id: int, timeout_s: Optional[float] = None
    ) -> Optional[int]:
        data = self.read_data(servo_id, REG_PRESENT_LOAD, 2, timeout_s=timeout_s)
        if data is None:
            return None
        return _decode_sign_magnitude(_le_u16(data[0], data[1]), sign_bit=15)

    def write_goal_position(
        self,
        servo_id: int,
        raw_position: int,
        *,
        acc: int,
        speed: int,
        goal_time: int,
        expect_ack: bool,
        timeout_s: Optional[float] = None,
    ) -> bool:
        pos = _encode_sign_magnitude(raw_position, sign_bit=15)
        payload = bytes(
            (
                acc & 0xFF,
                pos & 0xFF,
                (pos >> 8) & 0xFF,
                goal_time & 0xFF,
                (goal_time >> 8) & 0xFF,
                speed & 0xFF,
                (speed >> 8) & 0xFF,
            )
        )
        return self.write_data(
            servo_id,
            REG_ACCELERATION,
            payload,
            expect_ack=expect_ack,
            timeout_s=timeout_s,
        )


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def load_calibration(
    path: Path,
) -> tuple[dict[int, MotorCalibration], dict[int, str]]:
    """Load a calibration JSON file.

    Supports both plain and nested (``{"calibration": {...}}``) formats.

    Returns
    -------
    by_id : dict mapping servo ID → MotorCalibration
    names : dict mapping servo ID → joint name
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Calibration JSON root must be an object")

    if "calibration" in data and isinstance(data["calibration"], dict):
        data = data["calibration"]

    by_id: dict[int, MotorCalibration] = {}
    names: dict[int, str] = {}

    for joint_name, entry in data.items():
        if not isinstance(joint_name, str) or not isinstance(entry, dict):
            continue
        required = {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
        if not required.issubset(entry.keys()):
            continue

        cal = MotorCalibration(
            id=int(entry["id"]),
            drive_mode=int(entry["drive_mode"]),
            homing_offset=int(entry["homing_offset"]),
            range_min=int(entry["range_min"]),
            range_max=int(entry["range_max"]),
        )
        by_id[cal.id] = cal
        names[cal.id] = joint_name

    if not by_id:
        raise ValueError(f"No valid motor calibration entries found in {path}")

    return by_id, names


def raw_to_degrees(raw: int, cal: MotorCalibration) -> float:
    """Convert a raw encoder tick to degrees using the given calibration."""
    mid = (cal.range_min + cal.range_max) / 2.0
    return (raw - mid) * 360.0 / (MODEL_RESOLUTION - 1)


def degrees_to_raw(deg: float, cal: MotorCalibration) -> int:
    """Convert degrees to a raw encoder tick, clamped to the calibrated range."""
    mid = (cal.range_min + cal.range_max) / 2.0
    raw = int(round((deg * (MODEL_RESOLUTION - 1) / 360.0) + mid))
    lo = min(cal.range_min, cal.range_max)
    hi = max(cal.range_min, cal.range_max)
    return _clamp(raw, lo, hi)


# ---------------------------------------------------------------------------
# High-level arm API
# ---------------------------------------------------------------------------

class SO101Arm:
    """High-level controller for the SO-101 robotic arm.

    Parameters
    ----------
    port : str
        Serial device path, e.g. ``"/dev/ttyACM0"`` or ``"COM3"``.
    baudrate : int
        Bus baudrate (default 1 000 000).
    serial_timeout : float
        pyserial read timeout in seconds (default 0.02).
    reply_timeout : float
        Maximum time to wait for a servo reply packet (default 0.05).
    calibration_file : str | Path | None
        Direct path to a calibration JSON file.
    calibration_dir : str | Path | None
        Directory that contains ``<calibration_id>.json``.
    calibration_id : str
        Used with *calibration_dir* to locate ``<calibration_id>.json``
        (default ``"so101_follower"``).

    Examples
    --------
    Using as a context manager (recommended)::

        with SO101Arm("/dev/ttyACM0", calibration_file="cal.json") as arm:
            print(arm.read_positions())
            arm.set_positions({"shoulder_pan": 0.0, "gripper": 15.0})

    Manual open/close::

        arm = SO101Arm("/dev/ttyACM0")
        arm.open()
        ...
        arm.close()
    """

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 1_000_000,
        serial_timeout: float = 0.02,
        reply_timeout: float = 0.05,
        calibration_file: Optional[str | Path] = None,
        calibration_dir: Optional[str | Path] = None,
        calibration_id: str = "so101_follower",
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._serial_timeout = serial_timeout
        self._reply_timeout = reply_timeout

        self._cal_by_id: dict[int, MotorCalibration] = {}
        self._id_to_name: dict[int, str] = dict(DEFAULT_ID_TO_NAME)

        # Resolve calibration path
        cal_path = self._resolve_cal_path(calibration_file, calibration_dir, calibration_id)
        if cal_path is not None and cal_path.is_file():
            self._cal_by_id, names = load_calibration(cal_path)
            self._id_to_name.update(names)
        elif cal_path is not None:
            # Non-fatal — operate without calibration
            pass

        self._bus: Optional[ST3215Bus] = None

    # ------------------------------------------------------------------
    # Context manager / lifecycle
    # ------------------------------------------------------------------

    def open(self) -> "SO101Arm":
        """Open the serial connection.  Called automatically by ``__enter__``."""
        if self._bus is None:
            self._bus = ST3215Bus(
                port=self._port,
                baudrate=self._baudrate,
                serial_timeout=self._serial_timeout,
                reply_timeout=self._reply_timeout,
            )
        return self

    def close(self) -> None:
        """Close the serial connection."""
        if self._bus is not None:
            self._bus.close()
            self._bus = None

    def __enter__(self) -> "SO101Arm":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ping(self, servo_id: int) -> bool:
        """Return ``True`` if the servo at *servo_id* responds."""
        return self._bus_required().ping(servo_id, timeout_s=self._reply_timeout)

    def scan(self, servo_ids: Optional[list[int]] = None) -> list[int]:
        """Ping *servo_ids* (default: 1–6) and return IDs that respond."""
        if servo_ids is None:
            servo_ids = list(range(1, 7))
        bus = self._bus_required()
        return [sid for sid in servo_ids if bus.ping(sid, timeout_s=self._reply_timeout)]

    def read_position_raw(self, servo_id: int) -> Optional[int]:
        """Read the raw encoder tick for a single servo."""
        return self._bus_required().read_position_raw(servo_id, timeout_s=self._reply_timeout)

    def read_positions_raw(
        self, servo_ids: Optional[list[int]] = None
    ) -> dict[int, Optional[int]]:
        """Read raw encoder ticks for all (or selected) servos.

        Returns a ``{servo_id: raw_tick}`` mapping.  ``None`` means no reply.
        """
        if servo_ids is None:
            servo_ids = list(self._id_to_name.keys()) or list(range(1, 7))
        bus = self._bus_required()
        return {
            sid: bus.read_position_raw(sid, timeout_s=self._reply_timeout)
            for sid in servo_ids
        }

    def read_loads_raw(
        self, servo_ids: Optional[list[int]] = None
    ) -> dict[int, Optional[int]]:
        """Read raw torque/load values for all (or selected) servos.

        Returns a ``{servo_id: raw_load}`` mapping.  ``None`` means no reply.
        """
        if servo_ids is None:
            servo_ids = list(self._id_to_name.keys()) or list(range(1, 7))
        bus = self._bus_required()
        return {
            sid: bus.read_load_raw(sid, timeout_s=self._reply_timeout)
            for sid in servo_ids
        }

    def read_positions(
        self, servo_ids: Optional[list[int]] = None
    ) -> dict[str, float | int | None]:
        """Read joint positions.

        Returns a dict keyed by joint name (or ``"id_N"`` if unnamed).
        Values are in **degrees** when calibration is available, or raw ticks
        (key suffix ``_raw``) otherwise.  ``None`` means no reply from that servo.
        """
        raw_map = self.read_positions_raw(servo_ids)
        return self._format_positions(raw_map)

    def set_position_raw(
        self,
        servo_id: int,
        raw_position: int,
        *,
        acc: int = 0,
        speed: int = 0,
        goal_time: int = 0,
        expect_ack: bool = True,
    ) -> bool:
        """Send a raw encoder tick target to one servo.

        Returns ``True`` on success (or when *expect_ack* is ``False``).
        """
        return self._bus_required().write_goal_position(
            servo_id,
            raw_position,
            acc=_clamp(acc, 0, 254),
            speed=_clamp(speed, 0, 65535),
            goal_time=_clamp(goal_time, 0, 65535),
            expect_ack=expect_ack,
            timeout_s=self._reply_timeout,
        )

    def set_positions_raw(
        self,
        targets: dict[int, int],
        *,
        acc: int = 0,
        speed: int = 0,
        goal_time: int = 0,
        expect_ack: bool = True,
        settle_seconds: float = 0.0,
    ) -> dict[int, bool]:
        """Send raw encoder tick targets to multiple servos.

        Parameters
        ----------
        targets : dict
            ``{servo_id: raw_tick}`` mapping.

        Returns
        -------
        dict mapping servo_id → success flag.
        """
        results: dict[int, bool] = {}
        for servo_id, raw in targets.items():
            ok = self.set_position_raw(
                servo_id,
                raw,
                acc=acc,
                speed=speed,
                goal_time=goal_time,
                expect_ack=expect_ack,
            )
            if not ok and expect_ack:
                raise RuntimeError(f"No ACK received for write to servo ID {servo_id}")
            results[servo_id] = ok

        if settle_seconds > 0:
            time.sleep(settle_seconds)

        return results

    def set_position(
        self,
        joint: str | int,
        degrees: float,
        *,
        acc: int = 0,
        speed: int = 0,
        goal_time: int = 0,
        expect_ack: bool = True,
    ) -> bool:
        """Move one joint to *degrees*.

        *joint* may be a joint name (e.g. ``"shoulder_pan"``) or a servo ID.
        Requires calibration to be loaded.
        """
        servo_id = self._resolve_joint(joint)
        if servo_id not in self._cal_by_id:
            raise ValueError(
                f"No calibration for joint '{joint}' (servo {servo_id}). "
                "Load a calibration file or use set_position_raw()."
            )
        raw = degrees_to_raw(degrees, self._cal_by_id[servo_id])
        return self.set_position_raw(
            servo_id, raw, acc=acc, speed=speed, goal_time=goal_time, expect_ack=expect_ack
        )

    def set_positions(
        self,
        targets: dict[str | int, float],
        *,
        acc: int = 0,
        speed: int = 0,
        goal_time: int = 0,
        expect_ack: bool = True,
        settle_seconds: float = 0.0,
    ) -> dict[str | int, bool]:
        """Move multiple joints to degree targets.

        Parameters
        ----------
        targets : dict
            ``{joint_name_or_id: degrees}`` mapping.
        settle_seconds : float
            Optional delay (seconds) after sending all commands.

        Returns
        -------
        dict mapping each input key → success flag.
        """
        results: dict[str | int, bool] = {}
        for joint, deg in targets.items():
            ok = self.set_position(
                joint, deg, acc=acc, speed=speed, goal_time=goal_time, expect_ack=expect_ack
            )
            results[joint] = ok

        if settle_seconds > 0:
            time.sleep(settle_seconds)

        return results

    # ------------------------------------------------------------------
    # Calibration helpers (public)
    # ------------------------------------------------------------------

    def load_calibration_file(self, path: str | Path) -> None:
        """Load (or reload) calibration from *path* at runtime."""
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Calibration file not found: {p}")
        self._cal_by_id, names = load_calibration(p)
        self._id_to_name.update(names)

    @property
    def calibrated_ids(self) -> list[int]:
        """Servo IDs that have calibration data loaded."""
        return list(self._cal_by_id.keys())

    @property
    def joint_names(self) -> dict[int, str]:
        """Mapping of servo ID → joint name."""
        return dict(self._id_to_name)

    # ------------------------------------------------------------------
    # Low-level bus access (escape hatch)
    # ------------------------------------------------------------------

    @property
    def bus(self) -> ST3215Bus:
        """Direct access to the underlying :class:`ST3215Bus` (advanced use)."""
        return self._bus_required()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bus_required(self) -> ST3215Bus:
        if self._bus is None:
            raise RuntimeError(
                "Serial port is not open. "
                "Use 'with SO101Arm(...) as arm:' or call arm.open() first."
            )
        return self._bus

    def _resolve_joint(self, joint: str | int) -> int:
        """Return a servo ID from a joint name or numeric ID."""
        if isinstance(joint, int):
            return joint
        # Reverse lookup: name → id
        name_to_id = {v: k for k, v in self._id_to_name.items()}
        if joint not in name_to_id:
            raise KeyError(f"Unknown joint name '{joint}'. Known joints: {list(name_to_id)}")
        return name_to_id[joint]

    def _format_positions(
        self, positions_raw: dict[int, Optional[int]]
    ) -> dict[str, float | int | None]:
        out: dict[str, float | int | None] = {}
        for servo_id, raw in positions_raw.items():
            name = self._id_to_name.get(servo_id, f"id_{servo_id}")
            if raw is None:
                out[f"{name}.pos"] = None
                continue
            if servo_id in self._cal_by_id:
                out[f"{name}.pos"] = round(raw_to_degrees(raw, self._cal_by_id[servo_id]), 3)
            else:
                out[f"{name}.pos_raw"] = raw
        return out

    @staticmethod
    def _resolve_cal_path(
        calibration_file: Optional[str | Path],
        calibration_dir: Optional[str | Path],
        calibration_id: str,
    ) -> Optional[Path]:
        if calibration_file is not None:
            p = Path(calibration_file).expanduser().resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Calibration file not found: {p}")
            return p
        if calibration_dir is not None:
            directory = Path(calibration_dir).expanduser().resolve()
            return directory / f"{calibration_id}.json"
        return None
