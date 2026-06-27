#!/usr/bin/env python3
"""TCP listener for Hand Tracking Streamer (HTS SDK) — publishes via IPC bus.

Receives frames from the Quest over TCP (hand_tracking_sdk) and publishes
them to the local IPC bus on port 5555 so any subscriber (teleop, ik_control,
visualisers, etc.) can consume VR data without ROS2.

Bus messages published (topic → payload keys):
  left_wrist / right_wrist  → x, y, z, qx, qy, qz, qw, timestamp
  left_pinch / right_pinch  → distance_cm, timestamp
  head_pose                 → x, y, z, qx, qy, qz, qw, timestamp

Usage:
    python tcp_wireless_ros.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bus import Publisher, VR_DATA_PORT

from hand_tracking_sdk import (
    HTSClient,
    HTSClientConfig,
    JointName,
    HandSide,
    StreamOutput,
    TransportMode,
)


class HTSPublisher:
    """Receives HTS frames and publishes them to the IPC bus."""

    def __init__(self, host: str, port: int, timeout_s: float = 1.0) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._pub = Publisher(VR_DATA_PORT)
        self._stop = threading.Event()

        t = threading.Thread(target=self._run_client, daemon=True, name="hts-client")
        t.start()

    def _run_client(self) -> None:
        try:
            client = HTSClient(
                HTSClientConfig(
                    transport_mode=TransportMode.TCP_SERVER,
                    host=self._host,
                    port=self._port,
                    timeout_s=self._timeout_s,
                    output=StreamOutput.FRAMES,
                    include_wall_time=True,
                )
            )
            logging.info(f"HTS client listening on {self._host}:{self._port} (TCP)")

            for frame in client.iter_events():
                if self._stop.is_set():
                    break
                try:
                    self._process_frame(frame)
                except Exception as exc:
                    logging.error(f"Frame processing error: {exc}")

        except Exception as exc:
            logging.error(f"HTS client error: {exc}")

    def _process_frame(self, frame) -> None:
        if frame.recv_time_unix_ns is not None:
            ts = frame.recv_time_unix_ns / 1_000_000_000.0
        else:
            ts = time.time()

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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tcp_wireless",
        description="HTS TCP listener — publishes VR data to IPC bus (port 5555)",
    )
    parser.add_argument("--host",      default="0.0.0.0",
                        help="Host/IP to bind the HTS server on (default: 0.0.0.0)")
    parser.add_argument("-p", "--port", type=int, default=8000,
                        help="HTS TCP port to listen on (default: 8000)")
    parser.add_argument("--timeout",   type=float, default=1.0,
                        help="TCP timeout in seconds (default: 1.0)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    pub = HTSPublisher(args.host, args.port, args.timeout)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        pub.close()


if __name__ == "__main__":
    main()
