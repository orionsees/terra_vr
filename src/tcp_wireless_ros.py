#!/usr/bin/env python3
"""TCP wireless listener for Hand Tracking Streamer (HTS) using official SDK.

Uses the hand_tracking_sdk library to handle TCP connection and parsing.
Publishes to ROS2 topics:

    /left_wrist                (geometry_msgs/PoseStamped)
    /right_wrist               (geometry_msgs/PoseStamped)
    /head_pose                 (geometry_msgs/PoseStamped)
    /left_hand/<joint>         (geometry_msgs/PointStamped)  x 21 joints
    /right_hand/<joint>        (geometry_msgs/PointStamped)  x 21 joints
    /left_hand/pinch_distance  (std_msgs/Float32)
    /right_hand/pinch_distance (std_msgs/Float32)

Example:
    python tcp_wireless_ros.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float32

from hand_tracking_sdk import (
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
    JointName,
    HandSide,
)


class HTSPosePublisher(Node):
    """ROS2 node that publishes HTS frames as poses and landmarks."""

    def __init__(self, host: str, port: int, timeout_s: float = 1.0) -> None:
        super().__init__("hts_pose_publisher")

        self._host = host
        self._port = port
        self._timeout_s = timeout_s

        # Best-effort QoS for high-frequency sensor data
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Wrist pose publishers
        self._wrist_pubs = {
            HandSide.LEFT: self.create_publisher(PoseStamped, "left_wrist", sensor_qos),
            HandSide.RIGHT: self.create_publisher(PoseStamped, "right_wrist", sensor_qos),
        }

        # Head pose publisher
        self._head_pose_pub = self.create_publisher(PoseStamped, "head_pose", sensor_qos)

        # Landmark publishers: {(HandSide.LEFT, JointName.THUMB_TIP): publisher, ...}
        self._landmark_pubs: dict[tuple[HandSide, JointName], rclpy.publisher.Publisher] = {}
        for side in [HandSide.LEFT, HandSide.RIGHT]:
            for joint in JointName:
                side_str = "left_hand" if side == HandSide.LEFT else "right_hand"
                topic = f"{side_str}/{joint.value.lower()}"
                self._landmark_pubs[(side, joint)] = self.create_publisher(
                    PointStamped, topic, sensor_qos
                )

        # Pinch distance publishers
        self._pinch_distance_pubs = {
            HandSide.LEFT: self.create_publisher(Float32, "left_hand/pinch_distance", sensor_qos),
            HandSide.RIGHT: self.create_publisher(Float32, "right_hand/pinch_distance", sensor_qos),
        }

        self.get_logger().info("HTSPosePublisher initialized")

        # Start SDK client in background thread
        self._stop_event = threading.Event()
        self._client_thread = threading.Thread(
            target=self._run_client,
            daemon=True,
            name="hts-client"
        )
        self._client_thread.start()

    def destroy_node(self) -> bool:
        """Signal client thread to stop and cleanup."""
        self._stop_event.set()
        return super().destroy_node()

    def _run_client(self) -> None:
        """Run HTS client in background thread (blocks on iter_events)."""
        try:
            # Configure SDK client for TCP server mode
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

            self.get_logger().info(
                f"HTS client listening on {self._host}:{self._port} (TCP)"
            )

            # Process frames from SDK
            for frame in client.iter_events():
                if self._stop_event.is_set():
                    break

                try:
                    self._process_frame(frame)
                except Exception as e:
                    self.get_logger().error(f"Error processing frame: {e}")

        except Exception as e:
            self.get_logger().error(f"HTS client error: {e}")

    def _process_frame(self, frame) -> None:
        """Process a single frame from HTS SDK."""

        # Use wall time if available, otherwise use ROS time
        if frame.recv_time_unix_ns is not None:
            timestamp_ns = frame.recv_time_unix_ns
            sec = timestamp_ns // 1_000_000_000
            nanosec = timestamp_ns % 1_000_000_000
        else:
            ros_time = self.get_clock().now()
            sec = ros_time.seconds_nanoseconds()[0]
            nanosec = ros_time.seconds_nanoseconds()[1]

        if frame.side == HandSide.HEAD:
            self._publish_head_pose(frame, sec, nanosec)
        else:
            self._publish_wrist_pose(frame, sec, nanosec)
            self._publish_landmarks(frame, sec, nanosec)
            self._publish_pinch_distance(frame)

    def _publish_head_pose(self, frame, sec: int, nanosec: int) -> None:
        """Publish head pose from HeadFrame."""
        head = frame.head

        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.pose.position.x = head.x
        msg.pose.position.y = head.y
        msg.pose.position.z = head.z
        msg.pose.orientation.x = head.qx
        msg.pose.orientation.y = head.qy
        msg.pose.orientation.z = head.qz
        msg.pose.orientation.w = head.qw
        self._head_pose_pub.publish(msg)

        self.get_logger().debug(
            f"Head: pos=({head.x:.3f}, {head.y:.3f}, {head.z:.3f})"
        )

    def _publish_wrist_pose(self, frame, sec: int, nanosec: int) -> None:
        """Publish wrist pose from HandFrame."""
        wrist = frame.wrist

        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.pose.position.x = wrist.x
        msg.pose.position.y = wrist.y
        msg.pose.position.z = wrist.z
        msg.pose.orientation.x = wrist.qx
        msg.pose.orientation.y = wrist.qy
        msg.pose.orientation.z = wrist.qz
        msg.pose.orientation.w = wrist.qw

        self._wrist_pubs[frame.side].publish(msg)

        self.get_logger().debug(
            f"{frame.side.value} wrist: pos=({wrist.x:.3f}, {wrist.y:.3f}, {wrist.z:.3f})"
        )

    def _publish_landmarks(self, frame, sec: int, nanosec: int) -> None:
        """Publish all joint landmarks from frame."""
        for joint in JointName:
            try:
                x, y, z = frame.get_joint(joint)

                msg = PointStamped()
                msg.header.frame_id = "world"
                msg.header.stamp.sec = sec
                msg.header.stamp.nanosec = nanosec
                msg.point.x = x
                msg.point.y = y
                msg.point.z = z

                self._landmark_pubs[(frame.side, joint)].publish(msg)

            except Exception as e:
                self.get_logger().warning(f"Failed to get joint {joint}: {e}")

    def _publish_pinch_distance(self, frame) -> None:
        """Calculate and publish pinch distance (thumb tip to index tip)."""
        try:
            # Get thumb tip and index tip positions
            thumb_x, thumb_y, thumb_z = frame.get_joint(JointName.THUMB_TIP)
            index_x, index_y, index_z = frame.get_joint(JointName.INDEX_TIP)

            # Calculate Euclidean distance
            distance_m = math.sqrt(
                (thumb_x - index_x) ** 2 +
                (thumb_y - index_y) ** 2 +
                (thumb_z - index_z) ** 2
            )
            distance_cm = distance_m * 100  # Convert to centimeters

            msg = Float32()
            msg.data = distance_cm
            self._pinch_distance_pubs[frame.side].publish(msg)

            self.get_logger().debug(
                f"{frame.side.value} pinch distance: {distance_cm:.2f} cm"
            )

        except Exception as e:
            self.get_logger().warning(f"Failed to calculate pinch distance: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tcp_wireless_ros",
        description="HTS TCP listener using official hand_tracking_sdk library",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP to bind to (default: 0.0.0.0 — all interfaces)",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="TCP timeout in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    rclpy.init()
    node = HTSPosePublisher(args.host, args.port, args.timeout)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
