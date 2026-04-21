#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, PushROSNamespace
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Find the installed package directory
    package_share_dir = FindPackageShare("arm_vr")
    bridge_script = [package_share_dir, "/src/so101_bridge.py"]
    tcp_wireless_ros = [package_share_dir, "/src/tcp_wireless_ros.py"]
    urdf_file = [package_share_dir, "/urdf/so101_follower.urdf"]
    module_dir_default = [package_share_dir, "/src"]
    right_calibration_default = [package_share_dir, "/config/right_arm.json"]
    left_calibration_default = [package_share_dir, "/config/left_arm.json"]
    
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "module_dir",
                default_value=module_dir_default,
                description="Directory containing so101.py",
            ),
            DeclareLaunchArgument(
                "right_arm_port",
                default_value="/dev/ttyACM0",
                description="Serial port connected to SO101 right arm",
            ),
            DeclareLaunchArgument(
                "left_arm_port",
                default_value="/dev/ttyACM1",
                description="Serial port connected to SO101 left arm",
            ),
            DeclareLaunchArgument(
                "right_calibration_file",
                default_value=right_calibration_default,
                description="Right arm calibration JSON path, empty to disable calibrated conversion",
            ),
            DeclareLaunchArgument(
                "left_calibration_file",
                default_value=left_calibration_default,
                description="Left arm calibration JSON path, empty to disable calibrated conversion",
            ),
            DeclareLaunchArgument("baudrate", default_value="1000000"),
            DeclareLaunchArgument("serial_timeout", default_value="0.02"),
            DeclareLaunchArgument("reply_timeout", default_value="0.05"),
            DeclareLaunchArgument("feedback_rate_hz", default_value="20.0"),
            DeclareLaunchArgument(
                "host",
                default_value="192.168.123.102",
                description="Host IP for wireless TCP connection",
            ),
            # Right arm bridge in its own namespace
            GroupAction(
                actions=[
                    PushROSNamespace("right_arm"),
                    ExecuteProcess(
                        cmd=[
                            "python3",
                            bridge_script,
                            "--module-dir",
                            LaunchConfiguration("module_dir"),
                            "--port",
                            LaunchConfiguration("right_arm_port"),
                            "--calibration-file",
                            LaunchConfiguration("right_calibration_file"),
                            "--baudrate",
                            LaunchConfiguration("baudrate"),
                            "--serial-timeout",
                            LaunchConfiguration("serial_timeout"),
                            "--reply-timeout",
                            LaunchConfiguration("reply_timeout"),
                            "--feedback-rate-hz",
                            LaunchConfiguration("feedback_rate_hz"),
                        ],
                        output="screen",
                        emulate_tty=True,
                    ),
                ]
            ),
            # Left arm bridge in its own namespace
            GroupAction(
                actions=[
                    PushROSNamespace("left_arm"),
                    ExecuteProcess(
                        cmd=[
                            "python3",
                            bridge_script,
                            "--module-dir",
                            LaunchConfiguration("module_dir"),
                            "--port",
                            LaunchConfiguration("left_arm_port"),
                            "--calibration-file",
                            LaunchConfiguration("left_calibration_file"),
                            "--baudrate",
                            LaunchConfiguration("baudrate"),
                            "--serial-timeout",
                            LaunchConfiguration("serial_timeout"),
                            "--reply-timeout",
                            LaunchConfiguration("reply_timeout"),
                            "--feedback-rate-hz",
                            LaunchConfiguration("feedback_rate_hz"),
                        ],
                        output="screen",
                        emulate_tty=True,
                    ),
                ]
            ),
            # TCP wireless ROS
            ExecuteProcess(
                cmd=[
                    "python3",
                    tcp_wireless_ros,
                    "--host",
                    LaunchConfiguration("host"),
                    "--port",
                    "8000"
                ],
                output="screen",
                emulate_tty=True,
            ),
        ]
    )