#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Find the installed package directory
    package_share_dir = FindPackageShare("arm_vr")
    bridge_script = [package_share_dir, "/src/so101_bridge.py"]
    tcp_wireless_ros = [package_share_dir, "/src/tcp_wireless_ros.py"]
    urdf_file = [package_share_dir, "/urdf/so101_follower.urdf"]
    module_dir_default = [package_share_dir, "/src"]
    calibration_file_default = [package_share_dir, "/config/right_arm.json"]
    
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "module_dir",
                default_value=module_dir_default,
                description="Directory containing so101.py",
            ),
            DeclareLaunchArgument(
                "port",
                default_value="/dev/ttyACM0",
                description="Serial port connected to SO101 arm",
            ),
            DeclareLaunchArgument(
                "calibration_file",
                default_value=calibration_file_default,
                description="Calibration JSON path, empty to disable calibrated conversion",
            ),
            DeclareLaunchArgument("baudrate", default_value="1000000"),
            DeclareLaunchArgument("serial_timeout", default_value="0.02"),
            DeclareLaunchArgument("reply_timeout", default_value="0.05"),
            DeclareLaunchArgument("feedback_rate_hz", default_value="20.0"),
            ExecuteProcess(
                cmd=[
                    "python3",
                    bridge_script,
                    "--module-dir",
                    LaunchConfiguration("module_dir"),
                    "--port",
                    LaunchConfiguration("port"),
                    "--calibration-file",
                    LaunchConfiguration("calibration_file"),
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
            ExecuteProcess(
                cmd=[
                    "python3",
                    tcp_wireless_ros,
                    "--host",
                    "192.168.123.102",
                    "--port",
                    "8000"
                ],
                output="screen",
                emulate_tty=True,
            ),
        ]
    )