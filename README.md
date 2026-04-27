# ARM VR - SO101 Robot Control

A ROS2 package for controlling dual SO101 robot arms with wireless TCP communication and calibration support.

## Installation

Create the conda environment with all dependencies:

```bash
conda env create -f environment.yml

ros2 launch arm_vr so101_bridge.launch.py \
  right_arm_port:=/dev/ttyACM0 \
  left_arm_port:=/dev/ttyACM1 \
  host:=0.0.0.0
```

## Configuration

The launch file supports the following parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `module_dir` | `src/` | Directory containing so101.py |
| `right_arm_port` | `/dev/ttyACM0` | Serial port for right arm |
| `left_arm_port` | `/dev/ttyACM1` | Serial port for left arm |
| `right_calibration_file` | `config/right_arm.json` | Right arm calibration file |
| `left_calibration_file` | `config/left_arm.json` | Left arm calibration file |
| `baudrate` | `1000000` | Serial communication speed |
| `serial_timeout` | `0.02` | Serial read timeout (seconds) |
| `reply_timeout` | `0.05` | Response timeout (seconds) |
| `feedback_rate_hz` | `50.0` | Feedback frequency (Hz) |
| `host` | `0.0.0.0` | Host IP for wireless TCP connection |
