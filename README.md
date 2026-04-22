conda env create -f environment.yml

ros2 launch arm_vr so101_bridge.launch.py \
  right_arm_port:=/dev/ttyACM0 \
  left_arm_port:=/dev/ttyACM1 \
  host:=192.168.123.102
