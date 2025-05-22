# fre_robot

## ROS2 Humble

## fre_robot_bringup launch files
`ros2 launch fre_robot_bringup` then;
- `zero.launch.py` - launches the robot driver without the joystick node.
- `zero_teleop.launch.py` - launches the robot driver with the joystick node.
- `imu.launch.py` - launches the IMU.
- `ld19.launch.py` - launches the lidar.
- `realsense_1.launch.py`, `realsense_2.launch.py` and `realsense_3.launch.py` - launch the realsense cameras, make sure the TF link and serial number in the launch files match the cameras on the robot.
