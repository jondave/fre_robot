# fre_robot

## ROS2 Humble

## fre_robot_bringup launch files
`ros2 launch fre_robot_bringup` then;
- `zero.launch.py` - launches the robot driver without the joystick node.
- `zero_teleop.launch.py` - launches the robot driver with the joystick node.
- `imu.launch.py` - launches the IMU.
- `ld19.launch.py` - launches the lidar.
- `realsense_1.launch.py`, `realsense_2.launch.py` and `realsense_3.launch.py` - launch the realsense cameras, make sure the TF link and serial number in the launch files match the cameras on the robot.

## Robot Bringup

To start the robot driver without joystick controller
```
ros2 launch fre_robot_bringup zero.launch.py
```

To start the robot driver with joystick controller
```
ros2 launch fre_robot_bringup zero_teleop.launch.py
```

To start IMU and Lidar
```
ros2 launch fre_robot_bringup accessories.launch.py
```

## Robot Navigation

### Localisation
To start local localisation (base_link to odom frame) launch the EKF node, this fuses the wheel odom and IMU data.
```
ros2 launch fre_robot_navigation ekf_localisation_local.launch.py
```

### SLAM and Nav2

To navigate without a map launch slam toolbox (to build a map) and Nav2.
```
ros2 launch roverrobotics_driver slam_launch.py
ros2 launch nav2_bringup navigation_launch.py slam:=True # default params

ros2 launch nav2_bringup navigation_launch.py slam:=True params_file:=/home/mark1/ros2_ws/src/fre_robot/fre_robot_navigation/config/nav2_params.yaml
```

To navigate with a saved map launch AMCL with global EKF localisation and map server and Nav2
```
ros2 launch fre_robot_navigation amcl.launch.py 
ros2 launch nav2_bringup navigation_launch.py
```

# Task 3
Launch files
```
ros2 launch nav2_bringup navigation_launch.py params_file:=$HOME/rover_workspace/src/roverrobotics_ros2/roverrobotics_driver/config/nav2_params.yaml
```

Rover driver EKF
```
ros2 launch roverrobotics_driver robot_localizer.launch.py
```

```
ros2 launch nav2_bringup navigation_launch.py params_file:=$HOME/new_nav2_params.yaml
```

