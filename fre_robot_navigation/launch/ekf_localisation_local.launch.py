from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=['/home/mark1/ros2_ws/src/fre_robot/fre_robot_navigation/config/ekf_localisation_local.yaml']
        ),
    ])
