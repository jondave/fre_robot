from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Map Server
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': '/home/mark1/ros2_ws/src/fre_robot/fre_robot_navigation/maps/map.yaml',
                'use_sim_time': False
            }]
        ),

        # AMCL
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=['/home/mark1/ros2_ws/src/fre_robot/fre_robot_navigation/config/amcl.yaml']
        ),

        # Global EKF Localization Node
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=['/home/mark1/ros2_ws/src/fre_robot/fre_robot_navigation/config/ekf_localisation_global.yaml']
        ),

        # Lifecycle Manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'node_names': ['map_server', 'amcl']
            }]
        ),

        # # Static transform publisher: base_footprint -> base_laser
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='static_transform_base_footprint_to_base_laser',
        #     arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'base_laser']
        # )
    ])
