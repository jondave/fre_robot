from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Paths to the other launch files
    ld19_launch_path = os.path.join(
        get_package_share_directory('fre_robot_bringup'),
        'launch',
        'ld19.launch.py'
    )

    imu_launch_path = os.path.join(
        get_package_share_directory('fre_robot_bringup'),
        'launch',
        'imu.launch.py'
    )

    # Include the launch files
    ld19_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ld19_launch_path)
    )

    imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(imu_launch_path)
    )

    # Launch description
    ld = LaunchDescription()

    # Add included launch files
    ld.add_action(ld19_launch)
    ld.add_action(imu_launch)

    return ld
