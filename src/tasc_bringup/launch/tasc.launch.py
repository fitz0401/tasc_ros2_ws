from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # TASC
    tasc_node = Node(
        package='tasc_bringup',
        executable='tasc_node',
        name='tasc_main_node',
        output='screen',
        parameters=[{}],
        remappings=[
            ('~/delta_twist_cmds', '/tasc/delta_twist_cmds'),
        ]
    )

    return LaunchDescription([
        tasc_node
    ])
