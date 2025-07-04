from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Camera
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                FindPackageShare('realsense2_camera').find('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ]),
        launch_arguments={
            'depth_module.depth_profile': '1280x720x30',
            'rgb_camera.color_profile': '1280x720x30',
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
        }.items()
    )

    # TF: fr3_link0 → ref_frame
    tf1 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_fr3_to_ref',
        arguments=[
            '0.735097', '0.646207', '0.664998',    # translation (x y z)
            '-0.370981', '-0.863263', '0.320188', '0.120952',  # quaternion (x y z w)
            'fr3_link0', 'ref_frame'
        ]
    )

    # TF: ref_frame → camera_link
    tf2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_ref_to_cam',
        arguments=[
            '-0.0010067', '0.014068', '-0.002151',
            '0.49272', '-0.49219', '0.50886', '0.50601',
            'ref_frame', 'camera_link'
        ]
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(
            FindPackageShare('aanp_bringup').find('aanp_bringup'),
            'rviz', 'camera_view.rviz'
        )],
        output='screen'
    )

    # Joy
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node'
    )

    # AANP
    aanp_node = Node(
        package='aanp_bringup',
        executable='aanp_node',
        name='aanp_main_node',
        output='screen',
        parameters=[{}],
        remappings=[
            ('~/delta_twist_cmds', '/aanp/delta_twist_cmds'),
        ]
    )

    return LaunchDescription([
        # tf1,
        # tf2,
        # realsense_launch,
        # joy_node,
        aanp_node,
        # rviz_node,
    ])
