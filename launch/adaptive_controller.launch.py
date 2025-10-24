import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for adaptive ResNet controller"""

    # Get package directory
    pkg_share = get_package_share_directory('adaptive_resnet_controller')

    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock'
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'adaptive_controller_params.yaml']),
        description='Path to the configuration file'
    )

    enable_training_arg = DeclareLaunchArgument(
        'enable_training',
        default_value='false',
        description='Enable training on startup'
    )

    # Create node
    adaptive_controller_node = Node(
        package='adaptive_resnet_controller',
        executable='adaptive_controller',
        name='adaptive_resnet_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'training.enabled_on_start': LaunchConfiguration('enable_training'),
            }
        ],
        remappings=[
            # You can add topic remappings here if needed
            # ('/quadsim/odom', '/your_custom_odom_topic'),
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        config_file_arg,
        enable_training_arg,
        adaptive_controller_node,
    ])