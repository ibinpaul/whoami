"""Launch the Dev 3 costmap node (costmap_ros.costmap_node).

Every parameter of costmap_node is required and has no default (topics,
frames, grid, timeouts, inflation radius, fail-safe ROI, footprint, ...;
see costmap_node.PARAMETERS). The real values are PENDING (Dev 2 / Dev 5 /
team decisions), so this launch file supplies none of them: the caller
passes a ROS 2 parameters YAML with `params_file`, which is itself required.

    ros2 launch ugv_navigation costmap.launch.py params_file:=/path/to/costmap.yaml

The YAML is keyed by the node name, `costmap_node`, or by `/**`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            description="ROS 2 parameters YAML for costmap_node (required, no default).",
        ),
        Node(
            package="ugv_navigation",
            executable="costmap_node",
            name="costmap_node",
            output="screen",
            parameters=[params_file],
        ),
    ])
