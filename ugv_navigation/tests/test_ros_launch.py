"""Tests for launch/costmap.launch.py (needs launch/launch_ros; skipped without ROS)."""

import importlib.util
import pathlib

import pytest

pytest.importorskip("launch")
pytest.importorskip("launch_ros")

from launch.actions import DeclareLaunchArgument  # noqa: E402
from launch_ros.actions import Node  # noqa: E402

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent.parent
LAUNCH_FILE = PACKAGE_DIR / "launch" / "costmap.launch.py"


def _launch_description():
    spec = importlib.util.spec_from_file_location("costmap_launch", LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def test_params_file_is_required_without_default():
    args = [e for e in _launch_description().entities if isinstance(e, DeclareLaunchArgument)]
    assert [a.name for a in args] == ["params_file"]
    assert args[0].default_value is None


def test_launches_the_existing_costmap_node_executable():
    nodes = [e for e in _launch_description().entities if isinstance(e, Node)]
    assert len(nodes) == 1
    assert nodes[0].node_package == "ugv_navigation"
    assert nodes[0].node_executable == "costmap_node"
    # The executable must be the console script declared in setup.py.
    assert "'costmap_node = costmap_ros.costmap_node:main'" in (
        PACKAGE_DIR / "setup.py"
    ).read_text(encoding="utf-8")


def test_setup_installs_launch_directory():
    setup_py = (PACKAGE_DIR / "setup.py").read_text(encoding="utf-8")
    assert "('share/' + package_name + '/launch', glob('launch/*.launch.py'))" in setup_py
