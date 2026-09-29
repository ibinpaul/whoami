"""Tests for config/robots/costmap_node.template.yaml (needs rclpy; skipped without ROS).

The template must list exactly the node's parameters, keep only the
confirmed Dev 1 topics, and leave every other value as a placeholder that
stops the node from starting.
"""

import pathlib

import pytest

yaml = pytest.importorskip("yaml")
rclpy = pytest.importorskip("rclpy")
pytest.importorskip("tf2_ros")

from rclpy.exceptions import InvalidParameterTypeException  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402

from costmap_ros.costmap_node import (  # noqa: E402
    FOOTPRINT_PARAMETERS,
    GEOMETRY_PARAMETERS,
    NODE_NAME,
    PARAMETERS,
    CostmapNode,
)

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE = PACKAGE_DIR / "config" / "robots" / "costmap_node.template.yaml"
PLACEHOLDER = ["TODO"]
ALL_PARAMETERS = PARAMETERS + GEOMETRY_PARAMETERS + FOOTPRINT_PARAMETERS

# Dev 1 publishers (turing/src/ugv_perception/node/adapter_node.py).
CONFIRMED = {
    "mask_topic": "/segmentation/mask",
    "camera_info_topic": "/segmentation/camera_info",
    "geometry.depth_cloud_topic": "/perception/depth_cloud",
}


def _flatten(tree, prefix=""):
    flat = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix + key + "."))
        else:
            flat[prefix + key] = value
    return flat


def _template_parameters():
    document = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    assert list(document) == [NODE_NAME]
    assert list(document[NODE_NAME]) == ["ros__parameters"]
    return _flatten(document[NODE_NAME]["ros__parameters"])


def test_template_lists_exactly_the_node_parameters():
    assert sorted(_template_parameters()) == sorted(name for name, _ in ALL_PARAMETERS)


def test_only_confirmed_values_are_filled_in():
    for name, value in _template_parameters().items():
        assert value == CONFIRMED.get(name, PLACEHOLDER), name


def test_placeholder_never_matches_a_parameter_type():
    # [TODO] loads as a STRING_ARRAY, so declare_parameter rejects it for
    # every parameter, including string ones.
    assert all(t != Parameter.Type.STRING_ARRAY for _, t in ALL_PARAMETERS)


def test_unfilled_template_does_not_start_the_node():
    context = rclpy.Context()
    rclpy.init(context=context)
    try:
        with pytest.raises(InvalidParameterTypeException):
            CostmapNode(context=context, cli_args=["--ros-args", "--params-file", str(TEMPLATE)])
    finally:
        rclpy.shutdown(context=context)


def test_setup_installs_config_robots():
    setup_py = (PACKAGE_DIR / "setup.py").read_text(encoding="utf-8")
    assert "('share/' + package_name + '/config/robots', glob('config/robots/*.yaml'))" in setup_py
