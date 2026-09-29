"""Per-costmap configuration: several independent costmap configurations.

The project intends two Dev 3 outputs, a global and a local costmap
(DEV3_DEV4_INTERFACE.md §3), but how they differ (frame, extent, update
behaviour, rolling or not) is NOT YET DEFINED (INTEGRATION_CHECKLIST §1).
So there is no global/local type in the code. One costmap is configured by
an existing GridInput (frame_id, resolution, origin_x, origin_y, width,
height) plus an inflation_radius, and these tests show that two such
configurations can be built and used independently.

"global"/"local" below are only test labels. Every frame name and number is
synthetic, not a project value.
"""

import ast
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from costmap_core.contracts import ContractError, GridInput
from costmap_core.grid import CostmapGridGeometry, GridError
from costmap_core.inflation import InflationError, inflate_costmap
from costmap_ros.grid_params import GRID_PARAMETER_NAMES, grid_input_from_parameters

LETHAL = 254

GLOBAL_LIKE = {
    "resolution": 0.5,
    "origin_x": -20.0,
    "origin_y": -15.0,
    "width": 80,
    "height": 60,
    "frame_id": "test_frame_a",
}
LOCAL_LIKE = {
    "resolution": 0.05,
    "origin_x": -1.5,
    "origin_y": -2.0,
    "width": 60,
    "height": 80,
    "frame_id": "test_frame_b",
}
GLOBAL_LIKE_RADIUS = 1.0
LOCAL_LIKE_RADIUS = 0.2

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("costmap_core", "costmap_ros")


def _geometry_tuple(grid: GridInput):
    g = grid.geometry
    return (g.resolution, g.origin_x, g.origin_y, g.width, g.height)


# --- valid configurations ---------------------------------------------------


@pytest.mark.parametrize("params", [GLOBAL_LIKE, LOCAL_LIKE], ids=["global", "local"])
def test_valid_configuration_keeps_every_value(params):
    grid = grid_input_from_parameters(params)
    assert grid.frame_id == params["frame_id"]
    assert _geometry_tuple(grid) == (
        params["resolution"], params["origin_x"], params["origin_y"], params["width"], params["height"],
    )
    assert grid.shape == (params["height"], params["width"])


@pytest.mark.parametrize("radius", [0.0, 0.01, GLOBAL_LIKE_RADIUS, LOCAL_LIKE_RADIUS])
def test_valid_inflation_radius_accepted_by_inflation(radius):
    costs = np.zeros((5, 5), dtype=np.int64)
    costs[2, 2] = LETHAL
    out = inflate_costmap(costs, 0.1, radius)
    assert out[2, 2] == LETHAL


# --- invalid values -----------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -0.1, math.nan, math.inf, -math.inf])
def test_invalid_resolution_rejected(bad):
    with pytest.raises(GridError, match="resolution"):
        grid_input_from_parameters({**LOCAL_LIKE, "resolution": bad})


@pytest.mark.parametrize("bad", [True, "0.1", None])
def test_non_numeric_resolution_rejected(bad):
    with pytest.raises(ContractError, match="resolution"):
        grid_input_from_parameters({**LOCAL_LIKE, "resolution": bad})


@pytest.mark.parametrize("name", ["width", "height"])
@pytest.mark.parametrize("bad", [0, -1, 1.5, 10.0, True, "10", None])
def test_invalid_dimensions_rejected(name, bad):
    with pytest.raises(GridError, match=name):
        grid_input_from_parameters({**GLOBAL_LIKE, name: bad})


@pytest.mark.parametrize("name", ["origin_x", "origin_y"])
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_origin_rejected(name, bad):
    with pytest.raises(GridError, match="origin"):
        grid_input_from_parameters({**GLOBAL_LIKE, name: bad})


@pytest.mark.parametrize("bad", ["", None, 0, b"frame"])
def test_invalid_frame_rejected(bad):
    with pytest.raises(ContractError, match="frame_id"):
        grid_input_from_parameters({**LOCAL_LIKE, "frame_id": bad})


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, True, "0.5", None])
def test_invalid_inflation_radius_rejected_by_inflation(bad):
    with pytest.raises(InflationError, match="inflation_radius"):
        inflate_costmap(np.zeros((3, 3), dtype=np.int64), 0.1, bad)


# --- independence -------------------------------------------------------------


def test_two_configurations_are_independent():
    a = grid_input_from_parameters(dict(GLOBAL_LIKE))
    b = grid_input_from_parameters(dict(LOCAL_LIKE))
    assert a.frame_id != b.frame_id
    assert _geometry_tuple(a) != _geometry_tuple(b)
    # Building b did not change a, and neither can be modified afterwards.
    assert a == grid_input_from_parameters(GLOBAL_LIKE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.frame_id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.geometry.width = 1


def test_same_frame_different_geometry_is_allowed():
    """Nothing ties a frame to one extent: sharing or splitting frames is a team decision."""
    a = grid_input_from_parameters(GLOBAL_LIKE)
    b = grid_input_from_parameters({**LOCAL_LIKE, "frame_id": GLOBAL_LIKE["frame_id"]})
    assert a.frame_id == b.frame_id
    assert _geometry_tuple(a) != _geometry_tuple(b)


def test_inflation_radius_is_per_configuration():
    costs = np.zeros((21, 21), dtype=np.int64)
    costs[10, 10] = LETHAL
    small = inflate_costmap(costs, 0.1, LOCAL_LIKE_RADIUS)
    large = inflate_costmap(costs, 0.1, GLOBAL_LIKE_RADIUS)
    assert np.count_nonzero(small) < np.count_nonzero(large)
    np.testing.assert_array_equal(costs[10, 10], LETHAL)  # input untouched


# --- no hard-coded frames or defaults ------------------------------------------


def test_grid_parameters_have_no_defaults():
    with pytest.raises(ContractError) as excinfo:
        grid_input_from_parameters({})
    for name in GRID_PARAMETER_NAMES:
        assert name in str(excinfo.value)


@pytest.mark.parametrize("cls", [GridInput, CostmapGridGeometry])
def test_grid_dataclasses_have_no_field_defaults(cls):
    for field in dataclasses.fields(cls):
        assert field.default is dataclasses.MISSING, field.name
        assert field.default_factory is dataclasses.MISSING, field.name


FORBIDDEN_EXACT = {"map", "odom", "base_link", "local_costmap", "global_costmap"}
FORBIDDEN_SUBSTRINGS = ("base_link", "local_costmap", "global_costmap")


def _docstring_nodes(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                yield body[0].value


def _string_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {id(n) for n in _docstring_nodes(tree)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value


def _source_files():
    files = [p for d in SOURCE_DIRS for p in sorted((PACKAGE_ROOT / d).glob("*.py"))]
    assert files, "no source files found"
    return files


def test_no_hard_coded_frame_or_costmap_names_in_code():
    """Docstrings and comments may mention the documented example frames;
    code string literals may not."""
    offenders = []
    for path in _source_files():
        for lineno, value in _string_literals(path):
            stripped = value.strip("/")
            if stripped in FORBIDDEN_EXACT or any(s in value for s in FORBIDDEN_SUBSTRINGS):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{lineno}: {value!r}")
    assert offenders == []


# --- node-level (needs rclpy) ---------------------------------------------------


def _node_params(grid_params, radius):
    return {
        "mask_topic": "/test/mask",
        "camera_info_topic": "/test/camera_info",
        "costmap_topic": "/test/costmap",
        "target_frame": grid_params["frame_id"],
        "grid.resolution": grid_params["resolution"],
        "grid.origin_x": grid_params["origin_x"],
        "grid.origin_y": grid_params["origin_y"],
        "grid.width": grid_params["width"],
        "grid.height": grid_params["height"],
        "tf_timeout_s": 0.1,
        "inflation_radius": radius,
        "camera_info_durability": "volatile",
        "mask_max_age_s": 0.5,
        "output_stamp_source": "mask",
        "geometry.enabled": False,
        "fail_safe.roi_min_x": 0.0,
        "fail_safe.roi_max_x": 1.0,  # synthetic; inside both grids above
        "fail_safe.roi_min_y": -1.0,
        "fail_safe.roi_max_y": 1.0,
        "fail_safe.check_period_s": 0.1,
        "footprint.enabled": False,
    }


def _costmap_node():
    pytest.importorskip("rclpy")
    pytest.importorskip("tf2_ros")
    from costmap_ros import costmap_node

    return costmap_node


@pytest.mark.parametrize(
    "grid_params, radius",
    [(GLOBAL_LIKE, GLOBAL_LIKE_RADIUS), (LOCAL_LIKE, LOCAL_LIKE_RADIUS)],
    ids=["global", "local"],
)
def test_node_config_accepts_each_configuration(grid_params, radius):
    node = _costmap_node()
    config = node.config_from_parameters(_node_params(grid_params, radius))
    assert config.target_frame == grid_params["frame_id"]
    assert config.grid == grid_input_from_parameters(grid_params)
    assert config.inflation_radius == radius


def test_node_configs_are_independent_and_output_their_own_grid():
    node = _costmap_node()
    pytest.importorskip("nav_msgs.msg")
    from builtin_interfaces.msg import Time

    from costmap_ros.occupancy_grid import occupancy_grid_from_costmap

    a = node.config_from_parameters(_node_params(GLOBAL_LIKE, GLOBAL_LIKE_RADIUS))
    b = node.config_from_parameters(_node_params(LOCAL_LIKE, LOCAL_LIKE_RADIUS))
    assert (a.target_frame, a.inflation_radius) == (GLOBAL_LIKE["frame_id"], GLOBAL_LIKE_RADIUS)
    assert (b.target_frame, b.inflation_radius) == (LOCAL_LIKE["frame_id"], LOCAL_LIKE_RADIUS)
    for config, params in ((a, GLOBAL_LIKE), (b, LOCAL_LIKE)):
        msg = occupancy_grid_from_costmap(
            np.zeros(config.grid.shape, dtype=np.int64), config.grid, stamp=Time(sec=1)
        )
        assert msg.header.frame_id == params["frame_id"]
        assert (msg.info.width, msg.info.height) == (params["width"], params["height"])
        assert msg.info.resolution == pytest.approx(params["resolution"])
        assert (msg.info.origin.position.x, msg.info.origin.position.y) == (
            params["origin_x"], params["origin_y"],
        )


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, True, "0.5"])
def test_node_config_rejects_invalid_inflation_radius(bad):
    node = _costmap_node()
    with pytest.raises(ContractError, match="inflation_radius"):
        node.config_from_parameters(_node_params(LOCAL_LIKE, bad))


@pytest.mark.parametrize(
    "override, match",
    [
        ({"grid.resolution": 0.0}, "resolution"),
        ({"grid.width": 0}, "width"),
        ({"grid.height": -3}, "height"),
        ({"target_frame": ""}, "frame_id"),
    ],
)
def test_node_config_rejects_invalid_grid(override, match):
    node = _costmap_node()
    with pytest.raises((ContractError, GridError), match=match):
        node.config_from_parameters({**_node_params(GLOBAL_LIKE, GLOBAL_LIKE_RADIUS), **override})


def test_node_config_has_no_field_defaults():
    node = _costmap_node()
    for field in dataclasses.fields(node.CostmapNodeConfig):
        assert field.default is dataclasses.MISSING, field.name
        assert field.default_factory is dataclasses.MISSING, field.name
