"""Tests for costmap_ros.grid_params (pure Python; runs without ROS)."""

import math

import pytest

from costmap_core.contracts import ContractError, GridInput
from costmap_core.grid import GridError
from costmap_ros.grid_params import GRID_PARAMETER_NAMES, grid_input_from_parameters

# Synthetic test values only; the real grid configuration is PENDING.
PARAMS = {
    "resolution": 0.5,
    "origin_x": -2.0,
    "origin_y": -3.0,
    "width": 8,
    "height": 12,
    "frame_id": "test_grid_frame",
}


def test_explicit_parameters_build_grid_input():
    grid = grid_input_from_parameters(PARAMS)
    assert isinstance(grid, GridInput)
    g = grid.geometry
    assert (g.resolution, g.origin_x, g.origin_y, g.width, g.height) == (0.5, -2.0, -3.0, 8, 12)
    assert grid.frame_id == "test_grid_frame"
    assert grid.shape == (12, 8)


def test_integer_valued_reals_accepted():
    grid = grid_input_from_parameters({**PARAMS, "origin_x": 0, "resolution": 1})
    assert grid.geometry.origin_x == 0.0 and isinstance(grid.geometry.origin_x, float)


@pytest.mark.parametrize("name", GRID_PARAMETER_NAMES)
def test_every_parameter_is_required(name):
    params = {k: v for k, v in PARAMS.items() if k != name}
    with pytest.raises(ContractError, match="missing"):
        grid_input_from_parameters(params)


def test_unknown_parameter_rejected():
    with pytest.raises(ContractError, match="unknown"):
        grid_input_from_parameters({**PARAMS, "inflation_radius": 1.0})


@pytest.mark.parametrize("name", ["resolution", "origin_x", "origin_y"])
@pytest.mark.parametrize("bad", ["0.5", None, True])
def test_non_numeric_reals_rejected(name, bad):
    with pytest.raises(ContractError, match=name):
        grid_input_from_parameters({**PARAMS, name: bad})


@pytest.mark.parametrize(
    "overrides",
    [
        {"resolution": 0.0},
        {"resolution": -0.1},
        {"resolution": math.nan},
        {"origin_x": math.inf},
        {"width": 0},
        {"height": -1},
        {"width": 8.0},
        {"height": True},
    ],
)
def test_invalid_grid_geometry_rejected(overrides):
    with pytest.raises(GridError):
        grid_input_from_parameters({**PARAMS, **overrides})


@pytest.mark.parametrize("frame_id", ["", None, 3])
def test_invalid_frame_id_rejected(frame_id):
    with pytest.raises(ContractError, match="frame_id"):
        grid_input_from_parameters({**PARAMS, "frame_id": frame_id})
