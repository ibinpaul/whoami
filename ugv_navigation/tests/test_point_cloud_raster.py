"""Tests for costmap_core.point_cloud_raster.

Every grid geometry, point, stamp and frame name here is SYNTHETIC. The
real grid resolution/extent and costmap frame are PENDING.
"""

import numpy as np
import pytest

from costmap_core import grid as grid_module
from costmap_core.contracts import (
    ContractError,
    GridInput,
    OccupancyInput,
    PointCloudInput,
)
from costmap_core.grid import CostmapGridGeometry, GridError, world_to_grid_cell
from costmap_core.point_cloud_raster import rasterize_obstacle_points

GROUND = "test_ground_frame"
STAMP = 11_000_000_007

# Synthetic 0.5 m grid, 4 cols x 3 rows, covering x in [0, 2), y in [0, 1.5).
RES = 0.5
W, H = 4, 3


def _grid(*, resolution=RES, origin_x=0.0, origin_y=0.0, width=W, height=H, frame_id=GROUND):
    return GridInput(
        geometry=CostmapGridGeometry(
            resolution=resolution, origin_x=origin_x, origin_y=origin_y, width=width, height=height
        ),
        frame_id=frame_id,
    )


def _cloud(points, *, frame_id=GROUND, stamp_ns=STAMP, dtype=np.float64):
    return PointCloudInput(
        points=np.asarray(points, dtype=dtype).reshape(-1, 3), stamp_ns=stamp_ns, frame_id=frame_id
    )


def _occupied_cells(result):
    return sorted(map(tuple, np.argwhere(result.occupied).tolist()))


def _reference(points, grid):
    """Per-point loop over grid.world_to_grid_cell, skipping GridError."""
    expected = np.zeros((grid.geometry.height, grid.geometry.width), dtype=bool)
    for x, y, _z in np.asarray(points, dtype=np.float64):
        try:
            cell = world_to_grid_cell(float(x), float(y), grid.geometry)
        except GridError:
            continue
        expected[cell.row, cell.col] = True
    return expected


# --- cell mapping ----------------------------------------------------------------


def test_one_point_maps_to_correct_cell():
    # x = 1.2 -> col 2, y = 0.7 -> row 1
    result = rasterize_obstacle_points(_cloud([[1.2, 0.7, 1.0]]), _grid())
    assert _occupied_cells(result) == [(1, 2)]


def test_multiple_points_different_cells():
    points = [[0.1, 0.1, 1.0], [1.9, 1.4, 1.0], [0.6, 1.1, 1.0]]
    result = rasterize_obstacle_points(_cloud(points), _grid())
    assert _occupied_cells(result) == [(0, 0), (2, 1), (2, 3)]


def test_multiple_points_same_cell_marked_once():
    points = [[1.1, 0.6, 0.5], [1.2, 0.7, 1.0], [1.49, 0.99, 1.5], [1.1, 0.6, 0.5]]
    result = rasterize_obstacle_points(_cloud(points), _grid())
    assert _occupied_cells(result) == [(1, 2)]
    assert result.occupied.dtype == np.bool_


def test_z_is_ignored():
    result = rasterize_obstacle_points(_cloud([[0.1, 0.1, -3.0], [0.1, 0.1, 99.0]]), _grid())
    assert _occupied_cells(result) == [(0, 0)]


def test_point_exactly_on_interior_boundary_goes_to_upper_cell():
    # x = 0.5 is the lower edge of col 1 (half-open cells).
    result = rasterize_obstacle_points(_cloud([[0.5, 1.0, 1.0]]), _grid())
    assert _occupied_cells(result) == [(2, 1)]


def test_point_on_grid_origin_is_inside():
    result = rasterize_obstacle_points(_cloud([[0.0, 0.0, 1.0]]), _grid())
    assert _occupied_cells(result) == [(0, 0)]


def test_point_just_inside_upper_boundary():
    x = np.nextafter(2.0, -np.inf)
    y = np.nextafter(1.5, -np.inf)
    result = rasterize_obstacle_points(_cloud([[x, y, 1.0]]), _grid())
    assert _occupied_cells(result) == [(H - 1, W - 1)]


@pytest.mark.parametrize(
    "point",
    [
        [2.0, 0.1, 1.0],                              # on the upper x edge (excluded)
        [0.1, 1.5, 1.0],                              # on the upper y edge (excluded)
        [np.nextafter(0.0, -np.inf), 0.1, 1.0],       # just below origin_x
        [0.1, np.nextafter(0.0, -np.inf), 1.0],       # just below origin_y
    ],
)
def test_point_just_outside_boundary_is_dropped(point):
    result = rasterize_obstacle_points(_cloud([point]), _grid())
    assert not result.occupied.any()


def test_out_of_grid_points_dropped_in_grid_points_kept():
    points = [[-5.0, 0.1, 1.0], [0.1, 0.1, 1.0], [10.0, 10.0, 1.0], [1.9, -0.1, 1.0]]
    result = rasterize_obstacle_points(_cloud(points), _grid())
    assert _occupied_cells(result) == [(0, 0)]


def test_all_points_outside_grid():
    points = [[-1.0, -1.0, 1.0], [5.0, 0.5, 1.0], [0.5, 5.0, 1.0]]
    result = rasterize_obstacle_points(_cloud(points), _grid())
    assert result.occupied.shape == (H, W)
    assert not result.occupied.any()


def test_empty_cloud_gives_all_false_grid():
    result = rasterize_obstacle_points(_cloud(np.empty((0, 3))), _grid())
    assert isinstance(result, OccupancyInput)
    assert result.occupied.shape == (H, W)
    assert not result.occupied.any()
    assert result.stamp_ns == STAMP
    assert result.frame_id == GROUND


def test_output_shape_is_grid_height_width():
    result = rasterize_obstacle_points(_cloud([[0.1, 0.1, 1.0]]), _grid(width=7, height=5))
    assert result.occupied.shape == (5, 7)


@pytest.mark.parametrize("resolution", [0.05, 0.1, 0.25, 1.0, 2.0])
def test_different_resolutions(resolution):
    grid = _grid(resolution=resolution, width=10, height=10)
    x, y = 3.3 * resolution, 7.8 * resolution  # col 3, row 7
    result = rasterize_obstacle_points(_cloud([[x, y, 1.0]]), grid)
    assert _occupied_cells(result) == [(7, 3)]


def test_non_zero_origin_and_negative_world_coordinates():
    # Grid covers x in [-3, -1), y in [-2, -0.5).
    grid = _grid(origin_x=-3.0, origin_y=-2.0)
    points = [
        [-3.0, -2.0, 1.0],    # origin corner -> (0, 0)
        [-1.2, -0.6, 1.0],    # col 3, row 2
        [-2.4, -1.4, 1.0],    # col 1, row 1
        [-0.9, -1.0, 1.0],    # x past the grid -> dropped
        [0.1, 0.1, 1.0],      # positive world coords outside this grid
    ]
    result = rasterize_obstacle_points(_cloud(points), grid)
    assert _occupied_cells(result) == [(0, 0), (1, 1), (2, 3)]


def test_positive_offset_origin():
    grid = _grid(origin_x=10.0, origin_y=20.0)
    result = rasterize_obstacle_points(_cloud([[10.6, 20.1, 1.0], [0.6, 0.1, 1.0]]), grid)
    assert _occupied_cells(result) == [(0, 1)]


def test_float32_points_match_scalar_helper():
    points = np.array([[0.49999997, 0.5, 1.0], [1.5, 1.0000001, 1.0]], dtype=np.float32)
    grid = _grid()
    result = rasterize_obstacle_points(_cloud(points, dtype=np.float32), grid)
    np.testing.assert_array_equal(result.occupied, _reference(points, grid))


# --- metadata and data safety ---------------------------------------------------


def test_stamp_and_frame_preserved():
    cloud = _cloud([[0.1, 0.1, 1.0]], frame_id="some_costmap_frame", stamp_ns=42_000_000_001)
    result = rasterize_obstacle_points(cloud, _grid(frame_id="some_costmap_frame"))
    assert result.stamp_ns == 42_000_000_001
    assert result.frame_id == "some_costmap_frame"


def test_input_not_mutated():
    cloud = _cloud([[0.1, 0.1, 1.0], [5.0, 5.0, 1.0]])
    before = cloud.points.copy()
    rasterize_obstacle_points(cloud, _grid())
    np.testing.assert_array_equal(cloud.points, before)


def test_output_readonly_and_independent_between_calls():
    grid = _grid()
    first = rasterize_obstacle_points(_cloud([[0.1, 0.1, 1.0]]), grid)
    second = rasterize_obstacle_points(_cloud([[1.9, 1.4, 1.0]]), grid)
    assert not first.occupied.flags.writeable
    with pytest.raises(ValueError):
        first.occupied[0, 0] = False
    assert not np.shares_memory(first.occupied, second.occupied)
    assert _occupied_cells(first) == [(0, 0)]


def test_result_feeds_existing_geometry_costmap_unchanged():
    from costmap_core.geometry_costmap import DEFAULT_GEOMETRY_COST_VALUES, build_geometry_costmap

    result = rasterize_obstacle_points(_cloud([[1.2, 0.7, 1.0]]), _grid())
    costmap = build_geometry_costmap(result.occupied)
    assert costmap[1, 2] == DEFAULT_GEOMETRY_COST_VALUES.lethal_cost
    assert np.count_nonzero(costmap == DEFAULT_GEOMETRY_COST_VALUES.lethal_cost) == 1


# --- scale / vectorisation --------------------------------------------------------


def test_large_cloud_matches_per_point_reference():
    rng = np.random.default_rng(3)
    grid = _grid(resolution=0.1, origin_x=-10.0, origin_y=-10.0, width=200, height=200)
    points = rng.uniform(-12.0, 12.0, size=(20_000, 3))
    result = rasterize_obstacle_points(_cloud(points), grid)
    np.testing.assert_array_equal(result.occupied, _reference(points, grid))


def test_camera_scale_cloud_does_not_use_scalar_helper(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("per-point world_to_grid_cell called")

    monkeypatch.setattr(grid_module, "world_to_grid_cell", forbidden)
    rng = np.random.default_rng(4)
    grid = _grid(resolution=0.1, origin_x=-20.0, origin_y=-20.0, width=400, height=400)
    points = rng.uniform(-25.0, 25.0, size=(640 * 480, 3))
    result = rasterize_obstacle_points(_cloud(points), grid)
    assert result.occupied.shape == (400, 400)
    assert result.occupied.any()


# --- rejection -------------------------------------------------------------------


def test_frame_mismatch_rejected():
    with pytest.raises(ContractError, match="frame_id"):
        rasterize_obstacle_points(_cloud([[0.1, 0.1, 1.0]], frame_id="optical"), _grid())


def test_wrong_types_rejected():
    with pytest.raises(ContractError, match="cloud"):
        rasterize_obstacle_points(np.zeros((1, 3)), _grid())
    with pytest.raises(ContractError, match="grid"):
        rasterize_obstacle_points(_cloud([[0.1, 0.1, 1.0]]), _grid().geometry)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(resolution=0.0),
        dict(resolution=-0.1),
        dict(resolution=float("nan")),
        dict(resolution=float("inf")),
        dict(width=0),
        dict(height=-1),
        dict(width=2.0),
        dict(height=True),
        dict(origin_x=float("nan")),
        dict(origin_y=float("inf")),
    ],
)
def test_invalid_grid_geometry_rejected_by_existing_rules(kwargs):
    with pytest.raises(GridError):
        _grid(**kwargs)


def test_empty_grid_frame_rejected():
    with pytest.raises(ContractError):
        _grid(frame_id="")


def test_invalid_points_rejected_by_point_cloud_input():
    with pytest.raises(ContractError, match="finite"):
        _cloud([[np.nan, 0.1, 1.0]])
