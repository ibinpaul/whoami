"""Tests for costmap_ros.occupancy_grid (needs nav_msgs; skipped without ROS)."""

import numpy as np
import pytest

pytest.importorskip("nav_msgs.msg")

from builtin_interfaces.msg import Time  # noqa: E402

from costmap_core.class_to_cost import DEFAULT_COST_VALUES  # noqa: E402
from costmap_core.contracts import ContractError, GridInput  # noqa: E402
from costmap_core.geometry_costmap import DEFAULT_GEOMETRY_COST_VALUES  # noqa: E402
from costmap_core.grid import CostmapGridGeometry, world_to_grid_cell  # noqa: E402
from costmap_ros.occupancy_grid import (  # noqa: E402
    COST_TO_OCCUPANCY,
    costs_to_occupancy,
    occupancy_grid_from_costmap,
)

FRAME = "test_grid_frame"  # synthetic; the real frame name is PENDING


def _grid(*, width=4, height=3, resolution=0.25, origin_x=-1.5, origin_y=2.0, frame_id=FRAME):
    return GridInput(
        geometry=CostmapGridGeometry(
            resolution=resolution, origin_x=origin_x, origin_y=origin_y, width=width, height=height
        ),
        frame_id=frame_id,
    )


def _stamp(sec=12, nanosec=345):
    return Time(sec=sec, nanosec=nanosec)


@pytest.mark.parametrize(
    "cost, expected",
    [(0, 0), (1, 1), (126, 49), (252, 98), (253, 99), (254, 100), (255, -1)],
)
def test_nav2_cost_translation(cost, expected):
    assert costs_to_occupancy(np.array([[cost]]))[0, 0] == expected


def test_translation_is_monotonic_below_unknown():
    assert np.all(np.diff(COST_TO_OCCUPANCY[:255].astype(int)) >= 0)


def test_core_cost_values_map_to_expected_occupancy():
    costs = np.array([[
        DEFAULT_COST_VALUES.traversable_cost,
        DEFAULT_COST_VALUES.hazard_cost,
        DEFAULT_COST_VALUES.unknown_cost,
        DEFAULT_GEOMETRY_COST_VALUES.lethal_cost,
    ]])
    assert costs_to_occupancy(costs).tolist() == [[0, 100, -1, 100]]


def test_output_dtype_is_int8():
    assert costs_to_occupancy(np.zeros((2, 2), dtype=np.int64)).dtype == np.int8


@pytest.mark.parametrize("bad", [-1, 256, 1000])
def test_out_of_range_cost_rejected(bad):
    with pytest.raises(ContractError, match="0..255"):
        costs_to_occupancy(np.array([[0, bad]]))


@pytest.mark.parametrize(
    "costs", [np.zeros((2, 2), dtype=np.float64), np.zeros(4, dtype=np.int64), np.zeros((1, 2, 2), dtype=np.int64)]
)
def test_non_2d_integer_rejected(costs):
    with pytest.raises(ContractError, match="2D integer"):
        costs_to_occupancy(costs)


def test_message_header_and_info():
    grid = _grid()
    msg = occupancy_grid_from_costmap(np.zeros(grid.shape, dtype=np.int64), grid, stamp=_stamp())
    assert msg.header.frame_id == FRAME
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (12, 345)
    assert msg.info.map_load_time == msg.header.stamp
    assert msg.info.resolution == pytest.approx(0.25)
    assert (msg.info.width, msg.info.height) == (4, 3)
    assert (msg.info.origin.position.x, msg.info.origin.position.y) == (-1.5, 2.0)
    assert msg.info.origin.position.z == 0.0
    q = msg.info.origin.orientation
    assert (q.x, q.y, q.z, q.w) == (0.0, 0.0, 0.0, 1.0)


def test_data_is_row_major_with_row_as_y():
    grid = _grid()
    costs = np.zeros(grid.shape, dtype=np.int64)
    costs[1, 2] = 254  # row = y index 1, col = x index 2
    costs[2, 0] = 255
    msg = occupancy_grid_from_costmap(costs, grid, stamp=_stamp())
    data = list(msg.data)
    assert len(data) == 4 * 3
    assert data[1 * 4 + 2] == 100
    assert data[2 * 4 + 0] == -1
    assert sum(1 for v in data if v != 0) == 2


def test_shape_mismatch_rejected():
    grid = _grid()
    with pytest.raises(ContractError, match="does not match grid shape"):
        occupancy_grid_from_costmap(np.zeros((4, 3), dtype=np.int64), grid, stamp=_stamp())


def test_non_grid_input_rejected():
    with pytest.raises(ContractError, match="GridInput"):
        occupancy_grid_from_costmap(np.zeros((3, 4), dtype=np.int64), object(), stamp=_stamp())


# --- Output interface: every value comes from the caller (GridInput, stamp, costs). ---
# All grids, frames and stamps below are synthetic test values, not project values.

FREE = DEFAULT_COST_VALUES.traversable_cost
UNKNOWN = DEFAULT_COST_VALUES.unknown_cost
LETHAL = DEFAULT_GEOMETRY_COST_VALUES.lethal_cost


@pytest.mark.parametrize(
    "width, height, resolution, origin_x, origin_y",
    [
        (1, 1, 1.0, 0.0, 0.0),
        (7, 2, 0.05, -3.25, 10.5),  # wider than tall
        (2, 9, 0.4, 100.0, -0.125),  # taller than wide
        (5, 5, 0.1, -0.25, -0.25),
    ],
)
def test_metadata_preserved_for_any_grid(width, height, resolution, origin_x, origin_y):
    grid = _grid(width=width, height=height, resolution=resolution, origin_x=origin_x, origin_y=origin_y)
    msg = occupancy_grid_from_costmap(np.zeros(grid.shape, dtype=np.int64), grid, stamp=_stamp())
    assert (msg.info.width, msg.info.height) == (width, height)
    # info.resolution is float32 on the wire.
    assert msg.info.resolution == pytest.approx(resolution, rel=1e-6)
    assert (msg.info.origin.position.x, msg.info.origin.position.y) == (origin_x, origin_y)
    assert len(msg.data) == width * height


@pytest.mark.parametrize("sec, nanosec", [(0, 0), (0, 999_999_999), (2**31 - 1, 0), (1_700_000_000, 1)])
def test_stamp_copied_unchanged_to_header_and_map_load_time(sec, nanosec):
    grid = _grid()
    msg = occupancy_grid_from_costmap(np.zeros(grid.shape, dtype=np.int64), grid, stamp=_stamp(sec, nanosec))
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (sec, nanosec)
    assert (msg.info.map_load_time.sec, msg.info.map_load_time.nanosec) == (sec, nanosec)


def test_stamp_only_changes_the_stamp_fields():
    grid = _grid()
    costs = np.full(grid.shape, UNKNOWN, dtype=np.int64)
    a = occupancy_grid_from_costmap(costs, grid, stamp=_stamp(1, 2))
    b = occupancy_grid_from_costmap(costs, grid, stamp=_stamp(3, 4))
    assert a.header.stamp != b.header.stamp
    assert a.header.frame_id == b.header.frame_id
    assert list(a.data) == list(b.data)
    b.info.map_load_time = a.info.map_load_time
    assert a.info == b.info


@pytest.mark.parametrize("frame_id", ["a", "grid_frame_x", "ns/some_frame", "robot_1/odom_like"])
def test_frame_id_taken_from_grid(frame_id):
    grid = _grid(frame_id=frame_id)
    msg = occupancy_grid_from_costmap(np.zeros(grid.shape, dtype=np.int64), grid, stamp=_stamp())
    assert msg.header.frame_id == frame_id


def test_same_geometry_different_frames_differ_only_in_frame():
    costs = np.zeros((3, 4), dtype=np.int64)
    a = occupancy_grid_from_costmap(costs, _grid(frame_id="frame_a"), stamp=_stamp())
    b = occupancy_grid_from_costmap(costs, _grid(frame_id="frame_b"), stamp=_stamp())
    assert (a.header.frame_id, b.header.frame_id) == ("frame_a", "frame_b")
    assert a.info == b.info
    assert list(a.data) == list(b.data)


@pytest.mark.parametrize("width, height", [(6, 2), (2, 6)])
def test_transposed_costs_rejected_for_non_square_grid(width, height):
    grid = _grid(width=width, height=height)
    with pytest.raises(ContractError, match="does not match grid shape"):
        occupancy_grid_from_costmap(np.zeros((width, height), dtype=np.int64), grid, stamp=_stamp())


@pytest.mark.parametrize("width, height", [(5, 3), (3, 5)])
def test_data_is_full_c_order_flattening(width, height):
    grid = _grid(width=width, height=height)
    # Distinct occupancy per cell: costs 1..w*h all lie in the inflated band.
    costs = np.arange(1, width * height + 1, dtype=np.int64).reshape(height, width)
    msg = occupancy_grid_from_costmap(costs, grid, stamp=_stamp())
    expected = COST_TO_OCCUPANCY[costs]
    assert list(msg.data) == expected.ravel(order="C").tolist()
    for row in range(height):
        for col in range(width):
            assert msg.data[row * width + col] == expected[row, col]


def test_world_point_lands_at_nav2_index():
    """Nav2/OccupancyGrid index = my * width + mx, with mx from x and my from y."""
    grid = _grid(width=6, height=4, resolution=0.5, origin_x=-1.0, origin_y=3.0)
    geometry = grid.geometry
    x, y = 1.3, 4.2  # synthetic point inside the grid
    cell = world_to_grid_cell(x, y, geometry)
    costs = np.zeros(grid.shape, dtype=np.int64)
    costs[cell.row, cell.col] = LETHAL
    msg = occupancy_grid_from_costmap(costs, grid, stamp=_stamp())
    mx = int((x - msg.info.origin.position.x) // msg.info.resolution)
    my = int((y - msg.info.origin.position.y) // msg.info.resolution)
    assert msg.data[my * msg.info.width + mx] == 100
    assert sum(1 for v in msg.data if v != 0) == 1


def test_unknown_free_lethal_placed_correctly():
    grid = _grid(width=3, height=2)
    costs = np.array([[FREE, UNKNOWN, LETHAL],
                      [LETHAL, FREE, UNKNOWN]], dtype=np.int64)
    msg = occupancy_grid_from_costmap(costs, grid, stamp=_stamp())
    assert list(msg.data) == [0, -1, 100, 100, 0, -1]


def test_unknown_is_never_free():
    grid = _grid()
    msg = occupancy_grid_from_costmap(np.full(grid.shape, UNKNOWN, dtype=np.int64), grid, stamp=_stamp())
    assert set(msg.data) == {-1}


def test_input_costs_not_modified():
    grid = _grid()
    costs = np.full(grid.shape, UNKNOWN, dtype=np.int64)
    costs[0, 0] = LETHAL
    before = costs.copy()
    occupancy_grid_from_costmap(costs, grid, stamp=_stamp())
    np.testing.assert_array_equal(costs, before)


def test_out_of_range_cost_rejected_by_message_builder():
    grid = _grid()
    costs = np.zeros(grid.shape, dtype=np.int64)
    costs[2, 3] = 256
    with pytest.raises(ContractError, match="0..255"):
        occupancy_grid_from_costmap(costs, grid, stamp=_stamp())
