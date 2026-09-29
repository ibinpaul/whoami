"""Dev 3 costmap array -> temporary nav_msgs/OccupancyGrid output.

Temporary output for inspection (e.g. RViz) until the Nav2 integration is
decided. The cost -> occupancy translation is Nav2's own table from
nav2_costmap_2d Costmap2DPublisher (cost_translation_table_), which assumes
the Nav2 cost conventions that class_to_cost.DEFAULT_COST_VALUES and
geometry_costmap.DEFAULT_GEOMETRY_COST_VALUES use:

    0 (FREE_SPACE)            -> 0
    1..252 (inflated)         -> 1 + 97 * (cost - 1) // 251   (1..98)
    253 (INSCRIBED_INFLATED)  -> 99
    254 (LETHAL_OBSTACLE)     -> 100
    255 (NO_INFORMATION)      -> -1

Any cost outside 0..255 is rejected, never clipped.

Layout: CostmapGridGeometry and OccupancyGrid share the same convention
(origin = min-x, min-y corner of cell (row 0, col 0); row increases with y,
col with x), so the grid is flattened row-major unchanged.
"""

from __future__ import annotations

import numpy as np
from nav_msgs.msg import OccupancyGrid

from costmap_core.contracts import ContractError, GridInput

NAV2_FREE_SPACE = 0
NAV2_INSCRIBED_INFLATED_OBSTACLE = 253
NAV2_LETHAL_OBSTACLE = 254
NAV2_NO_INFORMATION = 255


def _nav2_cost_translation_table() -> np.ndarray:
    table = np.empty(256, dtype=np.int8)
    table[NAV2_FREE_SPACE] = 0
    for cost in range(1, NAV2_INSCRIBED_INFLATED_OBSTACLE):
        table[cost] = 1 + (97 * (cost - 1)) // 251
    table[NAV2_INSCRIBED_INFLATED_OBSTACLE] = 99
    table[NAV2_LETHAL_OBSTACLE] = 100
    table[NAV2_NO_INFORMATION] = -1
    return table


COST_TO_OCCUPANCY = _nav2_cost_translation_table()
COST_TO_OCCUPANCY.setflags(write=False)


def costs_to_occupancy(costs: np.ndarray) -> np.ndarray:
    """Translate a 2D integer cost array (0..255) to OccupancyGrid values (int8)."""
    costs = np.asarray(costs)
    if costs.ndim != 2 or not np.issubdtype(costs.dtype, np.integer):
        raise ContractError(
            f"costs must be a 2D integer array, got shape {costs.shape} dtype {costs.dtype}"
        )
    if costs.size and (costs.min() < 0 or costs.max() > 255):
        raise ContractError(
            f"costs must be in 0..255, got range [{int(costs.min())}, {int(costs.max())}]"
        )
    return COST_TO_OCCUPANCY[costs]


def occupancy_grid_from_costmap(costs: np.ndarray, grid: GridInput, *, stamp) -> OccupancyGrid:
    """Build an OccupancyGrid for `costs` on `grid`, stamped with `stamp`.

    stamp: builtin_interfaces/Time chosen by the caller (the stamp policy is
        the node's, not this helper's). header.frame_id = grid.frame_id.
    """
    if not isinstance(grid, GridInput):
        raise ContractError(f"grid must be a GridInput, got {type(grid).__name__}")
    occupancy = costs_to_occupancy(costs)
    if occupancy.shape != grid.shape:
        raise ContractError(
            f"costs shape {occupancy.shape} does not match grid shape (height, width) = {grid.shape}"
        )

    geometry = grid.geometry
    msg = OccupancyGrid()
    msg.header.stamp = stamp
    msg.header.frame_id = grid.frame_id
    msg.info.map_load_time = stamp
    msg.info.resolution = float(geometry.resolution)
    msg.info.width = geometry.width
    msg.info.height = geometry.height
    msg.info.origin.position.x = float(geometry.origin_x)
    msg.info.origin.position.y = float(geometry.origin_y)
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0
    msg.data = occupancy.ravel(order="C").tolist()
    return msg
