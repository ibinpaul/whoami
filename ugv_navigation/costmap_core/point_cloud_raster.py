"""Filtered obstacle points -> OccupancyInput on the costmap grid.

Input: a PointCloudInput that already passed point_cloud_filter (ground
frame, obstacle candidates only) and the GridInput it is rasterised onto.
Only x and y are used; z was already judged by the filter.

World -> cell uses the existing grid.py convention through
mask_projection._ground_to_flat_cells (the vectorised
grid.world_to_grid_cell): col = floor((x - origin_x) / resolution),
row = floor((y - origin_y) / resolution), half-open grid extent. Points
outside the grid are dropped, never clamped -- the same rule the mask
projection applies to out-of-grid ground points.

Semantics (OccupancyInput): True = at least one obstacle point fell in the
cell. False = this observation gave no obstacle evidence there; it does NOT
mean free space. An empty cloud gives an all-False grid with that meaning.
Several points in one cell just mark it once; nothing is counted.
"""

from __future__ import annotations

import numpy as np

from costmap_core.contracts import (
    ContractError,
    GridInput,
    OccupancyInput,
    PointCloudInput,
    _require_type,
)
from costmap_core.mask_projection import _ground_to_flat_cells


def rasterize_obstacle_points(cloud: PointCloudInput, grid: GridInput) -> OccupancyInput:
    """Mark every grid cell containing at least one of cloud's points.

    cloud.frame_id must equal grid.frame_id. The result has the grid's
    (height, width) shape, cloud.stamp_ns and grid.frame_id. Vectorised;
    the input is not modified.

    Raises ContractError for wrong types or a frame mismatch. Invalid grid
    geometry is already rejected when the CostmapGridGeometry is built.
    """
    _require_type(cloud, PointCloudInput, name="cloud")
    _require_type(grid, GridInput, name="grid")
    if cloud.frame_id != grid.frame_id:
        raise ContractError(
            f"cloud.frame_id {cloud.frame_id!r} does not match grid.frame_id {grid.frame_id!r}"
        )

    geometry = grid.geometry
    points = cloud.points.astype(np.float64)
    _, flat_cells = _ground_to_flat_cells(points[:, 0], points[:, 1], geometry)

    occupied = np.zeros(geometry.height * geometry.width, dtype=bool)
    occupied[flat_cells] = True
    return OccupancyInput(
        occupied=occupied.reshape(geometry.height, geometry.width),
        stamp_ns=cloud.stamp_ns,
        frame_id=grid.frame_id,
    )
