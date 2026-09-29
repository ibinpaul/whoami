"""Test-only reference: the original per-pixel project_mask_to_costmap loop.

Kept verbatim (apart from the name) from costmap_core/mask_projection.py
before it was vectorised, so tests can check the production implementation
against it cell for cell. Not used by production code.
"""

from __future__ import annotations

import numpy as np

from costmap_core.class_to_cost import (
    DEFAULT_COST_VALUES,
    CostValues,
    InvalidSemanticClassError,
    SemanticClass,
    class_to_cost,
)
from costmap_core.grid import CostmapGridGeometry, GridError, world_to_grid_cell
from costmap_core.mask_projection import SEMANTIC_CLASS_PRECEDENCE
from costmap_core.projection import (
    CameraGroundGeometry,
    CameraIntrinsics,
    CameraPose,
    ProjectionError,
    project_pixel_to_ground,
)


def reference_project_mask_to_costmap(
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    geometry: CameraGroundGeometry | CameraPose,
    grid_geometry: CostmapGridGeometry,
    cost_values: CostValues = DEFAULT_COST_VALUES,
) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise InvalidSemanticClassError(
            f"Expected a 2D semantic class mask, got array with shape "
            f"{mask.shape} (ndim={mask.ndim})."
        )

    shape = (grid_geometry.height, grid_geometry.width)
    costmap = np.full(shape, cost_values.unknown_cost, dtype=np.int64)
    # Precedence rank of the class currently held by each cell; -1 = untouched.
    winning_rank = np.full(shape, -1, dtype=np.int64)

    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            class_id = int(mask[row, col])
            cost = class_to_cost(class_id, cost_values)
            rank = SEMANTIC_CLASS_PRECEDENCE[SemanticClass(class_id)]

            try:
                ground_point = project_pixel_to_ground(
                    float(col), float(row), intrinsics, geometry
                )
            except ProjectionError:
                continue  # ray does not intersect the ground; skip this pixel

            try:
                cell = world_to_grid_cell(ground_point.x, ground_point.y, grid_geometry)
            except GridError:
                continue  # outside the grid; skip, never clamp

            if rank > winning_rank[cell.row, cell.col]:
                costmap[cell.row, cell.col] = cost
                winning_rank[cell.row, cell.col] = rank

    return costmap
