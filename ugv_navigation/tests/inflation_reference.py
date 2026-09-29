"""Test-only reference: the original per-lethal-cell inflate_costmap loop.

Kept verbatim (apart from the name) from costmap_core/inflation.py before it
was vectorised, so tests can check the production implementation against it
cell for cell. It reuses the production validation helpers and cost formula,
which were not changed. Not used by production code.
"""

from __future__ import annotations

import math

import numpy as np

from costmap_core.inflation import (
    DEFAULT_LETHAL_COST,
    InflationError,
    _inflation_cost_at_distance,
    _require_finite_number,
)


def reference_inflate_costmap(
    costmap: np.ndarray,
    resolution: float,
    inflation_radius: float,
    lethal_cost: int = DEFAULT_LETHAL_COST,
) -> np.ndarray:
    grid = np.asarray(costmap)

    if grid.ndim != 2:
        raise InflationError(
            f"costmap must be 2D, got shape {grid.shape} (ndim={grid.ndim})."
        )
    if grid.shape[0] == 0 or grid.shape[1] == 0:
        raise InflationError(
            f"costmap dimensions must be non-empty, got shape {grid.shape}."
        )

    resolution = _require_finite_number(resolution, name="resolution")
    if resolution <= 0.0:
        raise InflationError(f"resolution must be > 0, got {resolution!r}.")

    inflation_radius = _require_finite_number(inflation_radius, name="inflation_radius")
    if inflation_radius < 0.0:
        raise InflationError(f"inflation_radius must be >= 0, got {inflation_radius!r}.")

    result = grid.astype(np.int64, copy=True)

    lethal_rows, lethal_cols = np.nonzero(grid == lethal_cost)
    if lethal_rows.size == 0 or inflation_radius == 0.0:
        return result

    height, width = grid.shape
    radius_cells = math.ceil(inflation_radius / resolution)

    for lr, lc in zip(lethal_rows.tolist(), lethal_cols.tolist()):
        r_min = max(0, lr - radius_cells)
        r_max = min(height - 1, lr + radius_cells)
        c_min = max(0, lc - radius_cells)
        c_max = min(width - 1, lc + radius_cells)
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                distance_m = math.hypot((r - lr) * resolution, (c - lc) * resolution)
                cost = int(
                    round(_inflation_cost_at_distance(distance_m, inflation_radius, lethal_cost))
                )
                if cost > result[r, c]:
                    result[r, c] = cost

    return result
