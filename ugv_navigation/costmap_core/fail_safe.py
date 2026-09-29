"""Stale-mask fail-safe costmap (architecture §8.6).

architecture.md §8.6: "Stale or adapter failure -> /ugv/perception_degraded
+ front ROI lethal/max-inflate + safety hold." §8.4: "Stale mask must not
be treated as current." This module builds only the costmap part: the
front ROI lethal, inflated, and every other cell unknown. The
perception_degraded flag is Dev 1's; the safety hold is Dev 5's.

Not specified by the architecture, so supplied by the caller (no
defaults): the ROI extent and the inflation radius. "max-inflate" has no
defined value; the caller passes a radius. With the default costs every
non-ROI cell is unknown (255), and inflation keeps max(original,
inflated), so inflation cannot change those cells. The ROI is an
axis-aligned rectangle in the grid's own frame (the node's target_frame);
"front" is whatever the caller expresses in that frame. A cell is in the
ROI if it overlaps the rectangle at all (conservative).

Nothing here is free space: no semantic mask is current, so cells outside
the ROI get the unknown cost, never the traversable cost. The result is a
pure function of its arguments, so rebuilding it never compounds costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from costmap_core.class_to_cost import DEFAULT_COST_VALUES, CostValues
from costmap_core.contracts import ContractError, _require_type
from costmap_core.geometry_costmap import DEFAULT_GEOMETRY_COST_VALUES, GeometryCostValues
from costmap_core.grid import CostmapGridGeometry
from costmap_core.inflation import inflate_costmap


def _require_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ContractError(f"{name} must be finite, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class FailSafeRegion:
    """Fail-safe ROI in the grid frame, metres. No defaults: PENDING values.

    Covers x in [min_x, max_x], y in [min_y, max_y]; min < max on both axes.
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def __post_init__(self) -> None:
        for name in ("min_x", "max_x", "min_y", "max_y"):
            object.__setattr__(self, name, _require_finite(getattr(self, name), name=name))
        if not self.min_x < self.max_x:
            raise ContractError(f"min_x ({self.min_x}) must be < max_x ({self.max_x})")
        if not self.min_y < self.max_y:
            raise ContractError(f"min_y ({self.min_y}) must be < max_y ({self.max_y})")


def fail_safe_region_mask(region: FailSafeRegion, grid: CostmapGridGeometry) -> np.ndarray:
    """Bool (height, width) array: True for every cell overlapping the region.

    The region must lie inside the grid extent, so no part of it is silently
    clipped away; otherwise ContractError.
    """
    _require_type(region, FailSafeRegion, name="region")
    _require_type(grid, CostmapGridGeometry, name="grid")
    grid_max_x = grid.origin_x + grid.width * grid.resolution
    grid_max_y = grid.origin_y + grid.height * grid.resolution
    if not (grid.origin_x <= region.min_x and region.max_x <= grid_max_x
            and grid.origin_y <= region.min_y and region.max_y <= grid_max_y):
        raise ContractError(
            f"fail-safe region x [{region.min_x}, {region.max_x}], y [{region.min_y}, {region.max_y}] "
            f"is not inside the grid extent x [{grid.origin_x}, {grid_max_x}], "
            f"y [{grid.origin_y}, {grid_max_y}]"
        )
    col_lo = grid.origin_x + np.arange(grid.width) * grid.resolution
    row_lo = grid.origin_y + np.arange(grid.height) * grid.resolution
    cols = (col_lo < region.max_x) & (col_lo + grid.resolution > region.min_x)
    rows = (row_lo < region.max_y) & (row_lo + grid.resolution > region.min_y)
    return rows[:, None] & cols[None, :]


def build_fail_safe_costmap(
    region: FailSafeRegion,
    grid: CostmapGridGeometry,
    *,
    inflation_radius: float,
    cost_values: CostValues = DEFAULT_COST_VALUES,
    geometry_cost_values: GeometryCostValues = DEFAULT_GEOMETRY_COST_VALUES,
) -> np.ndarray:
    """Read-only int64 (height, width) costmap: ROI lethal, inflated by
    inflation_radius (required), everything else unknown_cost. Uses the same lethal value and inflation
    as the normal pipeline (geometry_cost_values.lethal_cost)."""
    roi = fail_safe_region_mask(region, grid)
    lethal = geometry_cost_values.lethal_cost
    costmap = np.full((grid.height, grid.width), cost_values.unknown_cost, dtype=np.int64)
    costmap[roi] = lethal
    result = inflate_costmap(costmap, grid.resolution, inflation_radius, lethal)
    result.setflags(write=False)
    return result
