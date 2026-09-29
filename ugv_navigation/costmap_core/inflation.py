"""Costmap inflation: a distance-based safety buffer around lethal cells.

Keeps a planner from routing directly against an obstacle by raising the
cost of cells near a lethal cell, decaying with distance. This is the
Dev 3 core implementation only: ROS-independent, no Nav2, no TF, no robot
footprint / footprint padding (those are separate tasks -- see
architecture.md / dev.md's Dev 3 "Inflation" and "Footprint Padding"
items, which are listed as distinct steps).

No inflation-cost formula is defined anywhere in PROJECT_CONTEXT.md or
architecture.md, so this module defines its own, kept deliberately
isolated in `_inflation_cost_at_distance` so it can be swapped for a real
Nav2-compatible formula (e.g. exponential decay driven by
`cost_scaling_factor` / `inscribed_radius`) once those parameters are
actually specified for this project.

Implementation: the cost for each (row, col) offset from a lethal cell is
computed once, with the same scalar maths the original per-cell loop used,
and then applied to all lethal cells at once with numpy, one offset at a
time. The output is identical to the original loop, kept for comparison in
tests/inflation_reference.py.
"""

from __future__ import annotations

import math

import numpy as np

from costmap_core.class_to_cost import require_cost

DEFAULT_LETHAL_COST = 254


class InflationError(ValueError):
    """Raised when inflate_costmap inputs are invalid.

    Deliberately not silently coerced -- a malformed costmap or malformed
    inflation geometry must fail loudly rather than silently produce a
    costmap with undefined safety meaning.
    """


def _inflation_cost_at_distance(
    distance_m: float, inflation_radius_m: float, lethal_cost: int
) -> float:
    """Dev 3 placeholder decay: linear from `lethal_cost - 1` at the
    obstacle itself down to 0 at `inflation_radius_m`.

    Kept strictly below `lethal_cost` at every distance so an inflated
    cell is never indistinguishable from an actual lethal obstacle cell.
    Distance >= inflation_radius_m (or a non-positive radius) contributes
    no cost.
    """
    if inflation_radius_m <= 0.0 or distance_m >= inflation_radius_m:
        return 0.0
    max_cost = lethal_cost - 1
    fraction = 1.0 - (distance_m / inflation_radius_m)
    return max_cost * fraction


def _require_finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InflationError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise InflationError(f"{name} must be finite, got {value!r}")
    return value


def inflate_costmap(
    costmap: np.ndarray,
    resolution: float,
    inflation_radius: float,
    lethal_cost: int = DEFAULT_LETHAL_COST,
) -> np.ndarray:
    """Inflate a 2D costmap around every cell equal to `lethal_cost`.

    For every cell within `inflation_radius` metres (Euclidean, using
    `resolution` metres/cell) of any lethal cell, the cost decays with
    distance via `_inflation_cost_at_distance`. A cell's final cost is
    `max(original_cost, inflation_cost)`: an inflated cost never reduces
    an existing higher cost (e.g. a nearby but separate lethal or hazard
    cell), and lethal cells always remain lethal (the inflation formula
    never reaches `lethal_cost` itself, so the original lethal value
    always wins the max).

    Args:
        costmap: 2D array of per-cell costs.
        resolution: grid resolution in metres/cell. Must be finite > 0.
        inflation_radius: safety buffer radius in metres. Must be finite
            and >= 0. A radius of 0 leaves the costmap unchanged.
        lethal_cost: the cost value treated as a lethal obstacle, an
            integer cost in 0..255 (class_to_cost.require_cost).
            Defaults to 254, matching `class_to_cost.CostValues` and
            `geometry_costmap.GeometryCostValues` / `costmap_fusion`.

    Returns:
        A new 2D numpy array (dtype int64), same shape as `costmap`.
        `costmap` itself is never mutated.

    Raises:
        InflationError: if `costmap` is not 2D or has a zero-length
            dimension, if `resolution` is not finite or <= 0, if
            `inflation_radius` is not finite or < 0, or if `lethal_cost`
            is not an integer cost in 0..255.
    """
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

    lethal_cost = require_cost(lethal_cost, name="lethal_cost", error=InflationError)

    result = grid.astype(np.int64, copy=True)

    lethal = grid == lethal_cost
    lethal_rows, lethal_cols = np.nonzero(lethal)
    if lethal_rows.size == 0 or inflation_radius == 0.0:
        return result

    height, width = grid.shape
    radius_cells = math.ceil(inflation_radius / resolution)
    # Offsets beyond the grid size can never land in the grid.
    reach_rows = min(radius_cells, height - 1)
    reach_cols = min(radius_cells, width - 1)
    kernel = _inflation_kernel(reach_rows, reach_cols, resolution, inflation_radius, lethal_cost)

    # Every lethal cell raises each cell of its square window (clipped to the
    # grid) to at least kernel[offset]: result = max(original, all
    # contributions). Applied one offset at a time to all lethal cells at
    # once; max is order-independent, so this equals the per-cell loop.
    # Offsets whose cost cannot raise any cell (<= the smallest current
    # cost; result only grows) are skipped.
    floor_cost = result.min()
    offsets = [
        (dr, dc, int(kernel[dr + reach_rows, dc + reach_cols]))
        for dr in range(-reach_rows, reach_rows + 1)
        for dc in range(-reach_cols, reach_cols + 1)
        if kernel[dr + reach_rows, dc + reach_cols] > floor_cost
    ]
    if not offsets:
        return result
    bounds = (
        int(lethal_rows.min()), int(lethal_rows.max()) + 1,
        int(lethal_cols.min()), int(lethal_cols.max()) + 1,
    )

    costs = [cost for _, _, cost in offsets]
    k_min, k_max = min(costs), max(costs)
    # Sentinel below every applied cost; non-lethal sources add -penalty so
    # their contribution (<= k_max - penalty = sentinel) never counts.
    sentinel, penalty = k_min - 1, k_max - k_min + 1
    for dtype in (np.int16, np.int32):
        info = np.iinfo(dtype)
        if k_max <= info.max and k_min - penalty >= info.min:
            break
    else:
        _apply_offsets_masked(result, lethal, offsets, bounds)
        return result

    source = np.where(lethal, 0, -penalty).astype(dtype)
    reached = np.full(result.shape, sentinel, dtype=dtype)
    for window, source_window, cost in _offset_windows(reached, source, offsets, bounds):
        np.maximum(window, source_window + dtype(cost), out=window)
    np.maximum(result, reached, out=result, where=reached > sentinel)
    return result


def _inflation_kernel(
    reach_rows: int,
    reach_cols: int,
    resolution: float,
    inflation_radius: float,
    lethal_cost: int,
) -> np.ndarray:
    """Integer inflation cost for every (row, col) offset from a lethal cell.

    Shape (2 * reach_rows + 1, 2 * reach_cols + 1); centre = offset (0, 0).
    Each entry is computed with the same scalar maths as the original
    per-cell loop -- int(round(_inflation_cost_at_distance(math.hypot(
    dr * resolution, dc * resolution), ...))) -- so values are identical.
    Offsets inside the square window but beyond the radius are 0, as before.
    Computed for dr, dc >= 0 and mirrored: (-n) * resolution == -(n *
    resolution) exactly and math.hypot ignores signs.
    """
    quadrant = np.empty((reach_rows + 1, reach_cols + 1), dtype=np.int64)
    for dr in range(reach_rows + 1):
        for dc in range(reach_cols + 1):
            distance_m = math.hypot(dr * resolution, dc * resolution)
            quadrant[dr, dc] = int(
                round(_inflation_cost_at_distance(distance_m, inflation_radius, lethal_cost))
            )
    rows = np.concatenate([quadrant[:0:-1], quadrant])
    return np.concatenate([rows[:, :0:-1], rows], axis=1)


def _offset_windows(target, source, offsets, bounds):
    """Yield (target window, source window, cost) per offset: the target cells
    at (row + dr, col + dc) for source cells (row, col) in the lethal bounding
    box `bounds` = (row_lo, row_hi, col_lo, col_hi), clipped to the grid."""
    height, width = target.shape
    row_lo, row_hi, col_lo, col_hi = bounds
    for dr, dc, cost in offsets:
        r0, r1 = max(0, row_lo + dr), min(height, row_hi + dr)
        c0, c1 = max(0, col_lo + dc), min(width, col_hi + dc)
        if r0 < r1 and c0 < c1:
            yield target[r0:r1, c0:c1], source[r0 - dr : r1 - dr, c0 - dc : c1 - dc], cost


def _apply_offsets_masked(result, lethal, offsets, bounds) -> None:
    """Fallback for kernel costs too wide for an int32 accumulator: raise
    result in place (int64) wherever the shifted lethal mask is set."""
    for window, lethal_window, cost in _offset_windows(result, lethal, offsets, bounds):
        np.maximum(window, np.int64(cost), out=window, where=lethal_window)
