"""Vectorised inflate_costmap must match the original per-lethal-cell loop
(tests/inflation_reference.py) exactly: same values, dtype and errors.
All grids, resolutions, radii and cost values are synthetic test values.
"""

import math

import numpy as np
import pytest

from costmap_core import inflation
from costmap_core.inflation import InflationError, inflate_costmap
from inflation_reference import reference_inflate_costmap

LETHAL, UNKNOWN, FREE = 254, 255, 0


def assert_matches_reference(costmap, resolution, radius, *lethal_cost):
    expected = reference_inflate_costmap(costmap, resolution, radius, *lethal_cost)
    actual = inflate_costmap(costmap, resolution, radius, *lethal_cost)
    assert actual.dtype == expected.dtype == np.int64
    np.testing.assert_array_equal(actual, expected)
    return actual


def _grid(h, w, fill=FREE):
    return np.full((h, w), fill, dtype=np.int64)


def _random_grid(h, w, seed, values=(FREE, LETHAL, UNKNOWN, 50, 253), p=None):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array(values, dtype=np.int64), size=(h, w), p=p)


# --- scenarios -------------------------------------------------------------------


def test_radius_zero():
    grid = _random_grid(20, 30, seed=0)
    result = assert_matches_reference(grid, 0.1, 0.0)
    np.testing.assert_array_equal(result, grid)


def test_no_lethal_cells():
    grid = _random_grid(20, 30, seed=1, values=(FREE, UNKNOWN, 100))
    result = assert_matches_reference(grid, 0.1, 0.5)
    np.testing.assert_array_equal(result, grid)


@pytest.mark.parametrize("cell", [(10, 12), (0, 0), (0, 12), (19, 29), (19, 0), (10, 29)])
def test_one_lethal_cell_centre_edges_and_corners(cell):
    grid = _grid(20, 30)
    grid[cell] = LETHAL
    result = assert_matches_reference(grid, 0.1, 0.55)
    assert result[cell] == LETHAL
    assert (result > FREE).sum() > 1


def test_multiple_lethal_cells():
    grid = _grid(40, 40)
    for cell in [(3, 3), (20, 25), (36, 8), (10, 30)]:
        grid[cell] = LETHAL
    assert_matches_reference(grid, 0.1, 0.8)


def test_overlapping_inflation_regions_use_max_not_sum():
    grid = _grid(15, 25)
    grid[7, 10] = LETHAL
    grid[7, 13] = LETHAL
    grid[8, 11] = LETHAL
    result = assert_matches_reference(grid, 0.1, 0.6)
    assert result.max() == LETHAL
    assert (result[result != LETHAL] < LETHAL).all()


def test_lethal_cells_on_every_boundary():
    grid = _grid(12, 17)
    grid[0, :] = LETHAL
    grid[:, -1] = LETHAL
    grid[-1, 3] = LETHAL
    assert_matches_reference(grid, 0.25, 1.0)


@pytest.mark.parametrize("resolution", [0.01, 0.03, 0.05, 0.1, 0.25, 0.3, 1.0, 2.0])
def test_different_resolutions(resolution):
    grid = _random_grid(30, 30, seed=2, p=[0.8, 0.05, 0.1, 0.03, 0.02])
    assert_matches_reference(grid, resolution, 0.5)


@pytest.mark.parametrize("radius", [1e-9, 0.01, 0.05, 0.1, 0.15, 0.3, 0.5, 0.55, 1.0, 2.5, 100.0])
def test_different_radii(radius):
    grid = _random_grid(25, 35, seed=3, p=[0.8, 0.05, 0.1, 0.03, 0.02])
    assert_matches_reference(grid, 0.1, radius)


@pytest.mark.parametrize("resolution, radius", [(0.1, 0.5), (0.1, 0.3), (0.05, 0.25), (0.2, 0.6), (0.1, 0.7)])
def test_radius_near_multiple_of_resolution(resolution, radius):
    # e.g. 0.5 / 0.1 = 5.000000000000001 -> ceil = 6 cells; boundary distances hit
    # the "distance >= radius -> 0" edge exactly or within an ulp.
    grid = _random_grid(30, 30, seed=4, p=[0.85, 0.05, 0.05, 0.03, 0.02])
    assert_matches_reference(grid, resolution, radius)


def test_existing_nonzero_costs_kept_when_higher():
    grid = _random_grid(30, 30, seed=5, values=(FREE, LETHAL, UNKNOWN, 1, 120, 200, 252, 253))
    assert_matches_reference(grid, 0.1, 0.7)


def test_negative_existing_costs_raised_by_window_corners():
    # The original loop visits the whole square window, so even cells beyond
    # the radius (cost 0) are raised to 0 when their cost is negative.
    grid = _grid(15, 15, fill=-5)
    grid[7, 7] = LETHAL
    result = assert_matches_reference(grid, 0.1, 0.3)
    assert result[4, 4] == 0  # window corner, beyond the radius
    assert result[0, 0] == -5  # outside the window


@pytest.mark.parametrize("lethal_cost", [254, 255, 100, 2, 1, 0])
def test_custom_lethal_cost(lethal_cost):
    rng = np.random.default_rng(6)
    grid = rng.integers(-3, 50, size=(20, 20)).astype(np.int64)
    grid[rng.random((20, 20)) < 0.05] = lethal_cost
    assert_matches_reference(grid, 0.1, 0.45, lethal_cost)


@pytest.mark.parametrize(
    "lethal_cost",
    [
        -5,
        32768,  # kernel max 32767: int16 accumulator at its limit
        32769,  # int32 accumulator
        2**31 + 5,  # too wide for int32: masked int64 fallback
    ],
)
def test_wide_accumulator_paths_match_reference(monkeypatch, lethal_cost):
    """The public API now rejects lethal_cost outside 0..255 (so the kernel
    always fits int16), but the int32 / masked int64 branches are kept
    unchanged; bypass the validation to keep them equivalent to the reference."""
    monkeypatch.setattr(inflation, "require_cost", lambda value, **_: value)
    rng = np.random.default_rng(6)
    grid = rng.integers(-3, 50, size=(20, 20)).astype(np.int64)
    grid[rng.random((20, 20)) < 0.05] = lethal_cost
    assert_matches_reference(grid, 0.1, 0.45, lethal_cost)


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (2, 1), (1, 9), (9, 1), (3, 3), (2, 5)])
def test_small_grids(shape):
    rng = np.random.default_rng(7)
    grid = rng.choice(np.array([FREE, LETHAL, UNKNOWN]), size=shape)
    grid.flat[0] = LETHAL
    assert_matches_reference(grid, 0.1, 0.35)


def test_all_lethal_grid():
    assert_matches_reference(_grid(10, 10, fill=LETHAL), 0.1, 0.5)


def test_radius_larger_than_grid():
    grid = _grid(6, 9)
    grid[2, 3] = LETHAL
    assert_matches_reference(grid, 0.1, 50.0)


def test_non_int64_inputs():
    grid = _random_grid(12, 14, seed=8, p=[0.8, 0.05, 0.1, 0.03, 0.02])
    assert_matches_reference(grid.astype(np.uint8), 0.1, 0.5)
    assert_matches_reference(grid.astype(np.int32), 0.1, 0.5)
    assert_matches_reference(grid.astype(np.float64), 0.1, 0.5)
    assert_matches_reference(grid.tolist(), 0.1, 0.5)
    float_grid = grid.astype(np.float64)
    float_grid[0, 0] = 254.5  # not equal to lethal_cost; truncated to 254 in the output
    assert_matches_reference(float_grid, 0.1, 0.5)


def test_random_grids():
    rng = np.random.default_rng(1234)
    for _ in range(60):
        h, w = int(rng.integers(1, 40)), int(rng.integers(1, 40))
        values = rng.choice([FREE, LETHAL, UNKNOWN, 7, 128, 253, -2], size=int(rng.integers(1, 6)), replace=False)
        grid = rng.choice(values, size=(h, w)).astype(np.int64)
        resolution = float(rng.choice([0.02, 0.05, 0.1, 0.2, 0.5, 1.0]))
        radius = float(rng.uniform(0.0, 1.5))
        assert_matches_reference(grid, resolution, radius)


def test_representative_grid_from_projection_scale():
    grid = _random_grid(100, 100, seed=9, values=(FREE, LETHAL, UNKNOWN), p=[0.75, 0.05, 0.2])
    assert_matches_reference(grid, 0.1, 0.5)


@pytest.mark.parametrize("offset_from_kernel_value", [-1, 0])
def test_background_just_below_and_at_each_kernel_value(offset_from_kernel_value):
    # Offsets whose cost cannot raise any cell are skipped; this pins the skip
    # threshold: a cost one above the background must still be applied, and
    # a cost equal to it changes nothing.
    resolution, radius = 0.1, 0.45
    reach = math.ceil(radius / resolution)
    values = np.unique(inflation._inflation_kernel(reach, reach, resolution, radius, LETHAL))
    for value in values:
        grid = _grid(15, 15, fill=int(value) + offset_from_kernel_value)
        grid[7, 7] = LETHAL
        grid[2, 12] = LETHAL
        assert_matches_reference(grid, resolution, radius)


def test_input_not_mutated_and_result_is_new_array():
    grid = _random_grid(10, 10, seed=10)
    before = grid.copy()
    result = inflate_costmap(grid, 0.1, 0.4)
    np.testing.assert_array_equal(grid, before)
    assert result is not grid and not np.shares_memory(result, grid)


@pytest.mark.parametrize(
    "args",
    [
        (np.zeros(5), 0.1, 0.5),
        (np.zeros((0, 3)), 0.1, 0.5),
        (np.zeros((3, 3)), 0.0, 0.5),
        (np.zeros((3, 3)), math.nan, 0.5),
        (np.zeros((3, 3)), 0.1, -1.0),
        (np.zeros((3, 3)), 0.1, math.inf),
        (np.zeros((3, 3)), True, 0.5),
    ],
)
def test_errors_match_reference(args):
    with pytest.raises(InflationError) as expected:
        reference_inflate_costmap(*args)
    with pytest.raises(InflationError) as actual:
        inflate_costmap(*args)
    assert str(actual.value) == str(expected.value)


def test_kernel_matches_scalar_formula_for_every_offset():
    for resolution, radius, lethal_cost in [(0.1, 0.5, 254), (0.05, 1.0, 254), (0.3, 0.7, 100)]:
        reach = math.ceil(radius / resolution)
        kernel = inflation._inflation_kernel(reach, reach, resolution, radius, lethal_cost)
        for dr in range(-reach, reach + 1):
            for dc in range(-reach, reach + 1):
                distance = math.hypot(dr * resolution, dc * resolution)
                expected = int(round(inflation._inflation_cost_at_distance(distance, radius, lethal_cost)))
                assert kernel[dr + reach, dc + reach] == expected


# --- mutation tests: the scenarios above must catch a changed combine rule -------------------


def _mutant(update):
    """The reference loop with its `if cost > result: result = cost` replaced by `update`."""

    def run(costmap, resolution, radius, lethal_cost=254):
        grid = np.asarray(costmap)
        result = grid.astype(np.int64, copy=True)
        rows, cols = np.nonzero(grid == lethal_cost)
        if rows.size == 0 or radius == 0.0:
            return result
        h, w = grid.shape
        rc = math.ceil(radius / resolution)
        for lr, lc in zip(rows.tolist(), cols.tolist()):
            for r in range(max(0, lr - rc), min(h - 1, lr + rc) + 1):
                for c in range(max(0, lc - rc), min(w - 1, lc + rc) + 1):
                    d = math.hypot((r - lr) * resolution, (c - lc) * resolution)
                    cost = int(round(inflation._inflation_cost_at_distance(d, radius, lethal_cost)))
                    update(result, grid, r, c, cost)
        return result

    return run


def _overwrite(result, grid, r, c, cost):  # last write wins, ignores existing cost
    result[r, c] = cost


def _sum(result, grid, r, c, cost):  # overlapping regions add up
    result[r, c] = min(result[r, c] + cost, 10**6) if cost > 0 else result[r, c]


def _max_with_original_only(result, grid, r, c, cost):  # max against original, not accumulated
    result[r, c] = max(int(grid[r, c]), cost)


def _min_non_zero(result, grid, r, c, cost):  # weakest contribution wins
    if 0 < cost and (result[r, c] == grid[r, c] or cost < result[r, c]) and cost > grid[r, c]:
        result[r, c] = cost


def _lethal_overwritable(result, grid, r, c, cost):  # max, but lethal cells can be lowered
    if cost > result[r, c] or grid[r, c] == 254:
        result[r, c] = cost


MUTANTS = {
    "overwrite": _mutant(_overwrite),
    "sum": _mutant(_sum),
    "max_with_original_only": _mutant(_max_with_original_only),
    "min_non_zero": _mutant(_min_non_zero),
    "lethal_overwritable": _mutant(_lethal_overwritable),
}


def _scenario_suite():
    """A compact set of the scenarios above, as (costmap, resolution, radius)."""
    overlap = _grid(15, 25)
    overlap[7, 10] = overlap[7, 13] = overlap[8, 11] = LETHAL
    existing = _random_grid(30, 30, seed=5, values=(FREE, LETHAL, UNKNOWN, 1, 120, 200, 252, 253))
    single = _grid(20, 30)
    single[10, 12] = LETHAL
    return [(overlap, 0.1, 0.6), (existing, 0.1, 0.7), (single, 0.1, 0.55)]


def test_production_passes_scenario_suite():
    for grid, resolution, radius in _scenario_suite():
        assert_matches_reference(grid, resolution, radius)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_scenario_suite_detects_changed_combine_rule(name):
    mutant = MUTANTS[name]
    differs = any(
        not np.array_equal(mutant(g, res, rad), reference_inflate_costmap(g, res, rad))
        for g, res, rad in _scenario_suite()
    )
    assert differs, f"mutant {name!r} was not detected by the scenario suite"
