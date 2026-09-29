"""Tests for costmap_core.fail_safe (stale-mask fail-safe costmap, architecture §8.6).

Every region, grid and radius here is SYNTHETIC. The real ROI extent and
"max-inflate" value are PENDING.
"""

import dataclasses
import math

import numpy as np
import pytest

from costmap_core.class_to_cost import DEFAULT_COST_VALUES, CostValues
from costmap_core.contracts import ContractError
from costmap_core.fail_safe import FailSafeRegion, build_fail_safe_costmap, fail_safe_region_mask
from costmap_core.grid import CostmapGridGeometry

LETHAL = 254
UNKNOWN = DEFAULT_COST_VALUES.unknown_cost
FREE = DEFAULT_COST_VALUES.traversable_cost

# Synthetic grid: 0.5 m cells, x in [0, 10), y in [-2, 2); 20 cols x 8 rows.
GRID = CostmapGridGeometry(resolution=0.5, origin_x=0.0, origin_y=-2.0, width=20, height=8)
REGION = dict(min_x=0.0, max_x=2.0, min_y=-1.0, max_y=1.0)


def _region(**overrides):
    kwargs = dict(REGION)
    kwargs.update(overrides)
    return FailSafeRegion(**kwargs)


def _cells(mask):
    return {tuple(rc) for rc in np.argwhere(mask).tolist()}


# --- region -----------------------------------------------------------------------


def test_region_has_no_defaults():
    for field in dataclasses.fields(FailSafeRegion):
        assert field.default is dataclasses.MISSING, field.name
    with pytest.raises(TypeError):
        FailSafeRegion()


@pytest.mark.parametrize(
    "overrides",
    [
        dict(min_x=2.0), dict(min_x=3.0), dict(min_y=1.0), dict(max_y=-2.0),
        dict(min_x=math.nan), dict(max_x=math.inf), dict(min_y=-math.inf),
        dict(max_y=None), dict(min_x="0"), dict(max_x=True),
    ],
)
def test_invalid_region_rejected(overrides):
    with pytest.raises(ContractError):
        _region(**overrides)


def test_region_mask_cell_aligned():
    # x [0, 2] -> cols 0..3 (col 4 starts at 2.0 and only touches);
    # y [-1, 1] -> rows 2..5.
    mask = fail_safe_region_mask(_region(), GRID)
    assert mask.shape == (8, 20)
    assert _cells(mask) == {(r, c) for r in range(2, 6) for c in range(4)}


def test_region_mask_includes_partially_covered_cells():
    # x [0.6, 1.1] touches cols 1 (0.5..1.0) and 2 (1.0..1.5).
    mask = fail_safe_region_mask(_region(min_x=0.6, max_x=1.1, min_y=0.1, max_y=0.2), GRID)
    assert _cells(mask) == {(4, 1), (4, 2)}


def test_region_on_grid_edges_is_accepted():
    mask = fail_safe_region_mask(_region(min_x=0.0, max_x=10.0, min_y=-2.0, max_y=2.0), GRID)
    assert mask.all()


@pytest.mark.parametrize(
    "overrides",
    [dict(min_x=-0.1), dict(max_x=10.01), dict(min_y=-2.5), dict(max_y=2.5),
     dict(min_x=20.0, max_x=21.0)],
)
def test_region_outside_grid_rejected_not_clipped(overrides):
    with pytest.raises(ContractError, match="not inside the grid"):
        fail_safe_region_mask(_region(**overrides), GRID)


def test_region_mask_wrong_types():
    with pytest.raises(ContractError):
        fail_safe_region_mask(dict(REGION), GRID)
    with pytest.raises(ContractError):
        fail_safe_region_mask(_region(), "grid")


# --- costmap ------------------------------------------------------------------------


def test_fail_safe_costmap_roi_lethal_rest_unknown_never_free():
    costmap = build_fail_safe_costmap(_region(), GRID, inflation_radius=0.0)
    roi = fail_safe_region_mask(_region(), GRID)
    assert costmap.dtype == np.int64 and costmap.shape == (8, 20)
    assert np.all(costmap[roi] == LETHAL)
    assert np.all(costmap[~roi] == UNKNOWN)
    assert not np.any(costmap == FREE)


def test_default_costs_inflation_cannot_change_unknown_cells():
    # Inflation keeps max(original, inflated); unknown (255) already exceeds
    # every inflated cost, so the radius has no visible effect by default.
    np.testing.assert_array_equal(
        build_fail_safe_costmap(_region(), GRID, inflation_radius=1.5),
        build_fail_safe_costmap(_region(), GRID, inflation_radius=0.0),
    )


def test_inflation_applied_with_existing_inflation_when_unknown_cost_is_lower():
    costs = CostValues(unknown_cost=100, traversable_cost=0, hazard_cost=254)
    costmap = build_fail_safe_costmap(_region(), GRID, inflation_radius=1.0, cost_values=costs)
    roi = fail_safe_region_mask(_region(), GRID)
    assert np.all(costmap[roi] == LETHAL)
    assert costmap[3, 4] > 100            # next to the ROI: inflated above unknown
    assert costmap[3, 19] == 100          # far away: unknown
    assert not np.any(costmap == 0)


def test_fail_safe_costmap_readonly_and_deterministic():
    a = build_fail_safe_costmap(_region(), GRID, inflation_radius=0.5)
    b = build_fail_safe_costmap(_region(), GRID, inflation_radius=0.5)
    np.testing.assert_array_equal(a, b)
    assert not np.shares_memory(a, b)
    assert not a.flags.writeable


@pytest.mark.parametrize("radius", [-0.1, math.nan, math.inf])
def test_invalid_inflation_radius_rejected(radius):
    from costmap_core.inflation import InflationError

    with pytest.raises(InflationError):
        build_fail_safe_costmap(_region(), GRID, inflation_radius=radius)


def test_inflation_radius_is_required():
    with pytest.raises(TypeError):
        build_fail_safe_costmap(_region(), GRID)
