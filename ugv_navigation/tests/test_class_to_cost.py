import math

import numpy as np
import pytest

from costmap_core.class_to_cost import (
    MAX_COST,
    MIN_COST,
    CostValues,
    InvalidCostValueError,
    InvalidSemanticClassError,
    SemanticClass,
    class_to_cost,
    require_cost,
)


def test_unknown_maps_to_configured_unknown_cost():
    cost_values = CostValues(unknown_cost=255, traversable_cost=0, hazard_cost=254)
    assert class_to_cost(SemanticClass.UNKNOWN, cost_values) == 255


def test_traversable_maps_to_configured_free_cost():
    cost_values = CostValues(unknown_cost=255, traversable_cost=0, hazard_cost=254)
    assert class_to_cost(SemanticClass.TRAVERSABLE, cost_values) == 0


def test_hazard_maps_to_configured_lethal_cost():
    cost_values = CostValues(unknown_cost=255, traversable_cost=0, hazard_cost=254)
    assert class_to_cost(SemanticClass.HAZARD, cost_values) == 254


def test_default_cost_values_used_when_not_provided():
    assert class_to_cost(SemanticClass.TRAVERSABLE) == 0


def test_custom_cost_values_are_respected():
    custom = CostValues(unknown_cost=128, traversable_cost=10, hazard_cost=200)
    assert class_to_cost(SemanticClass.HAZARD, custom) == 200


@pytest.mark.parametrize("invalid_class_id", [-1, 3, 99, 255])
def test_invalid_class_id_raises_explicit_error(invalid_class_id):
    with pytest.raises(InvalidSemanticClassError):
        class_to_cost(invalid_class_id)


# --- cost value validation ---------------------------------------------------

COST_FIELDS = ("unknown_cost", "traversable_cost", "hazard_cost")


def test_defaults_unchanged():
    assert CostValues() == CostValues(unknown_cost=255, traversable_cost=0, hazard_cost=254)
    assert (MIN_COST, MAX_COST) == (0, 255)


@pytest.mark.parametrize(
    "values",
    [
        (255, 0, 254),  # defaults
        (128, 10, 200),
        (10, 20, 30),  # unknown below traversable: no order imposed on unknown
        (200, 7, 254),
        (255, 0, 200),
        (255, 0, 253),
        (0, 1, 255),  # range edges
        (1, 0, 255),
    ],
)
def test_valid_custom_costs_accepted(values):
    unknown, traversable, hazard = values
    cv = CostValues(unknown_cost=unknown, traversable_cost=traversable, hazard_cost=hazard)
    assert (cv.unknown_cost, cv.traversable_cost, cv.hazard_cost) == values


@pytest.mark.parametrize("field", COST_FIELDS)
@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), 1.0, 254.0, 254.9, math.nan, math.inf, "254", None, [254], 1 + 0j],
)
def test_non_integer_costs_rejected(field, bad):
    with pytest.raises(InvalidCostValueError, match=f"{field} must be an integer cost"):
        CostValues(**{field: bad})


@pytest.mark.parametrize("field", COST_FIELDS)
@pytest.mark.parametrize("bad", [-1, 256, 1000, -255, np.int64(-1), np.uint16(256)])
def test_out_of_range_costs_rejected(field, bad):
    with pytest.raises(InvalidCostValueError, match="0..255"):
        CostValues(**{field: bad})


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.int64])
def test_numpy_integers_accepted_and_normalised_to_int(dtype):
    cv = CostValues(unknown_cost=dtype(200), traversable_cost=dtype(3), hazard_cost=dtype(250))
    assert (cv.unknown_cost, cv.traversable_cost, cv.hazard_cost) == (200, 3, 250)
    assert all(type(v) is int for v in (cv.unknown_cost, cv.traversable_cost, cv.hazard_cost))


@pytest.mark.parametrize(
    "kwargs, why",
    [
        (dict(unknown_cost=0), "unknown == traversable would make unknown free"),
        (dict(unknown_cost=254), "unknown == hazard would make unseen cells lethal"),
        (dict(traversable_cost=254, hazard_cost=254), "hazard == traversable"),
        (dict(unknown_cost=7, traversable_cost=7, hazard_cost=7), "all identical"),
    ],
)
def test_identical_costs_rejected(kwargs, why):
    with pytest.raises(InvalidCostValueError, match="pairwise distinct"):
        CostValues(**kwargs)


@pytest.mark.parametrize("traversable, hazard", [(254, 0), (200, 100), (1, 0)])
def test_traversable_must_be_below_hazard(traversable, hazard):
    with pytest.raises(InvalidCostValueError, match="traversable_cost .* must be < hazard_cost"):
        CostValues(unknown_cost=255, traversable_cost=traversable, hazard_cost=hazard)


def test_float_unknown_can_no_longer_truncate_to_lethal():
    """Before validation, unknown_cost=254.9 was truncated to 254 (= lethal) by int64 costmaps."""
    with pytest.raises(InvalidCostValueError):
        CostValues(unknown_cost=254.9)


def test_cost_values_remain_frozen():
    cv = CostValues()
    with pytest.raises(AttributeError):
        cv.hazard_cost = 1


@pytest.mark.parametrize("value", [0, 255, np.uint8(17)])
def test_require_cost_accepts(value):
    assert require_cost(value, name="x") == int(value)


def test_require_cost_uses_given_error_type():
    class CustomError(ValueError):
        pass

    with pytest.raises(CustomError, match="x must be in"):
        require_cost(256, name="x", error=CustomError)
