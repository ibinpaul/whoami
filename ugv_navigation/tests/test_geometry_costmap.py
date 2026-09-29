import math

import numpy as np
import pytest

from costmap_core.geometry_costmap import (
    GeometryCostmapError,
    GeometryCostValues,
    build_geometry_costmap,
)


def test_all_false_mask_is_all_free_cost():
    mask = np.zeros((3, 4), dtype=bool)
    costmap = build_geometry_costmap(mask)
    assert costmap.shape == mask.shape
    assert np.all(costmap == 0)


def test_all_true_mask_is_all_lethal_cost():
    mask = np.ones((3, 4), dtype=bool)
    costmap = build_geometry_costmap(mask)
    assert costmap.shape == mask.shape
    assert np.all(costmap == 254)


def test_mixed_occupied_and_free_cells():
    mask = np.array(
        [
            [False, True, False],
            [True, True, False],
        ]
    )
    costmap = build_geometry_costmap(mask)
    expected = [
        [0, 254, 0],
        [254, 254, 0],
    ]
    assert costmap.tolist() == expected


def test_output_shape_matches_input_shape():
    mask = np.zeros((6, 9), dtype=bool)
    costmap = build_geometry_costmap(mask)
    assert costmap.shape == mask.shape


def test_empty_2d_mask_returns_empty_costmap_of_same_shape():
    mask = np.zeros((0, 0), dtype=bool)
    costmap = build_geometry_costmap(mask)
    assert costmap.shape == (0, 0)


def test_non_square_empty_mask_preserves_shape():
    mask = np.zeros((0, 5), dtype=bool)
    costmap = build_geometry_costmap(mask)
    assert costmap.shape == (0, 5)


def test_non_2d_input_raises():
    mask_1d = np.array([True, False, True])
    with pytest.raises(GeometryCostmapError):
        build_geometry_costmap(mask_1d)

    mask_3d = np.zeros((2, 2, 2), dtype=bool)
    with pytest.raises(GeometryCostmapError):
        build_geometry_costmap(mask_3d)


def test_non_boolean_dtype_raises():
    int_mask = np.array([[0, 1], [1, 0]])
    with pytest.raises(GeometryCostmapError):
        build_geometry_costmap(int_mask)

    float_mask = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(GeometryCostmapError):
        build_geometry_costmap(float_mask)


def test_custom_lethal_and_free_cost_are_applied():
    custom = GeometryCostValues(lethal_cost=200, free_cost=10)
    mask = np.array([[False, True]])
    costmap = build_geometry_costmap(mask, custom)
    assert costmap.tolist() == [[10, 200]]


def test_default_lethal_cost_is_254():
    assert GeometryCostValues().lethal_cost == 254


# --- audit regressions ------------------------------------------------------


@pytest.mark.parametrize(
    "mask",
    [
        np.array([[0, 1], [1, 0]], dtype=np.uint8),
        np.array([[0, 254], [254, 0]], dtype=np.int64),
        np.array([[0.0, np.nan], [1.0, 0.0]]),
        np.array([[True, False], [False, True]], dtype=object),
        np.array([["1", "0"], ["0", "1"]]),
    ],
    ids=["uint8", "cost_ints", "float_nan", "object_bools", "str"],
)
def test_invalid_occupancy_values_rejected(mask):
    with pytest.raises(GeometryCostmapError, match="boolean"):
        build_geometry_costmap(mask)


def test_input_not_mutated_and_output_is_new_int64_array():
    mask = np.array([[True, False], [False, True]])
    before = mask.copy()
    out = build_geometry_costmap(mask)
    np.testing.assert_array_equal(mask, before)
    assert out.dtype == np.int64
    assert not np.shares_memory(out, mask)
    out[0, 0] = 0
    np.testing.assert_array_equal(mask, before)


def test_read_only_mask_accepted():
    mask = np.array([[True, False]])
    mask.setflags(write=False)
    assert build_geometry_costmap(mask).tolist() == [[254, 0]]


def test_non_contiguous_mask_keeps_row_col_layout():
    mask = np.array([[True, False, False], [False, False, True]]).T  # (3, 2), Fortran-ordered view
    out = build_geometry_costmap(mask)
    assert out.shape == (3, 2)
    assert out.tolist() == [[254, 0], [0, 0], [0, 254]]


# --- GeometryCostValues validation --------------------------------------------


def test_geometry_defaults_unchanged():
    assert GeometryCostValues() == GeometryCostValues(lethal_cost=254, free_cost=0)


@pytest.mark.parametrize("lethal, free", [(254, 0), (200, 10), (200, 0), (255, 254), (1, 0)])
def test_valid_geometry_costs_accepted(lethal, free):
    cv = GeometryCostValues(lethal_cost=lethal, free_cost=free)
    assert (cv.lethal_cost, cv.free_cost) == (lethal, free)


@pytest.mark.parametrize("field", ["lethal_cost", "free_cost"])
@pytest.mark.parametrize("bad", [True, np.bool_(False), 254.0, 0.5, math.nan, math.inf, "254", None])
def test_non_integer_geometry_costs_rejected(field, bad):
    with pytest.raises(GeometryCostmapError, match=f"{field} must be an integer cost"):
        GeometryCostValues(**{field: bad})


@pytest.mark.parametrize("field", ["lethal_cost", "free_cost"])
@pytest.mark.parametrize("bad", [-1, 256, np.int64(300)])
def test_out_of_range_geometry_costs_rejected(field, bad):
    with pytest.raises(GeometryCostmapError, match="0..255"):
        GeometryCostValues(**{field: bad})


@pytest.mark.parametrize("lethal, free", [(0, 0), (5, 5), (254, 254), (0, 5), (10, 200)])
def test_lethal_must_exceed_free(lethal, free):
    """Includes lethal_cost=0 and lethal == free (occupied indistinguishable from free)."""
    with pytest.raises(GeometryCostmapError, match="free_cost .* must be < lethal_cost"):
        GeometryCostValues(lethal_cost=lethal, free_cost=free)


def test_numpy_integer_geometry_costs_normalised():
    cv = GeometryCostValues(lethal_cost=np.uint8(250), free_cost=np.int32(3))
    assert (type(cv.lethal_cost), type(cv.free_cost)) == (int, int)
