import copy
import math

import pytest

from costmap_core.footprint import (
    FootprintError,
    pad_footprint,
    signed_area,
    validate_footprint,
    validate_padding,
)

# TEST-ONLY SYNTHETIC FOOTPRINTS. These are NOT the real UGV footprint
# (which Dev 5 has not provided yet) and must not be used as robot config.
SYNTHETIC_HALF_LENGTH = 0.5  # test-only value, metres
SYNTHETIC_HALF_WIDTH = 0.3  # test-only value, metres
SYNTHETIC_RECTANGLE_CCW = [
    (SYNTHETIC_HALF_LENGTH, SYNTHETIC_HALF_WIDTH),
    (-SYNTHETIC_HALF_LENGTH, SYNTHETIC_HALF_WIDTH),
    (-SYNTHETIC_HALF_LENGTH, -SYNTHETIC_HALF_WIDTH),
    (SYNTHETIC_HALF_LENGTH, -SYNTHETIC_HALF_WIDTH),
]
SYNTHETIC_RECTANGLE_CW = list(reversed(SYNTHETIC_RECTANGLE_CCW))
SYNTHETIC_TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]  # test-only
SYNTHETIC_PENTAGON = [  # test-only regular pentagon, circumradius 1.0
    (math.cos(math.radians(90 + 72 * k)), math.sin(math.radians(90 + 72 * k))) for k in range(5)
]

APPROX = pytest.approx


def _distance_to_line(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    return abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / math.hypot(bx - ax, by - ay)


def _assert_points_close(actual, expected):
    assert len(actual) == len(expected)
    for (ax, ay), (ex, ey) in zip(actual, expected):
        assert ax == APPROX(ex)
        assert ay == APPROX(ey)


# --- validation: valid footprints ---------------------------------------


@pytest.mark.parametrize(
    "footprint",
    [SYNTHETIC_RECTANGLE_CCW, SYNTHETIC_RECTANGLE_CW, SYNTHETIC_TRIANGLE, SYNTHETIC_PENTAGON],
)
def test_valid_footprint_is_accepted_and_returned_as_float_tuples(footprint):
    result = validate_footprint(footprint)
    _assert_points_close(result, footprint)
    assert all(isinstance(p, tuple) and all(isinstance(c, float) for c in p) for p in result)


def test_valid_footprint_accepts_lists_and_ints():
    result = validate_footprint([[0, 0], [2, 0], [2, 1], [0, 1]])
    assert result == [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]


def test_valid_footprint_allows_collinear_vertex():
    # Midpoint on the bottom edge of a synthetic 2x1 rectangle.
    validate_footprint([(0, 0), (1, 0), (2, 0), (2, 1), (0, 1)])


# --- validation: invalid footprints -------------------------------------


@pytest.mark.parametrize(
    "footprint",
    [
        [],
        [(0, 0), (1, 0)],  # too few vertices
        [(0, 0), (1, 0), (2, 0)],  # zero area
        [(0, 0), (1, 0), (1, 0), (0, 1)],  # repeated consecutive vertex
        [(0, 0), (1, 0), (0, 1), (0, 0)],  # explicitly closed ring
        [(0, 0), (1, 1), (1, 0), (0, 1)],  # bow-tie self-intersection
        [(0, 0), (2, 0), (1, 0.5), (2, 2), (0, 2)],  # concave
        [(0, 0), (1, 0), (math.nan, 1)],  # non-finite
        [(0, 0), (1, 0), (math.inf, 1)],  # non-finite
        [(0, 0), (1, 0), (0, 1, 2)],  # vertex not a pair
        [(0, 0), (1, 0), ("0", 1)],  # non-numeric coordinate
        [(0, 0), (1, 0), (True, 1)],  # bool coordinate
        "abc",
        None,
    ],
)
def test_invalid_footprint_is_rejected(footprint):
    with pytest.raises(FootprintError):
        validate_footprint(footprint)


def test_self_winding_star_is_rejected():
    # Pentagram: every turn has the same direction, but edges cross.
    star = [SYNTHETIC_PENTAGON[(2 * k) % 5] for k in range(5)]
    with pytest.raises(FootprintError, match="self-intersecting"):
        validate_footprint(star)


def test_invalid_footprint_is_rejected_by_pad_footprint():
    with pytest.raises(FootprintError):
        pad_footprint([(0, 0), (1, 0)], 0.1)


# --- padding validation -------------------------------------------------


@pytest.mark.parametrize("padding", [0, 0.0, 0.1, 2])
def test_valid_padding_is_accepted(padding):
    assert validate_padding(padding) == float(padding)


@pytest.mark.parametrize("padding", [-0.01, -1, math.nan, math.inf, -math.inf, None, "0.1", True])
def test_invalid_padding_is_rejected(padding):
    with pytest.raises(FootprintError):
        validate_padding(padding)


def test_negative_padding_rejected_by_pad_footprint():
    with pytest.raises(FootprintError, match=">= 0"):
        pad_footprint(SYNTHETIC_RECTANGLE_CCW, -0.1)


# --- padding behaviour --------------------------------------------------


def test_zero_padding_returns_unchanged_copy():
    result = pad_footprint(SYNTHETIC_RECTANGLE_CCW, 0.0)
    assert result == SYNTHETIC_RECTANGLE_CCW
    assert result is not SYNTHETIC_RECTANGLE_CCW


@pytest.mark.parametrize("rectangle", [SYNTHETIC_RECTANGLE_CCW, SYNTHETIC_RECTANGLE_CW])
def test_rectangle_padding_grows_each_half_dimension_by_padding(rectangle):
    padding = 0.1  # test-only value, metres
    result = pad_footprint(rectangle, padding)
    hl = SYNTHETIC_HALF_LENGTH + padding
    hw = SYNTHETIC_HALF_WIDTH + padding
    expected = [(math.copysign(hl, x), math.copysign(hw, y)) for x, y in rectangle]
    _assert_points_close(result, expected)


def test_off_centre_rectangle_is_padded_outward_not_away_from_origin():
    # Synthetic rectangle entirely in +x: the rear edge (x=1) must move to
    # x=0.9, i.e. outward from the polygon, not away from the origin.
    result = pad_footprint([(1, 0), (3, 0), (3, 1), (1, 1)], 0.1)
    _assert_points_close(result, [(0.9, -0.1), (3.1, -0.1), (3.1, 1.1), (0.9, 1.1)])


@pytest.mark.parametrize(
    "footprint",
    [SYNTHETIC_RECTANGLE_CCW, SYNTHETIC_RECTANGLE_CW, SYNTHETIC_TRIANGLE, SYNTHETIC_PENTAGON],
)
@pytest.mark.parametrize("padding", [0.05, 0.25, 1.0])
def test_every_edge_moves_outward_by_exactly_padding(footprint, padding):
    result = pad_footprint(footprint, padding)
    n = len(footprint)
    for i in range(n):
        a, b = footprint[i], footprint[(i + 1) % n]
        pa, pb = result[i], result[(i + 1) % n]
        # Padded edge endpoints lie on a line exactly `padding` from the original edge line.
        assert _distance_to_line(pa, a, b) == APPROX(padding)
        assert _distance_to_line(pb, a, b) == APPROX(padding)
    # ...and on the outside: every original vertex is inside the padded polygon.
    for vertex in footprint:
        assert _point_strictly_inside_convex(vertex, result)


def _point_strictly_inside_convex(p, polygon):
    n = len(polygon)
    crosses = []
    for i in range(n):
        (ax, ay), (bx, by) = polygon[i], polygon[(i + 1) % n]
        crosses.append((bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax))
    return all(c > 0 for c in crosses) or all(c < 0 for c in crosses)


@pytest.mark.parametrize(
    "footprint",
    [SYNTHETIC_RECTANGLE_CCW, SYNTHETIC_RECTANGLE_CW, SYNTHETIC_TRIANGLE, SYNTHETIC_PENTAGON],
)
@pytest.mark.parametrize("padding", [0.05, 0.25, 1.0])
def test_padded_footprint_remains_geometrically_valid(footprint, padding):
    result = pad_footprint(footprint, padding)
    assert validate_footprint(result) == result
    # Same vertex count, same winding direction, strictly larger area.
    assert len(result) == len(footprint)
    assert math.copysign(1, signed_area(result)) == math.copysign(1, signed_area(footprint))
    assert abs(signed_area(result)) > abs(signed_area(footprint))


def test_collinear_vertex_moves_straight_out():
    result = pad_footprint([(0, 0), (1, 0), (2, 0), (2, 1), (0, 1)], 0.1)
    _assert_points_close(
        result, [(-0.1, -0.1), (1.0, -0.1), (2.1, -0.1), (2.1, 1.1), (-0.1, 1.1)]
    )


def test_input_footprint_is_not_mutated():
    footprint = [[0.5, 0.3], [-0.5, 0.3], [-0.5, -0.3], [0.5, -0.3]]  # test-only, mutable lists
    snapshot = copy.deepcopy(footprint)
    pad_footprint(footprint, 0.2)
    validate_footprint(footprint)
    assert footprint == snapshot

    rect_snapshot = list(SYNTHETIC_RECTANGLE_CCW)
    pad_footprint(SYNTHETIC_RECTANGLE_CCW, 0.2)
    assert SYNTHETIC_RECTANGLE_CCW == rect_snapshot


# --- audit: malformed input, duplicates, overflow ----------------------------


@pytest.mark.parametrize(
    "footprint",
    [
        [(0, 0), (1, 0), {0: 0, 1: 1}],  # vertex is a mapping, not a pair
        [(0, 0), (1, 0), (1,)],  # vertex too short
        [(0, 0), (1, 0), "01"],  # vertex is a string of length 2
        [(0, 0), (1, 0), ((0, 1),)],  # nested pair
        ((x, y) for x, y in SYNTHETIC_TRIANGLE),  # generator, not a sequence
        {(0, 0), (1, 0), (0, 1)},  # set: unordered
    ],
)
def test_malformed_points_rejected(footprint):
    with pytest.raises(FootprintError):
        validate_footprint(footprint)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_coordinate_rejected_on_either_axis(axis, bad):
    footprint = [list(p) for p in SYNTHETIC_TRIANGLE]
    footprint[1][axis] = bad
    with pytest.raises(FootprintError, match="finite"):
        validate_footprint(footprint)


def test_non_consecutive_duplicate_vertex_rejected():
    # Revisiting vertex (0, 0) mid-ring cannot form a simple convex polygon.
    with pytest.raises(FootprintError):
        validate_footprint([(0, 0), (2, 0), (2, 2), (0, 0), (0, 2)])


@pytest.mark.parametrize(
    "footprint",
    [
        [(0, 0), (2, 0), (4, 0), (2, 0), (2, 2), (0, 2)],  # spike out and back along an edge
        [(0, 0), (3, 0), (1, 0), (1, 1), (0, 1)],  # edge doubles back over itself
    ],
)
def test_degenerate_spikes_rejected(footprint):
    with pytest.raises(FootprintError):
        validate_footprint(footprint)


def test_double_wound_polygon_rejected():
    triangle = [(1.0, 0.0), (-0.5, 0.8660254), (-0.5, -0.8660254)]  # test-only
    with pytest.raises(FootprintError, match="self-intersecting"):
        validate_footprint(triangle * 2)


def test_collinear_points_that_overflow_area_are_rejected():
    # Finite coordinates, but x0*y1 overflows to inf and inf - inf = nan,
    # which used to slip past the zero-area and convexity checks.
    with pytest.raises(FootprintError, match="not finite"):
        validate_footprint([(0.0, 0.0), (1e160, 1e160), (2e160, 2e160)])


def test_large_but_representable_footprint_still_accepted():
    """No magnitude limit is invented: only overflowing arithmetic is refused."""
    s = 1e100  # test-only scale, far beyond any robot
    assert validate_footprint([(s, s), (-s, s), (-s, -s), (s, -s)])[0] == (s, s)


def test_padding_that_overflows_is_rejected():
    with pytest.raises(FootprintError, match="padded vertex"):
        pad_footprint(SYNTHETIC_TRIANGLE, 1e308)


def test_negative_zero_padding_is_zero():
    assert pad_footprint(SYNTHETIC_RECTANGLE_CCW, -0.0) == validate_footprint(SYNTHETIC_RECTANGLE_CCW)


@pytest.mark.parametrize("padding", [math.nan, math.inf, -math.inf])
def test_non_finite_padding_rejected_by_pad_footprint(padding):
    with pytest.raises(FootprintError, match="finite"):
        pad_footprint(SYNTHETIC_RECTANGLE_CCW, padding)


def test_validate_returns_new_list_each_call():
    a = validate_footprint(SYNTHETIC_RECTANGLE_CCW)
    b = validate_footprint(SYNTHETIC_RECTANGLE_CCW)
    assert a == b and a is not b
    a.append((9.0, 9.0))
    assert validate_footprint(SYNTHETIC_RECTANGLE_CCW) == b


# --- audit: Python ints beyond float range, determinism ----------------------


@pytest.mark.parametrize("axis", [0, 1])
def test_int_coordinate_beyond_float_range_is_footprint_error(axis):
    vertex = [0, 1]
    vertex[axis] = 10 ** 400  # float(10**400) raises OverflowError
    with pytest.raises(FootprintError, match="too large"):
        validate_footprint([(0, 0), tuple(vertex), (1, 1)])


def test_int_padding_beyond_float_range_is_footprint_error():
    with pytest.raises(FootprintError, match="too large"):
        pad_footprint([(0, 0), (1, 0), (1, 1), (0, 1)], 10 ** 400)


def test_padding_is_deterministic_and_returns_fresh_lists():
    square = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    a = pad_footprint(square, 0.1)
    b = pad_footprint(square, 0.1)
    assert a == b and a is not b
    assert a == [(-0.6, -0.6), (0.6, -0.6), (0.6, 0.6), (-0.6, 0.6)]
