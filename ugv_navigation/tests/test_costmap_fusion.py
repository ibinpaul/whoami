import numpy as np
import pytest

from costmap_core.class_to_cost import DEFAULT_COST_VALUES
from costmap_core.costmap_fusion import CostmapFusionError, fuse_costmaps
from costmap_core.geometry_costmap import DEFAULT_GEOMETRY_COST_VALUES

# Explicit synthetic test data -- reuses the same default cost constants
# already defined in class_to_cost.py / geometry_costmap.py, not invented
# values.
UNKNOWN = DEFAULT_COST_VALUES.unknown_cost  # 255
TRAVERSABLE = DEFAULT_COST_VALUES.traversable_cost  # 0
HAZARD = DEFAULT_COST_VALUES.hazard_cost  # 254
LETHAL = DEFAULT_GEOMETRY_COST_VALUES.lethal_cost  # 254
FREE = DEFAULT_GEOMETRY_COST_VALUES.free_cost  # 0


def test_traversable_with_no_geometric_obstacle_stays_traversable():
    semantic = np.array([[TRAVERSABLE]])
    geometry = np.array([[FREE]])
    result = fuse_costmaps(semantic, geometry)
    assert result.tolist() == [[TRAVERSABLE]]


def test_hazard_with_no_geometric_obstacle_stays_hazard():
    semantic = np.array([[HAZARD]])
    geometry = np.array([[FREE]])
    result = fuse_costmaps(semantic, geometry)
    assert result.tolist() == [[HAZARD]]


def test_unknown_with_no_geometric_obstacle_stays_unknown():
    semantic = np.array([[UNKNOWN]])
    geometry = np.array([[FREE]])
    result = fuse_costmaps(semantic, geometry)
    assert result.tolist() == [[UNKNOWN]]


def test_traversable_with_geometric_lethal_becomes_lethal():
    # The critical rule: semantic traversable must never clear a
    # geometric lethal obstacle.
    semantic = np.array([[TRAVERSABLE]])
    geometry = np.array([[LETHAL]])
    result = fuse_costmaps(semantic, geometry)
    assert result.tolist() == [[LETHAL]]


def test_unknown_with_geometric_lethal_becomes_lethal():
    semantic = np.array([[UNKNOWN]])
    geometry = np.array([[LETHAL]])
    result = fuse_costmaps(semantic, geometry)
    assert result.tolist() == [[LETHAL]]


def test_hazard_with_geometric_lethal_stays_lethal():
    semantic = np.array([[HAZARD]])
    geometry = np.array([[LETHAL]])
    result = fuse_costmaps(semantic, geometry)
    assert result.tolist() == [[LETHAL]]


def test_mixed_matrix_covers_all_semantic_geometry_combinations():
    # rows: semantic {traversable, unknown, hazard}; cols: geometry {free, lethal}
    semantic = np.array(
        [
            [TRAVERSABLE, TRAVERSABLE],
            [UNKNOWN, UNKNOWN],
            [HAZARD, HAZARD],
        ]
    )
    geometry = np.array(
        [
            [FREE, LETHAL],
            [FREE, LETHAL],
            [FREE, LETHAL],
        ]
    )
    result = fuse_costmaps(semantic, geometry)
    expected = [
        [TRAVERSABLE, LETHAL],
        [UNKNOWN, LETHAL],
        [HAZARD, LETHAL],
    ]
    assert result.tolist() == expected


def test_shape_mismatch_raises_clear_error():
    semantic = np.zeros((2, 3))
    geometry = np.zeros((3, 2))
    with pytest.raises(CostmapFusionError):
        fuse_costmaps(semantic, geometry)


def test_non_2d_input_raises_clear_error():
    semantic_1d = np.array([TRAVERSABLE, HAZARD])
    geometry_1d = np.array([FREE, LETHAL])
    with pytest.raises(CostmapFusionError):
        fuse_costmaps(semantic_1d, geometry_1d)

    semantic_3d = np.zeros((2, 2, 2))
    geometry_3d = np.zeros((2, 2, 2))
    with pytest.raises(CostmapFusionError):
        fuse_costmaps(semantic_3d, geometry_3d)


def test_inputs_are_not_mutated():
    semantic = np.array([[TRAVERSABLE, HAZARD]])
    geometry = np.array([[LETHAL, FREE]])
    semantic_copy = semantic.copy()
    geometry_copy = geometry.copy()

    fuse_costmaps(semantic, geometry)

    assert np.array_equal(semantic, semantic_copy)
    assert np.array_equal(geometry, geometry_copy)


def test_result_is_a_new_array_not_a_view_into_the_inputs():
    semantic = np.array([[TRAVERSABLE]])
    geometry = np.array([[FREE]])
    result = fuse_costmaps(semantic, geometry)
    result[0, 0] = 999
    assert semantic[0, 0] == TRAVERSABLE
    assert geometry[0, 0] == FREE


def test_custom_lethal_cost_is_respected():
    semantic = np.array([[TRAVERSABLE, TRAVERSABLE]])
    geometry = np.array([[200, 100]])
    result = fuse_costmaps(semantic, geometry, lethal_cost=200)
    assert result.tolist() == [[200, TRAVERSABLE]]


# --- audit regressions ------------------------------------------------------


def test_bool_occupancy_passed_as_geometry_costmap_is_rejected():
    """A raw occupancy mask is not a cost array: it would compare unequal to
    lethal everywhere and silently drop every obstacle."""
    semantic = np.full((2, 2), TRAVERSABLE, dtype=np.int64)
    occupied = np.array([[True, False], [False, True]])
    with pytest.raises(CostmapFusionError, match="geometry_costmap must be an integer"):
        fuse_costmaps(semantic, occupied)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 2), dtype=bool),
        np.zeros((2, 2), dtype=np.float64),
        np.array([[np.nan, LETHAL], [FREE, FREE]]),
        np.array([["a", "b"], ["c", "d"]]),
        np.array([[None, None], [None, None]], dtype=object),
    ],
    ids=["bool", "float", "float_nan", "str", "object"],
)
@pytest.mark.parametrize("side", ["semantic", "geometry"])
def test_non_integer_cost_arrays_rejected(bad, side):
    good = np.zeros((2, 2), dtype=np.int64)
    args = (bad, good) if side == "semantic" else (good, bad)
    with pytest.raises(CostmapFusionError, match=f"{side}_costmap must be an integer"):
        fuse_costmaps(*args)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.int64, np.uint16])
def test_integer_dtypes_accepted(dtype):
    semantic = np.array([[TRAVERSABLE, UNKNOWN]], dtype=dtype)
    geometry = np.array([[LETHAL, LETHAL]], dtype=dtype)
    assert fuse_costmaps(semantic, geometry).tolist() == [[LETHAL, LETHAL]]


def test_every_semantic_cost_against_every_geometry_kind():
    """Exhaustive: semantic 0..255 x geometry {free, intermediate, lethal}."""
    semantic_values = np.arange(256, dtype=np.int64)
    for geometry_value in (FREE, 1, 128, 253, LETHAL):
        semantic = semantic_values.reshape(16, 16)
        geometry = np.full((16, 16), geometry_value, dtype=np.int64)
        fused = fuse_costmaps(semantic, geometry)
        if geometry_value == LETHAL:
            assert (fused == LETHAL).all()
        else:
            # Only geometry *lethal* overrides; anything else leaves semantics intact.
            np.testing.assert_array_equal(fused, semantic)


def test_random_grids_obey_geometry_lethal_precedence():
    rng = np.random.default_rng(1234)
    semantic_choices = np.array([TRAVERSABLE, HAZARD, UNKNOWN])
    for _ in range(50):
        shape = tuple(rng.integers(1, 12, size=2))
        semantic = rng.choice(semantic_choices, size=shape).astype(np.int64)
        occupied = rng.random(shape) < 0.3
        geometry = np.where(occupied, LETHAL, FREE).astype(np.int64)
        fused = fuse_costmaps(semantic, geometry)
        # Geometry lethal cells are lethal whatever the semantic class.
        assert (fused[occupied] == LETHAL).all()
        # Elsewhere the semantic cost is unchanged: free geometry clears nothing,
        # semantic hazard stays lethal and semantic unknown stays unknown.
        np.testing.assert_array_equal(fused[~occupied], semantic[~occupied])
        # Output only contains semantic values and lethal.
        assert set(np.unique(fused)) <= set(np.unique(semantic)) | {LETHAL}


def test_free_geometry_never_clears_semantic_lethal_or_unknown():
    semantic = np.array([[HAZARD, UNKNOWN], [UNKNOWN, HAZARD]], dtype=np.int64)
    geometry = np.full((2, 2), FREE, dtype=np.int64)
    np.testing.assert_array_equal(fuse_costmaps(semantic, geometry), semantic)


def test_rejected_inputs_are_not_mutated():
    semantic = np.full((2, 2), UNKNOWN, dtype=np.int64)
    occupied = np.array([[True, False], [False, True]])
    before_s, before_o = semantic.copy(), occupied.copy()
    with pytest.raises(CostmapFusionError):
        fuse_costmaps(semantic, occupied)
    np.testing.assert_array_equal(semantic, before_s)
    np.testing.assert_array_equal(occupied, before_o)


@pytest.mark.parametrize("bad", [True, 254.0, -1, 256, "254", None])
def test_invalid_fusion_lethal_cost_rejected(bad):
    semantic = np.zeros((1, 1), dtype=np.int64)
    with pytest.raises(CostmapFusionError, match="lethal_cost"):
        fuse_costmaps(semantic, semantic, lethal_cost=bad)


def test_bool_lethal_cost_cannot_match_occupancy_like_ones():
    """lethal_cost=True used to compare equal to geometry cost 1."""
    semantic = np.zeros((1, 2), dtype=np.int64)
    geometry = np.array([[1, 0]], dtype=np.int64)
    with pytest.raises(CostmapFusionError):
        fuse_costmaps(semantic, geometry, lethal_cost=True)


def test_numpy_integer_lethal_cost_accepted():
    semantic = np.array([[TRAVERSABLE]], dtype=np.int64)
    geometry = np.array([[LETHAL]], dtype=np.int64)
    assert fuse_costmaps(semantic, geometry, lethal_cost=np.int64(LETHAL)).tolist() == [[LETHAL]]
