"""Vectorised project_mask_to_costmap must match the original per-pixel loop
(tests/mask_projection_reference.py) exactly: same cells, same costs, same
errors. All camera, grid and mask values are synthetic test values.
"""

import math

import numpy as np
import pytest

from costmap_core.class_to_cost import CostValues, InvalidSemanticClassError, SemanticClass
from costmap_core.grid import CostmapGridGeometry
from costmap_core.mask_projection import _pixel_rays, _rays_to_ground, project_mask_to_costmap
from costmap_core.projection import (
    CameraGroundGeometry,
    CameraIntrinsics,
    CameraPose,
    ProjectionError,
    camera_ground_geometry_to_pose,
    camera_pose_from_quaternion,
    project_pixel_to_ground,
)
from mask_projection_reference import reference_project_mask_to_costmap

UNKNOWN, TRAVERSABLE, HAZARD = (int(c) for c in SemanticClass)

IDENTITY_POSE = CameraPose(rotation=np.eye(3), translation=[0.0, 0.0, 1.0])  # optical z = up: all sky
STRAIGHT_DOWN = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def _quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return dict(
        qx=sr * cp * cy - cr * sp * sy,
        qy=cr * sp * cy + sr * cp * sy,
        qz=cr * cp * sy - sr * sp * cy,
        qw=cr * cp * cy + sr * sp * sy,
    )


def _level_optical_pose(*, roll=0.0, pitch=0.0, yaw=0.0, t=(0.0, 0.0, 1.0)):
    """Optical frame looking along ground +x, then rotated by body roll/pitch/yaw."""
    body = camera_pose_from_quaternion(**_quat_from_rpy(roll, pitch, yaw), tx=0.0, ty=0.0, tz=1.0)
    optical = camera_ground_geometry_to_pose(CameraGroundGeometry(camera_height=1.0, pitch_rad=0.0))
    return CameraPose(rotation=body.rotation @ optical.rotation, translation=list(t))


def _intrinsics(w, h, f_scale=0.8):
    return CameraIntrinsics(fx=f_scale * w, fy=f_scale * w * 1.03, cx=(w - 1) / 2, cy=(h - 1) / 2 + 0.25)


def _random_mask(h, w, seed=0, classes=(UNKNOWN, TRAVERSABLE, HAZARD)):
    return np.random.default_rng(seed).choice(np.array(classes, dtype=np.uint8), size=(h, w))


GRID = CostmapGridGeometry(resolution=0.1, origin_x=-1.0, origin_y=-4.0, width=90, height=80)


def assert_matches_reference(mask, intrinsics, geometry, grid, cost_values=None):
    kwargs = {} if cost_values is None else {"cost_values": cost_values}
    expected = reference_project_mask_to_costmap(mask, intrinsics, geometry, grid, **kwargs)
    actual = project_mask_to_costmap(mask, intrinsics, geometry, grid, **kwargs)
    assert actual.dtype == expected.dtype == np.int64
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    return actual


# --- ground points are bit-identical to the single-pixel helpers ----------------------


@pytest.mark.parametrize(
    "geometry",
    [
        CameraGroundGeometry(camera_height=1.3, pitch_rad=0.41),
        CameraPose(rotation=STRAIGHT_DOWN, translation=[0.3, -0.2, 2.0]),
        _level_optical_pose(roll=0.07, pitch=0.33, yaw=-0.6, t=(1.5, -0.7, 0.9)),
        _level_optical_pose(roll=-0.2, pitch=0.05, yaw=2.9, t=(-3.0, 4.0, 1.7)),
    ],
    ids=["legacy", "straight_down", "rotated_a", "rotated_b"],
)
def test_ground_points_bit_identical_to_single_pixel(geometry):
    h, w = 37, 53
    intr = _intrinsics(w, h)
    hits, xs, ys = _rays_to_ground(_pixel_rays((h, w), intr), geometry)
    k = 0
    for i in range(h * w):
        row, col = divmod(i, w)
        try:
            point = project_pixel_to_ground(float(col), float(row), intr, geometry)
        except ProjectionError:
            assert not hits[i]
            continue
        assert hits[i]
        assert (xs[k], ys[k]) == (point.x, point.y)  # exact, not approx
        k += 1
    assert k == len(xs)


# --- costmaps match the reference -------------------------------------------------------


def test_identity_pose_all_sky():
    mask = _random_mask(24, 32)
    result = assert_matches_reference(mask, _intrinsics(32, 24), IDENTITY_POSE, GRID)
    assert (result == CostValues().unknown_cost).all()


def test_straight_down_pose():
    assert_matches_reference(
        _random_mask(24, 32, seed=1), _intrinsics(32, 24),
        CameraPose(rotation=STRAIGHT_DOWN, translation=[0.0, 0.0, 1.0]), GRID,
    )


@pytest.mark.parametrize("t", [(2.0, 0.0, 1.0), (-0.5, 1.25, 0.6), (3.7, -2.2, 2.5)])
def test_translated_camera(t):
    assert_matches_reference(
        _random_mask(30, 40, seed=2), _intrinsics(40, 30), _level_optical_pose(pitch=0.4, t=t), GRID,
    )


@pytest.mark.parametrize(
    "rpy", [(0.0, 0.5, 0.0), (0.0, 0.3, 1.1), (0.15, 0.45, -0.8), (-0.3, 0.9, 3.0), (0.0, 1.5707963, 0.0)]
)
def test_rotated_camera(rpy):
    roll, pitch, yaw = rpy
    assert_matches_reference(
        _random_mask(30, 40, seed=3), _intrinsics(40, 30),
        _level_optical_pose(roll=roll, pitch=pitch, yaw=yaw, t=(0.5, -0.5, 1.2)), GRID,
    )


@pytest.mark.parametrize("pitch", [-0.3, 0.0, 0.2, 0.7, 1.4])
def test_legacy_ground_geometry(pitch):
    assert_matches_reference(
        _random_mask(30, 40, seed=4), _intrinsics(40, 30),
        CameraGroundGeometry(camera_height=1.1, pitch_rad=pitch), GRID,
    )


def test_out_of_bounds_points_dropped_not_clamped():
    small = CostmapGridGeometry(resolution=0.2, origin_x=1.0, origin_y=-0.5, width=6, height=5)
    result = assert_matches_reference(
        np.full((40, 60), HAZARD, dtype=np.uint8), _intrinsics(60, 40), _level_optical_pose(pitch=0.5), small,
    )
    assert (result == CostValues().hazard_cost).any()


def test_grid_entirely_out_of_view():
    far = CostmapGridGeometry(resolution=0.5, origin_x=-50.0, origin_y=-50.0, width=4, height=4)
    result = assert_matches_reference(
        _random_mask(20, 20), _intrinsics(20, 20), _level_optical_pose(pitch=0.5), far,
    )
    assert (result == CostValues().unknown_cost).all()


def test_points_exactly_on_cell_boundaries():
    # Straight down, fx = fy = 1, principal point on pixel 2: ground offsets are
    # exact multiples of the 1.0 m resolution, so every point lies on a cell edge.
    intr = CameraIntrinsics(fx=1.0, fy=1.0, cx=2.0, cy=2.0)
    pose = CameraPose(rotation=STRAIGHT_DOWN, translation=[0.0, 0.0, 1.0])
    grid = CostmapGridGeometry(resolution=1.0, origin_x=-2.0, origin_y=-2.0, width=4, height=4)
    assert_matches_reference(_random_mask(5, 5, seed=5), intr, pose, grid)


def test_all_unknown_mask():
    result = assert_matches_reference(
        np.zeros((30, 40), dtype=np.uint8), _intrinsics(40, 30), _level_optical_pose(pitch=0.5), GRID,
    )
    assert (result == CostValues().unknown_cost).all()


@pytest.mark.parametrize("classes", [(UNKNOWN, TRAVERSABLE), (UNKNOWN, HAZARD), (TRAVERSABLE, HAZARD),
                                     (UNKNOWN, TRAVERSABLE, HAZARD)])
def test_class_mixtures_and_precedence(classes):
    result = assert_matches_reference(
        _random_mask(60, 80, seed=6, classes=classes), _intrinsics(80, 60), _level_optical_pose(pitch=0.6), GRID,
    )
    present = {CostValues().as_mapping()[SemanticClass(c)] for c in classes}
    assert set(np.unique(result)) <= present | {CostValues().unknown_cost}


def test_edge_pixels_only():
    mask = np.full((30, 40), TRAVERSABLE, dtype=np.uint8)
    mask[0, :] = HAZARD
    mask[-1, :] = HAZARD
    mask[:, 0] = UNKNOWN
    mask[:, -1] = HAZARD
    assert_matches_reference(mask, _intrinsics(40, 30), _level_optical_pose(pitch=0.7), GRID)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 3), (3, 2)])
@pytest.mark.parametrize("value", [UNKNOWN, TRAVERSABLE, HAZARD])
def test_small_masks(shape, value):
    h, w = shape
    intr = CameraIntrinsics(fx=2.0, fy=2.0, cx=(w - 1) / 2, cy=(h - 1) / 2)
    pose = CameraPose(rotation=STRAIGHT_DOWN, translation=[0.05, 0.05, 1.0])
    grid = CostmapGridGeometry(resolution=0.1, origin_x=-1.0, origin_y=-1.0, width=20, height=20)
    assert_matches_reference(np.full(shape, value, dtype=np.uint8), intr, pose, grid)


def test_custom_cost_values():
    costs = CostValues(unknown_cost=200, traversable_cost=7, hazard_cost=254)
    assert_matches_reference(
        _random_mask(30, 40, seed=7), _intrinsics(40, 30), _level_optical_pose(pitch=0.5), GRID, costs,
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.int32, np.int64, np.uint16, np.float64])
def test_other_mask_dtypes(dtype):
    mask = _random_mask(12, 16, seed=8).astype(dtype)
    assert_matches_reference(mask, _intrinsics(16, 12), _level_optical_pose(pitch=0.5), GRID)


def test_float_mask_truncates_like_int():
    mask = np.array([[0.0, 1.9, 2.5], [1.0, 0.2, 2.0]])
    assert_matches_reference(mask, _intrinsics(3, 2), _level_optical_pose(pitch=0.5), GRID)


def test_bool_and_list_masks():
    assert_matches_reference(np.array([[True, False], [False, True]]), _intrinsics(2, 2),
                             _level_optical_pose(pitch=0.5), GRID)
    assert_matches_reference([[0, 1], [2, 1]], _intrinsics(2, 2), _level_optical_pose(pitch=0.5), GRID)


def test_randomised_scenes():
    rng = np.random.default_rng(1234)
    for _ in range(40):
        h, w = int(rng.integers(1, 30)), int(rng.integers(1, 30))
        intr = CameraIntrinsics(
            fx=float(rng.uniform(1, 60)), fy=float(rng.uniform(1, 60)),
            cx=float(rng.uniform(-2, w + 2)), cy=float(rng.uniform(-2, h + 2)),
        )
        if rng.random() < 0.3:
            geometry = CameraGroundGeometry(camera_height=float(rng.uniform(0.1, 3)),
                                            pitch_rad=float(rng.uniform(-0.5, 1.6)))
        else:
            geometry = _level_optical_pose(
                roll=float(rng.uniform(-0.5, 0.5)), pitch=float(rng.uniform(-0.3, 1.6)),
                yaw=float(rng.uniform(-3.2, 3.2)),
                t=(float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3)), float(rng.uniform(0.1, 3))),
            )
        grid = CostmapGridGeometry(
            resolution=float(rng.choice([0.05, 0.1, 0.25, 1.0])),
            origin_x=float(rng.uniform(-5, 1)), origin_y=float(rng.uniform(-5, 1)),
            width=int(rng.integers(1, 60)), height=int(rng.integers(1, 60)),
        )
        mask = rng.integers(0, 3, size=(h, w)).astype(np.uint8)
        assert_matches_reference(mask, intr, geometry, grid)


def test_representative_320x240_scene():
    w, h = 320, 240
    assert_matches_reference(
        _random_mask(h, w, seed=9), _intrinsics(w, h),
        _level_optical_pose(roll=0.02, pitch=0.35, yaw=0.3, t=(0.4, -0.3, 1.0)), GRID,
    )


# --- errors match the reference (type and order) --------------------------------------------


def _raised(fn, *args):
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001 - comparing whatever each raises
        return type(exc), str(exc)
    return None


POSE = _level_optical_pose(pitch=0.5)


@pytest.mark.parametrize(
    "mask, geometry",
    [
        (np.array([0, 1, 2], dtype=np.uint8), POSE),                   # 1D
        (np.zeros((2, 2, 2), dtype=np.uint8), POSE),                   # 3D
        (np.array([[0, 1, 3]], dtype=np.uint8), POSE),                 # invalid id later
        (np.array([[5, 1], [1, 1]], dtype=np.uint8), POSE),            # invalid id first
        (np.array([[1, -1]], dtype=np.int8), POSE),                    # negative id
        (np.array([[1, 1]], dtype=np.uint8), (1.0, 0.5)),              # wrong geometry type
        (np.array([[3, 1]], dtype=np.uint8), (1.0, 0.5)),              # bad first pixel beats geometry
        (np.array([[1, 3]], dtype=np.uint8), (1.0, 0.5)),              # geometry beats later bad pixel
        (np.zeros((0, 4), dtype=np.uint8), (1.0, 0.5)),                # empty mask: nothing checked
        (np.array([[1.0, math.nan]]), POSE),                           # int(nan)
        (np.array([[7.0, math.nan]]), POSE),                           # invalid id before int(nan)
        (np.array([[1, 2**64 - 1]], dtype=np.uint64), POSE),           # huge unsigned id
    ],
)
def test_errors_match_reference(mask, geometry):
    intr = _intrinsics(max(mask.shape[-1], 1), max(mask.shape[0], 1)) if mask.ndim >= 1 else _intrinsics(1, 1)
    expected = _raised(reference_project_mask_to_costmap, mask, intr, geometry, GRID)
    actual = _raised(project_mask_to_costmap, mask, intr, geometry, GRID)
    assert actual == expected


def test_empty_mask_returns_unknown_grid():
    result = assert_matches_reference(np.zeros((0, 3), dtype=np.uint8), _intrinsics(3, 1), POSE, GRID)
    assert (result == CostValues().unknown_cost).all()


def test_input_mask_not_modified():
    mask = _random_mask(10, 12)
    before = mask.copy()
    project_mask_to_costmap(mask, _intrinsics(12, 10), POSE, GRID)
    np.testing.assert_array_equal(mask, before)
