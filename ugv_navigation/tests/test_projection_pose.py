"""Tests for the 6-DoF CameraPose extension of costmap_core.projection.

EVERY pose, intrinsic and pixel value below is SYNTHETIC and chosen so the
expected result can be derived by hand. None are real camera or robot values.

Conventions under test:
- camera optical frame: x-right, y-down, z-forward;
- ground/world frame: x-forward, y-left, z-up, ground = z = 0 plane;
- CameraPose.rotation maps optical-frame directions into the ground frame,
  CameraPose.translation is the optical centre in the ground frame.
"""

import dataclasses
import math

import numpy as np
import pytest

from costmap_core.projection import (
    POSE_TOLERANCE,
    CameraGroundGeometry,
    CameraIntrinsics,
    CameraPose,
    ProjectionError,
    camera_ground_geometry_to_pose,
    camera_pose_from_quaternion,
    camera_ray_to_ground_point,
    project_pixel_to_ground,
)

# Optical (x-right, y-down, z-forward) -> level world (x-forward, y-left, z-up),
# written out by hand rather than imported from the module under test.
OPTICAL_TO_LEVEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# Straight-down camera, zero yaw: optical z -> world -z, image right (optical
# +x) -> world -y, image down (optical +y) -> world -x.
STRAIGHT_DOWN = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

INTR = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0)
INTR_ANISO = CameraIntrinsics(fx=100.0, fy=80.0, cx=50.0, cy=40.0)
PIXELS = [(u, v) for u in range(0, 101, 10) for v in range(0, 81, 8)]


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot2d(a, xy):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])


def forward_pose(pitch, *, yaw=0.0, roll=0.0, x=0.0, y=0.0, h=2.0):
    """Camera looking along world +x rotated by yaw, pitched down, rolled
    about its own optical axis."""
    rotation = rot_z(yaw) @ rot_y(pitch) @ OPTICAL_TO_LEVEL @ rot_z(roll)
    return CameraPose(rotation=rotation, translation=[x, y, h])


def ground_xy(u, v, geometry, intrinsics=INTR):
    p = project_pixel_to_ground(u, v, intrinsics, geometry)
    return np.array([p.x, p.y])


def try_ground_xy(u, v, geometry, intrinsics=INTR):
    try:
        return ground_xy(u, v, geometry, intrinsics)
    except ProjectionError:
        return None


# --- CameraPose validation ---------------------------------------------------


def test_valid_pose_stores_read_only_float_copies():
    rotation = np.eye(3, dtype=int)
    translation = np.array([1, 2, 3])
    pose = CameraPose(rotation=rotation, translation=translation)
    rotation[0, 0] = 5
    translation[2] = -9
    assert pose.rotation.dtype == np.float64 and pose.translation.dtype == np.float64
    np.testing.assert_array_equal(pose.rotation, np.eye(3))
    np.testing.assert_array_equal(pose.translation, [1.0, 2.0, 3.0])
    assert not pose.rotation.flags.writeable
    assert not pose.translation.flags.writeable
    with pytest.raises(ValueError):
        pose.rotation[0, 0] = 2.0


def test_pose_accepts_nested_lists():
    pose = CameraPose(rotation=STRAIGHT_DOWN.tolist(), translation=[0.0, 0.0, 1.0])
    np.testing.assert_array_equal(pose.rotation, STRAIGHT_DOWN)


def test_pose_is_frozen():
    pose = CameraPose(rotation=np.eye(3), translation=[0.0, 0.0, 1.0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        pose.translation = np.array([0.0, 0.0, 2.0])


@pytest.mark.parametrize(
    "rotation",
    [np.eye(2), np.eye(4), np.ones(9), np.eye(3)[:, :2], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 5.0],
)
def test_bad_rotation_shape_rejected(rotation):
    with pytest.raises(ProjectionError, match="shape"):
        CameraPose(rotation=rotation, translation=[0.0, 0.0, 1.0])


@pytest.mark.parametrize("translation", [[0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [[0.0], [0.0], [1.0]], 1.0])
def test_bad_translation_shape_rejected(translation):
    with pytest.raises(ProjectionError, match="shape"):
        CameraPose(rotation=np.eye(3), translation=translation)


@pytest.mark.parametrize(
    "rotation, translation",
    [
        (np.eye(3, dtype=bool), [0.0, 0.0, 1.0]),
        ([["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]], [0.0, 0.0, 1.0]),
        (None, [0.0, 0.0, 1.0]),
        (np.eye(3), [True, False, True]),
        (np.eye(3), None),
    ],
)
def test_non_numeric_pose_arrays_rejected(rotation, translation):
    with pytest.raises(ProjectionError, match="numeric"):
        CameraPose(rotation=rotation, translation=translation)


@pytest.mark.parametrize("index, bad", [((0, 0), math.nan), ((1, 2), math.inf), ((2, 1), -math.inf)])
def test_non_finite_rotation_rejected(index, bad):
    rotation = np.eye(3)
    rotation[index] = bad
    with pytest.raises(ProjectionError, match="finite"):
        CameraPose(rotation=rotation, translation=[0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "translation", [[math.nan, 0.0, 1.0], [0.0, math.inf, 1.0], [0.0, 0.0, -math.inf], [0.0, 0.0, math.nan]]
)
def test_non_finite_translation_rejected(translation):
    with pytest.raises(ProjectionError, match="finite"):
        CameraPose(rotation=np.eye(3), translation=translation)


@pytest.mark.parametrize(
    "rotation",
    [
        2.0 * np.eye(3),
        np.zeros((3, 3)),
        np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.ones((3, 3)),
    ],
)
def test_non_orthonormal_rotation_rejected(rotation):
    with pytest.raises(ProjectionError, match="orthonormal"):
        CameraPose(rotation=rotation, translation=[0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "rotation",
    [np.diag([1.0, 1.0, -1.0]), -np.eye(3), OPTICAL_TO_LEVEL * np.array([[1.0], [1.0], [-1.0]])],
)
def test_reflection_rejected(rotation):
    # Orthonormal but det = -1: a mirror image, not a rigid rotation.
    with pytest.raises(ProjectionError, match="determinant"):
        CameraPose(rotation=rotation, translation=[0.0, 0.0, 1.0])


def test_rotation_tolerance_boundary():
    within = np.eye(3)
    within[0, 1] = POSE_TOLERANCE / 100.0
    CameraPose(rotation=within, translation=[0.0, 0.0, 1.0])

    beyond = np.eye(3)
    beyond[0, 1] = POSE_TOLERANCE * 10.0
    with pytest.raises(ProjectionError, match="orthonormal"):
        CameraPose(rotation=beyond, translation=[0.0, 0.0, 1.0])


@pytest.mark.parametrize("z", [0.0, -1e-9, -2.0])
def test_camera_at_or_below_ground_rejected(z):
    with pytest.raises(ProjectionError, match="translation z"):
        CameraPose(rotation=np.eye(3), translation=[0.0, 0.0, z])


def test_camera_just_above_ground_accepted():
    CameraPose(rotation=np.eye(3), translation=[-3.0, 4.0, 1e-9])


# --- Quaternion -> CameraPose -------------------------------------------------

S = math.sqrt(0.5)


def quat_pose(q, t=(0.0, 0.0, 1.0)):
    return camera_pose_from_quaternion(qx=q[0], qy=q[1], qz=q[2], qw=q[3], tx=t[0], ty=t[1], tz=t[2])


@pytest.mark.parametrize(
    "q, expected",
    [
        ((0.0, 0.0, 0.0, 1.0), np.eye(3)),
        ((S, 0.0, 0.0, S), rot_x(math.pi / 2)),
        ((0.0, S, 0.0, S), rot_y(math.pi / 2)),
        ((0.0, 0.0, S, S), rot_z(math.pi / 2)),
        ((0.0, 0.0, 1.0, 0.0), np.diag([-1.0, -1.0, 1.0])),
        ((1.0, 0.0, 0.0, 0.0), np.diag([1.0, -1.0, -1.0])),
    ],
)
def test_quaternion_to_rotation_known_values(q, expected):
    np.testing.assert_allclose(quat_pose(q).rotation, expected, atol=1e-12)


def test_quaternion_matches_rodrigues_formula():
    axis = np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0)
    angle = 0.7
    q = (*(axis * math.sin(angle / 2)), math.cos(angle / 2))
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    expected = np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)
    np.testing.assert_allclose(quat_pose(q).rotation, expected, atol=1e-12)


def test_negated_quaternion_gives_same_rotation():
    q = (0.1, -0.5, 0.3, math.sqrt(1.0 - 0.01 - 0.25 - 0.09))
    np.testing.assert_allclose(
        quat_pose(q).rotation, quat_pose(tuple(-c for c in q)).rotation, atol=1e-15
    )


def test_quaternion_translation_passed_through():
    pose = quat_pose((0.0, 0.0, 0.0, 1.0), t=(1.5, -2.25, 3.0))
    np.testing.assert_array_equal(pose.translation, [1.5, -2.25, 3.0])


def test_near_unit_quaternion_is_normalised():
    pose = quat_pose((0.0, 0.0, 0.0, 1.0 + POSE_TOLERANCE / 10.0))
    np.testing.assert_allclose(pose.rotation, np.eye(3), atol=1e-15)


@pytest.mark.parametrize(
    "q",
    [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 0.0, 1.0 + 1e-3), (0.5, 0.5, 0.5, 0.4)],
)
def test_non_unit_quaternion_rejected(q):
    with pytest.raises(ProjectionError, match="unit norm"):
        quat_pose(q)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_quaternion_rejected(bad):
    with pytest.raises(ProjectionError, match="finite"):
        quat_pose((bad, 0.0, 0.0, 1.0))


@pytest.mark.parametrize("bad", [True, "1", None, [1.0]])
def test_non_real_quaternion_rejected(bad):
    with pytest.raises(ProjectionError, match="real number"):
        quat_pose((0.0, 0.0, 0.0, bad))


@pytest.mark.parametrize(
    "t, match",
    [
        ((0.0, 0.0, 0.0), "translation z"),
        ((0.0, 0.0, -1.0), "translation z"),
        ((math.nan, 0.0, 1.0), "finite"),
        ((0.0, True, 1.0), "real number"),
    ],
)
def test_quaternion_helper_rejects_bad_translation(t, match):
    with pytest.raises(ProjectionError, match=match):
        quat_pose((0.0, 0.0, 0.0, 1.0), t=t)


def test_quaternion_helper_is_keyword_only():
    with pytest.raises(TypeError):
        camera_pose_from_quaternion(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)


# --- CameraGroundGeometry -> CameraPose equivalence -----------------------------


def test_zero_pitch_geometry_converts_to_exact_axis_change():
    pose = camera_ground_geometry_to_pose(CameraGroundGeometry(camera_height=1.5, pitch_rad=0.0))
    np.testing.assert_array_equal(pose.rotation, OPTICAL_TO_LEVEL)
    np.testing.assert_array_equal(pose.translation, [0.0, 0.0, 1.5])


def test_downward_geometry_converts_to_straight_down_pose():
    pose = camera_ground_geometry_to_pose(CameraGroundGeometry(camera_height=2.0, pitch_rad=math.pi / 2))
    np.testing.assert_allclose(pose.rotation, STRAIGHT_DOWN, atol=1e-15)


@pytest.mark.parametrize("height", [0.5, 2.0, 7.25])
@pytest.mark.parametrize("pitch_deg", [-10.0, 0.0, 15.0, 30.0, 45.0, 90.0, 135.0])
def test_converted_pose_projects_like_legacy_geometry(height, pitch_deg):
    geometry = CameraGroundGeometry(camera_height=height, pitch_rad=math.radians(pitch_deg))
    pose = camera_ground_geometry_to_pose(geometry)
    hits = 0
    for u, v in PIXELS:
        legacy = try_ground_xy(u, v, geometry)
        new = try_ground_xy(u, v, pose)
        assert (legacy is None) == (new is None), (u, v)
        if legacy is not None:
            hits += 1
            np.testing.assert_allclose(new, legacy, rtol=1e-12, atol=1e-12)
    assert hits > 0


def test_geometry_to_pose_rejects_other_types():
    with pytest.raises(TypeError):
        camera_ground_geometry_to_pose((2.0, 0.5))


def test_legacy_geometry_path_unchanged():
    # Same manual expectation as test_projection.py, via the legacy type.
    geometry = CameraGroundGeometry(camera_height=2.0, pitch_rad=math.radians(30.0))
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    ray = np.array([0.2, 0.0, 1.0])
    point = camera_ray_to_ground_point(ray, geometry)
    assert point.y == pytest.approx(-0.8)
    assert project_pixel_to_ground(420.0, 240.0, intr, geometry) == point


# --- Translation, yaw, roll -----------------------------------------------------


def test_camera_translation_shifts_ground_points():
    base = forward_pose(math.radians(30.0))
    shifted = forward_pose(math.radians(30.0), x=3.5, y=-1.25)
    for u, v in PIXELS:
        a, b = try_ground_xy(u, v, base), try_ground_xy(u, v, shifted)
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_allclose(b, a + [3.5, -1.25], atol=1e-12)


def test_camera_height_scales_ground_points():
    low = forward_pose(math.radians(40.0), h=2.0)
    high = forward_pose(math.radians(40.0), h=4.0)
    for u, v in PIXELS:
        a = try_ground_xy(u, v, low)
        if a is not None:
            np.testing.assert_allclose(ground_xy(u, v, high), 2.0 * a, atol=1e-12)


@pytest.mark.parametrize("yaw", [math.pi / 2, -math.pi / 4, math.pi, 0.3])
def test_yaw_rotates_ground_points_about_camera_footprint(yaw):
    x0, y0 = -2.0, 5.0
    base = forward_pose(math.radians(35.0))
    yawed = forward_pose(math.radians(35.0), yaw=yaw, x=x0, y=y0)
    for u, v in PIXELS:
        a = try_ground_xy(u, v, base)
        if a is not None:
            expected = np.array([x0, y0]) + rot2d(yaw, a)
            np.testing.assert_allclose(ground_xy(u, v, yawed), expected, atol=1e-12)


def test_hand_derived_yawed_pitched_camera():
    # Camera at (1, 2, 2), yawed 90 deg (facing +y), pitched 45 deg down:
    # the principal ray hits the ground 2 m ahead, at (1, 4).
    pose = forward_pose(math.radians(45.0), yaw=math.pi / 2, x=1.0, y=2.0, h=2.0)
    np.testing.assert_allclose(ground_xy(INTR.cx, INTR.cy, pose), [1.0, 4.0], atol=1e-12)


@pytest.mark.parametrize("roll", [0.4, 1.3, math.pi / 2, -2.0])
def test_roll_leaves_principal_point_fixed(roll):
    base = forward_pose(math.radians(30.0), x=1.0, y=-1.0)
    rolled = forward_pose(math.radians(30.0), roll=roll, x=1.0, y=-1.0)
    np.testing.assert_allclose(
        ground_xy(INTR.cx, INTR.cy, rolled), ground_xy(INTR.cx, INTR.cy, base), atol=1e-12
    )
    # ...but moves an off-centre pixel.
    assert not np.allclose(ground_xy(80.0, 70.0, rolled), ground_xy(80.0, 70.0, base))


def test_roll_180_mirrors_the_image():
    # Rolling 180 deg about the optical axis maps pixel (u, v) to the base
    # camera's pixel (2cx - u, 2cy - v).
    base = forward_pose(math.radians(50.0), yaw=0.8, x=2.0, y=3.0)
    rolled = forward_pose(math.radians(50.0), yaw=0.8, x=2.0, y=3.0, roll=math.pi)
    for u, v in PIXELS:
        a = try_ground_xy(2 * INTR.cx - u, 2 * INTR.cy - v, base)
        b = try_ground_xy(u, v, rolled)
        assert (a is None) == (b is None), (u, v)
        if a is not None:
            np.testing.assert_allclose(b, a, atol=1e-9)


def test_roll_90_rotates_pixel_offsets():
    # With fx == fy, rolling +90 deg maps pixel offset (du, dv) to the base
    # camera's offset (-dv, du).
    base = forward_pose(math.radians(60.0), x=-1.0, y=0.5)
    rolled = forward_pose(math.radians(60.0), x=-1.0, y=0.5, roll=math.pi / 2)
    for u, v in PIXELS:
        du, dv = u - INTR.cx, v - INTR.cy
        a = try_ground_xy(INTR.cx - dv, INTR.cy + du, base)
        b = try_ground_xy(u, v, rolled)
        assert (a is None) == (b is None), (u, v)
        if a is not None:
            np.testing.assert_allclose(b, a, atol=1e-9)


# --- Straight-down camera at arbitrary x, y, yaw ---------------------------------


@pytest.mark.parametrize("h", [0.8, 3.0])
@pytest.mark.parametrize(
    "x0, y0, yaw", [(0.0, 0.0, 0.0), (3.0, -2.0, 0.6), (-7.5, 4.25, -2.2), (1.0, 1.0, math.pi)]
)
def test_straight_down_camera_hand_derived(x0, y0, yaw, h):
    # Pixel offset (du, dv) lands at (x0, y0) + R2(yaw) (-dv h / fy, -du h / fx).
    pose = CameraPose(rotation=rot_z(yaw) @ STRAIGHT_DOWN, translation=[x0, y0, h])
    np.testing.assert_allclose(ground_xy(INTR_ANISO.cx, INTR_ANISO.cy, pose, INTR_ANISO), [x0, y0], atol=1e-12)
    for du in (-30.0, 0.0, 17.0):
        for dv in (-20.0, 0.0, 33.0):
            got = ground_xy(INTR_ANISO.cx + du, INTR_ANISO.cy + dv, pose, INTR_ANISO)
            local = np.array([-dv * h / INTR_ANISO.fy, -du * h / INTR_ANISO.fx])
            np.testing.assert_allclose(got, np.array([x0, y0]) + rot2d(yaw, local), atol=1e-12)


# --- Sky / non-downward rays ----------------------------------------------------


def test_level_camera_rejects_horizon_and_sky_pixels():
    pose = forward_pose(0.0, yaw=0.7, x=2.0, y=3.0, h=1.5)
    with pytest.raises(ProjectionError, match="does not intersect"):
        ground_xy(INTR.cx, INTR.cy, pose)  # exactly horizontal
    with pytest.raises(ProjectionError, match="does not intersect"):
        ground_xy(INTR.cx, INTR.cy - 10.0, pose)  # above the horizon
    ground_xy(INTR.cx, INTR.cy + 10.0, pose)  # below the horizon: hits


def test_upward_pitched_camera_rejects_principal_ray():
    with pytest.raises(ProjectionError):
        ground_xy(INTR.cx, INTR.cy, forward_pose(math.radians(-20.0)))


def test_camera_facing_straight_up_sees_no_ground():
    pose = CameraPose(rotation=rot_y(-math.pi / 2) @ OPTICAL_TO_LEVEL, translation=[0.0, 0.0, 1.0])
    for u, v in PIXELS:
        with pytest.raises(ProjectionError):
            ground_xy(u, v, pose)


def test_rolled_level_camera_has_a_vertical_horizon():
    # Level camera rolled +90 deg: only the right half of the image
    # (u > cx) looks below the horizon.
    pose = forward_pose(0.0, roll=math.pi / 2)
    ground_xy(INTR.cx + 10.0, INTR.cy, pose)
    for u in (INTR.cx, INTR.cx - 10.0):
        with pytest.raises(ProjectionError):
            ground_xy(u, INTR.cy, pose)


def test_wrong_geometry_type_is_type_error_not_projection_error():
    ray = np.array([0.0, 0.0, 1.0])
    for bad in [(2.0, 0.5), None, np.eye(3)]:
        with pytest.raises(TypeError):
            camera_ray_to_ground_point(ray, bad)


# --- Round trip -------------------------------------------------------------------


def test_round_trip_pixel_ground_pixel_for_random_poses():
    rng = np.random.default_rng(12345)
    hits = 0
    for _ in range(300):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        t = (rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0), rng.uniform(0.2, 5.0))
        pose = quat_pose(tuple(q), t=t)
        u, v = rng.uniform(0.0, 100.0), rng.uniform(0.0, 80.0)
        xy = try_ground_xy(u, v, pose)
        if xy is None:
            continue
        hits += 1
        p_cam = pose.rotation.T @ (np.array([xy[0], xy[1], 0.0]) - pose.translation)
        assert p_cam[2] > 0.0  # the ground point is in front of the camera
        assert INTR.fx * p_cam[0] / p_cam[2] + INTR.cx == pytest.approx(u, abs=1e-6)
        assert INTR.fy * p_cam[1] / p_cam[2] + INTR.cy == pytest.approx(v, abs=1e-6)
    assert hits >= 50
