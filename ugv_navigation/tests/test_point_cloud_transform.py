"""Tests for costmap_core.point_cloud_transform.

Every pose, point, stamp and frame name here is SYNTHETIC. The real camera
extrinsic (Dev 5), TF tree (Dev 2) and frame names are PENDING.
"""

import math

import numpy as np
import pytest

from costmap_core.contracts import CameraGroundInput, ContractError, PointCloudInput
from costmap_core.point_cloud_transform import transform_point_cloud
from costmap_core.projection import (
    CameraGroundGeometry,
    CameraPose,
    ProjectionError,
    camera_pose_from_quaternion,
)

CAM = "test_camera_optical_frame"
GROUND = "test_ground_frame"
STAMP = 7_000_000_123
POSE_STAMP = 7_000_000_999  # deliberately different from the cloud stamp

# Level camera (zero pitch/roll/yaw): optical z-forward -> ground x,
# optical x-right -> ground -y, optical y-down -> ground -z.
OPTICAL_TO_LEVEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _cloud(points, *, frame_id=CAM, stamp_ns=STAMP):
    return PointCloudInput(
        points=np.asarray(points, dtype=np.float32).reshape(-1, 3),
        stamp_ns=stamp_ns,
        frame_id=frame_id,
    )


def _ground(rotation=np.eye(3), translation=(0.0, 0.0, 1.0), *, camera_frame_id=CAM,
            ground_frame_id=GROUND, stamp_ns=POSE_STAMP, geometry=None):
    if geometry is None:
        geometry = CameraPose(rotation=rotation, translation=translation)
    return CameraGroundInput(
        geometry=geometry,
        camera_frame_id=camera_frame_id,
        ground_frame_id=ground_frame_id,
        stamp_ns=stamp_ns,
    )


POINTS = np.array(
    [[0.0, 0.0, 1.0], [1.0, -2.0, 3.0], [-0.5, 0.25, 8.0], [2.0, 1.0, 0.5]], dtype=np.float32
)


# --- maths --------------------------------------------------------------------


def test_identity_rotation_with_unit_height_only_shifts_z():
    # CameraPose requires translation z > 0, so "identity" is R = I plus the
    # smallest legal lift; the rotation part leaves points unchanged.
    result = transform_point_cloud(_cloud(POINTS), _ground(np.eye(3), (0.0, 0.0, 1.0)))
    np.testing.assert_allclose(result.points, POINTS.astype(np.float64) + [0.0, 0.0, 1.0])


def test_pure_translation():
    t = (3.0, -4.0, 2.5)
    result = transform_point_cloud(_cloud(POINTS), _ground(np.eye(3), t))
    np.testing.assert_allclose(result.points, POINTS.astype(np.float64) + t)


def test_pure_rotation_about_z():
    rotation = _rot_z(math.pi / 2)  # x -> y, y -> -x
    result = transform_point_cloud(_cloud([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                                   _ground(rotation, (0.0, 0.0, 1.0)))
    np.testing.assert_allclose(
        result.points - [0.0, 0.0, 1.0],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


def test_rotation_plus_translation_matches_reference():
    rotation = _rot_z(0.3) @ _rot_x(-1.1) @ OPTICAL_TO_LEVEL
    t = np.array([0.4, -1.2, 2.0])
    result = transform_point_cloud(_cloud(POINTS), _ground(rotation, t))
    expected = np.array([rotation @ p + t for p in POINTS.astype(np.float64)])
    np.testing.assert_allclose(result.points, expected, atol=1e-12)


def test_quaternion_pose_from_tf_path_gives_same_result_as_matrix():
    s = math.sqrt(0.5)
    pose = camera_pose_from_quaternion(qx=0.0, qy=0.0, qz=s, qw=s, tx=1.0, ty=2.0, tz=1.5)
    result = transform_point_cloud(_cloud(POINTS), _ground(geometry=pose))
    expected = POINTS.astype(np.float64) @ _rot_z(math.pi / 2).T + [1.0, 2.0, 1.5]
    np.testing.assert_allclose(result.points, expected, atol=1e-12)


def test_camera_ground_geometry_is_accepted_via_existing_conversion():
    # 90 deg pitch = camera looking straight down from 2 m (synthetic).
    geometry = CameraGroundGeometry(camera_height=2.0, pitch_rad=math.pi / 2)
    result = transform_point_cloud(_cloud([[0.0, 0.0, 2.0]]), _ground(geometry=geometry))
    np.testing.assert_allclose(result.points, [[0.0, 0.0, 0.0]], atol=1e-12)


def test_optical_axes_level_camera():
    # Synthetic level camera 1.5 m above the ground-frame origin.
    ground = _ground(OPTICAL_TO_LEVEL, (0.0, 0.0, 1.5))
    optical = [
        [0.0, 0.0, 4.0],   # 4 m straight ahead
        [1.0, 0.0, 4.0],   # 1 m to the right (optical +x)
        [0.0, 1.0, 4.0],   # 1 m below (optical +y is down)
        [0.0, 1.5, 4.0],   # on the ground plane
    ]
    result = transform_point_cloud(_cloud(optical), ground)
    np.testing.assert_allclose(
        result.points,
        [[4.0, 0.0, 1.5], [4.0, -1.0, 1.5], [4.0, 0.0, 0.5], [4.0, 0.0, 0.0]],
        atol=1e-12,
    )


def test_optical_axes_pitched_camera():
    # Synthetic camera 1.0 m up, pitched 45 deg down: its optical axis hits
    # the ground 1.0 m ahead, at range sqrt(2).
    geometry = CameraGroundGeometry(camera_height=1.0, pitch_rad=math.pi / 4)
    result = transform_point_cloud(_cloud([[0.0, 0.0, math.sqrt(2.0)]]), _ground(geometry=geometry))
    np.testing.assert_allclose(result.points, [[1.0, 0.0, 0.0]], atol=1e-6)


# --- structure ---------------------------------------------------------------


def test_many_points_and_order_preserved():
    points = np.arange(3 * 50, dtype=np.float32).reshape(50, 3)
    t = np.array([10.0, 20.0, 30.0])
    result = transform_point_cloud(_cloud(points), _ground(np.eye(3), t))
    assert result.points.shape == (50, 3)
    np.testing.assert_allclose(result.points, points + t)
    # reversing the input reverses the output: no sorting or dedup
    reversed_result = transform_point_cloud(_cloud(points[::-1]), _ground(np.eye(3), t))
    np.testing.assert_allclose(reversed_result.points, result.points[::-1])


def test_duplicate_points_kept():
    result = transform_point_cloud(_cloud([[1.0, 1.0, 1.0]] * 3), _ground())
    assert result.points.shape == (3, 3)


def test_empty_cloud_stays_empty():
    result = transform_point_cloud(_cloud(np.empty((0, 3))), _ground(OPTICAL_TO_LEVEL))
    assert isinstance(result, PointCloudInput)
    assert result.points.shape == (0, 3)
    assert result.frame_id == GROUND
    assert result.stamp_ns == STAMP


def test_output_frame_is_ground_frame_and_stamp_is_cloud_stamp():
    result = transform_point_cloud(_cloud(POINTS), _ground(ground_frame_id="some_target"))
    assert result.frame_id == "some_target"
    assert result.stamp_ns == STAMP
    assert result.stamp_ns != POSE_STAMP


def test_input_not_mutated_and_output_detached_readonly():
    cloud = _cloud(POINTS)
    before = cloud.points.copy()
    result = transform_point_cloud(cloud, _ground(np.eye(3), (5.0, 5.0, 5.0)))
    np.testing.assert_array_equal(cloud.points, before)
    assert cloud.frame_id == CAM and cloud.stamp_ns == STAMP
    assert not result.points.flags.writeable
    assert not np.shares_memory(result.points, cloud.points)
    assert result.points.dtype == np.float64


def test_large_cloud_is_vectorised():
    rng = np.random.default_rng(1)
    points = rng.uniform(-40.0, 40.0, size=(640 * 480, 3)).astype(np.float32)
    rotation = _rot_z(0.7) @ _rot_x(-0.4) @ OPTICAL_TO_LEVEL
    t = np.array([1.0, 2.0, 1.2])
    result = transform_point_cloud(_cloud(points), _ground(rotation, t))
    np.testing.assert_allclose(result.points, points.astype(np.float64) @ rotation.T + t,
                               atol=1e-9)


# --- rejection -----------------------------------------------------------------


def test_camera_frame_mismatch_rejected():
    with pytest.raises(ContractError, match="frame_id"):
        transform_point_cloud(_cloud(POINTS, frame_id="other_camera"), _ground())


def test_wrong_types_rejected():
    with pytest.raises(ContractError, match="cloud"):
        transform_point_cloud(POINTS, _ground())
    with pytest.raises(ContractError, match="camera_ground"):
        transform_point_cloud(_cloud(POINTS), CameraPose(rotation=np.eye(3), translation=(0, 0, 1)))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_transform_rejected(bad):
    rotation = np.eye(3)
    rotation[0, 1] = bad
    with pytest.raises(ProjectionError, match="finite"):
        _ground(rotation)
    with pytest.raises(ProjectionError, match="finite"):
        _ground(np.eye(3), (0.0, bad, 1.0))


@pytest.mark.parametrize(
    "rotation",
    [
        np.diag([1.0, 1.0, -1.0]),           # reflection, det -1
        2.0 * np.eye(3),                     # scaled
        np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),  # sheared
        np.zeros((3, 3)),
    ],
)
def test_invalid_rotation_rejected(rotation):
    with pytest.raises(ProjectionError):
        _ground(rotation)


def test_non_unit_quaternion_rejected():
    with pytest.raises(ProjectionError, match="unit norm"):
        camera_pose_from_quaternion(qx=0.0, qy=0.0, qz=0.0, qw=2.0, tx=0.0, ty=0.0, tz=1.0)


def test_invalid_points_rejected_by_point_cloud_input():
    with pytest.raises(ContractError, match="finite"):
        _cloud([[0.0, math.nan, 1.0]])
