"""Tests for costmap_core.point_cloud_filter.

Every threshold, pose, point, stamp and frame name here is SYNTHETIC. The
real obstacle height band and range limits are PENDING (team decision).
"""

import dataclasses
import math

import numpy as np
import pytest

from costmap_core.contracts import CameraGroundInput, ContractError, PointCloudInput
from costmap_core.point_cloud_filter import ObstacleFilterParams, filter_obstacle_points
from costmap_core.point_cloud_transform import transform_point_cloud
from costmap_core.projection import CameraGroundGeometry, CameraPose

CAM = "test_camera_optical_frame"
GROUND = "test_ground_frame"
STAMP = 9_000_000_042

# Synthetic camera origin in the ground frame: 1.0 m above the origin.
ORIGIN = (0.0, 0.0, 1.0)
# Synthetic thresholds: keep 0.5 m .. 2.0 m high, 1.0 m .. 10.0 m away.
PARAMS = dict(min_height=0.5, max_height=2.0, min_range=1.0, max_range=10.0)

# Level camera: optical z-forward -> ground x, x-right -> -y, y-down -> -z.
OPTICAL_TO_LEVEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _params(**overrides):
    kwargs = dict(PARAMS)
    kwargs.update(overrides)
    return ObstacleFilterParams(**kwargs)


def _camera_ground(*, translation=ORIGIN, rotation=OPTICAL_TO_LEVEL, geometry=None,
                   ground_frame_id=GROUND):
    if geometry is None:
        geometry = CameraPose(rotation=rotation, translation=translation)
    return CameraGroundInput(
        geometry=geometry, camera_frame_id=CAM, ground_frame_id=ground_frame_id, stamp_ns=STAMP
    )


def _cloud(points, *, frame_id=GROUND, dtype=np.float64):
    return PointCloudInput(
        points=np.asarray(points, dtype=dtype).reshape(-1, 3), stamp_ns=STAMP, frame_id=frame_id
    )


def _filter(points, **param_overrides):
    return filter_obstacle_points(_cloud(points), _camera_ground(), _params(**param_overrides))


# --- selection -------------------------------------------------------------------


def test_obstacle_points_pass():
    points = [[3.0, 0.0, 1.0], [5.0, 2.0, 0.75], [2.0, -1.0, 1.9]]
    result = _filter(points)
    np.testing.assert_array_equal(result.points, points)


def test_ground_points_are_dropped_not_marked():
    ground = [[3.0, 0.0, 0.0], [4.0, 1.0, 0.1], [6.0, -2.0, 0.49]]
    obstacle = [[3.0, 0.0, 1.0]]
    result = _filter(ground + obstacle)
    np.testing.assert_array_equal(result.points, obstacle)


def test_points_above_band_dropped():
    result = _filter([[3.0, 0.0, 2.01], [3.0, 0.0, 5.0]])
    assert result.points.shape == (0, 3)


def test_height_boundaries_inclusive():
    result = _filter([[3.0, 0.0, 0.5], [3.0, 0.0, 2.0]])
    np.testing.assert_array_equal(result.points, [[3.0, 0.0, 0.5], [3.0, 0.0, 2.0]])
    below = np.nextafter(0.5, -np.inf)
    above = np.nextafter(2.0, np.inf)
    assert _filter([[3.0, 0.0, below], [3.0, 0.0, above]]).points.shape == (0, 3)


def test_range_boundaries_min_inclusive_max_exclusive():
    # Points level with the camera (z = 1.0) so distance = |x|.
    result = _filter([[1.0, 0.0, 1.0], [10.0, 0.0, 1.0]])
    np.testing.assert_array_equal(result.points, [[1.0, 0.0, 1.0]])
    near = np.nextafter(1.0, -np.inf)
    far = np.nextafter(10.0, -np.inf)
    result = _filter([[near, 0.0, 1.0], [far, 0.0, 1.0]])
    np.testing.assert_array_equal(result.points, [[far, 0.0, 1.0]])


def test_range_is_3d_distance_from_camera_origin_not_frame_origin():
    # Camera at (5, 5, 1). The point (5, 5, 1.5) is 0.5 m from the camera
    # (too near) although 7+ m from the ground-frame origin.
    camera_ground = _camera_ground(translation=(5.0, 5.0, 1.0))
    near_camera = [[5.0, 5.0, 1.5]]
    result = filter_obstacle_points(_cloud(near_camera), camera_ground, _params())
    assert result.points.shape == (0, 3)
    # 3D: (8, 5, 2) is sqrt(9 + 1) = 3.16 m away -> kept.
    result = filter_obstacle_points(_cloud([[8.0, 5.0, 2.0]]), camera_ground, _params())
    assert result.points.shape == (1, 3)


def test_combined_criteria_each_reject_independently():
    points = [
        [3.0, 0.0, 1.0],    # kept
        [3.0, 0.0, 0.2],    # too low
        [3.0, 0.0, 2.5],    # too high
        [0.5, 0.0, 1.0],    # too near
        [20.0, 0.0, 1.0],   # too far
        [0.5, 0.0, 0.1],    # too near and too low
        [4.0, 3.0, 1.5],    # kept (5.02 m)
    ]
    result = _filter(points)
    np.testing.assert_array_equal(result.points, [[3.0, 0.0, 1.0], [4.0, 3.0, 1.5]])


def test_negative_height_band_is_allowed_when_supplied():
    # A caller may choose a band below the ground plane; nothing is assumed.
    result = _filter([[3.0, 0.0, -0.5], [3.0, 0.0, 1.0]], min_height=-1.0, max_height=-0.1)
    np.testing.assert_array_equal(result.points, [[3.0, 0.0, -0.5]])


def test_all_accepted():
    points = [[3.0, 0.0, 1.0], [4.0, 0.0, 1.0], [5.0, 0.0, 1.0]]
    np.testing.assert_array_equal(_filter(points).points, points)


def test_all_rejected_gives_empty_cloud_with_metadata():
    result = _filter([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    assert isinstance(result, PointCloudInput)
    assert result.points.shape == (0, 3)
    assert result.stamp_ns == STAMP
    assert result.frame_id == GROUND


def test_empty_input():
    result = _filter(np.empty((0, 3)))
    assert result.points.shape == (0, 3)
    assert result.stamp_ns == STAMP
    assert result.frame_id == GROUND


def test_empty_result_is_points_not_occupancy():
    # "No obstacle points" is a point set, never an occupancy grid, so it
    # cannot clear cells.
    from costmap_core.contracts import OccupancyInput

    result = _filter([[3.0, 0.0, 0.0]])
    assert not isinstance(result, OccupancyInput)


# --- metadata and data safety --------------------------------------------------


def test_stamp_and_frame_preserved():
    cloud = PointCloudInput(points=np.array([[3.0, 0.0, 1.0]]), stamp_ns=123_456_789_000,
                            frame_id="some_ground")
    result = filter_obstacle_points(cloud, _camera_ground(ground_frame_id="some_ground"), _params())
    assert result.stamp_ns == 123_456_789_000
    assert result.frame_id == "some_ground"


def test_order_preserved():
    points = np.array([[x, 0.0, 1.0] for x in (9.0, 2.0, 0.1, 7.0, 3.0, 50.0, 4.0)])
    result = _filter(points)
    np.testing.assert_array_equal(result.points[:, 0], [9.0, 2.0, 7.0, 3.0, 4.0])


def test_dtype_preserved():
    result = filter_obstacle_points(
        _cloud([[3.0, 0.0, 1.0]], dtype=np.float32), _camera_ground(), _params()
    )
    assert result.points.dtype == np.float32


def test_input_not_mutated_output_detached_readonly():
    cloud = _cloud([[3.0, 0.0, 1.0], [3.0, 0.0, 0.0]])
    before = cloud.points.copy()
    result = filter_obstacle_points(cloud, _camera_ground(), _params())
    np.testing.assert_array_equal(cloud.points, before)
    assert not result.points.flags.writeable
    assert not np.shares_memory(result.points, cloud.points)


def test_camera_ground_geometry_accepted():
    # Synthetic simplified pose: camera 1.0 m above the origin.
    camera_ground = _camera_ground(geometry=CameraGroundGeometry(camera_height=1.0, pitch_rad=0.3))
    result = filter_obstacle_points(_cloud([[0.5, 0.0, 1.0], [3.0, 0.0, 1.0]]),
                                    camera_ground, _params())
    np.testing.assert_array_equal(result.points, [[3.0, 0.0, 1.0]])


def test_end_to_end_with_transform():
    # Synthetic level camera 1.0 m up: optical (0, 0, 4) is 4 m ahead at
    # camera height (kept); optical (0, 1, 4) is on the ground (dropped).
    camera_ground = _camera_ground()
    optical = PointCloudInput(
        points=np.array([[0.0, 0.0, 4.0], [0.0, 1.0, 4.0]], dtype=np.float32),
        stamp_ns=STAMP,
        frame_id=CAM,
    )
    in_ground = transform_point_cloud(optical, camera_ground)
    result = filter_obstacle_points(in_ground, camera_ground, _params())
    np.testing.assert_allclose(result.points, [[4.0, 0.0, 1.0]], atol=1e-12)


def test_large_cloud_vectorised():
    rng = np.random.default_rng(2)
    points = rng.uniform([-20.0, -20.0, -1.0], [20.0, 20.0, 3.0], size=(640 * 480, 3))
    result = _filter(points)
    d = np.linalg.norm(points - np.array(ORIGIN), axis=1)
    keep = (points[:, 2] >= 0.5) & (points[:, 2] <= 2.0) & (d >= 1.0) & (d < 10.0)
    assert 0 < keep.sum() < len(points)
    np.testing.assert_array_equal(result.points, points[keep])


# --- rejection -------------------------------------------------------------------


def test_untransformed_optical_cloud_rejected():
    with pytest.raises(ContractError, match="transform the cloud first"):
        filter_obstacle_points(_cloud([[0.0, 0.0, 4.0]], frame_id=CAM), _camera_ground(), _params())


def test_wrong_types_rejected():
    with pytest.raises(ContractError, match="cloud"):
        filter_obstacle_points(np.zeros((1, 3)), _camera_ground(), _params())
    with pytest.raises(ContractError, match="camera_ground"):
        filter_obstacle_points(_cloud([[3.0, 0.0, 1.0]]), None, _params())
    with pytest.raises(ContractError, match="params"):
        filter_obstacle_points(_cloud([[3.0, 0.0, 1.0]]), _camera_ground(), dict(PARAMS))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(min_height=2.0, max_height=2.0),
        dict(min_height=3.0, max_height=2.0),
        dict(min_range=10.0, max_range=10.0),
        dict(min_range=11.0, max_range=10.0),
        dict(min_range=-0.1),
    ],
)
def test_invalid_threshold_relations_rejected(overrides):
    with pytest.raises(ContractError):
        _params(**overrides)


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_thresholds_rejected(name, bad):
    with pytest.raises(ContractError, match="finite"):
        _params(**{name: bad})


@pytest.mark.parametrize("name", list(PARAMS))
@pytest.mark.parametrize("bad", [None, "1.0", True, [1.0], np.array([1.0])])
def test_non_real_thresholds_rejected(name, bad):
    with pytest.raises(ContractError):
        _params(**{name: bad})


def test_integer_thresholds_are_stored_as_float():
    params = ObstacleFilterParams(min_height=0, max_height=2, min_range=0, max_range=10)
    assert all(type(getattr(params, f.name)) is float for f in dataclasses.fields(params))


def test_params_have_no_defaults():
    for field in dataclasses.fields(ObstacleFilterParams):
        assert field.default is dataclasses.MISSING, field.name
        assert field.default_factory is dataclasses.MISSING, field.name
    with pytest.raises(TypeError):
        ObstacleFilterParams()


def test_filter_has_no_default_params():
    with pytest.raises(TypeError):
        filter_obstacle_points(_cloud([[3.0, 0.0, 1.0]]), _camera_ground())
