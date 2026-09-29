"""Tests for costmap_ros.tf_lookup (needs tf2_ros; skipped without ROS).

All TF data is synthetic. Real TF (Dev 2 / Dev 5), frame names, timeout and
stale/extrapolation policy are PENDING.
"""

import math

import numpy as np
import pytest

tf2_ros = pytest.importorskip("tf2_ros")

from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.time import Time  # noqa: E402

from costmap_core.contracts import CameraGroundInput, ContractError, PointCloudInput  # noqa: E402
from costmap_core.projection import CameraPose, ProjectionError  # noqa: E402
from costmap_ros.tf_lookup import (  # noqa: E402
    TfCameraPoseLookup,
    TfLookupError,
    point_cloud_in_frame,
)

GROUND_FRAME = "test_ground_frame"  # synthetic; the real frame names are PENDING
CAMERA_FRAME = "test_camera_frame"
STAMP_NS = 5 * 1_000_000_000 + 250
TIMEOUT_S = 0.125  # synthetic; the real timeout is PENDING
S = math.sqrt(0.5)


def _tf_msg(*, t=(1.0, 2.0, 1.5), q=(0.0, 0.0, S, S), frame_id=GROUND_FRAME,
            child_frame_id=CAMERA_FRAME, stamp_ns=STAMP_NS):
    msg = TransformStamped()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    tr, rot = msg.transform.translation, msg.transform.rotation
    tr.x, tr.y, tr.z = t
    rot.x, rot.y, rot.z, rot.w = q
    return msg


class FakeBuffer:
    """Records lookup_transform calls; returns a fixed message or raises."""

    def __init__(self, *, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def lookup_transform(self, target_frame, source_frame, time, timeout):
        self.calls.append((target_frame, source_frame, time, timeout))
        if self.error is not None:
            raise self.error
        return self.result


def _lookup(buffer, **kwargs):
    params = {"target_frame": GROUND_FRAME, "source_frame": CAMERA_FRAME, "stamp_ns": STAMP_NS}
    params.update(kwargs)
    return TfCameraPoseLookup(buffer, timeout_s=TIMEOUT_S).lookup(**params)


# --- mocked Buffer ------------------------------------------------------------


def test_successful_lookup_returns_camera_ground_input():
    result = _lookup(FakeBuffer(result=_tf_msg()))
    assert isinstance(result, CameraGroundInput)
    assert isinstance(result.geometry, CameraPose)
    assert result.ground_frame_id == GROUND_FRAME
    assert result.camera_frame_id == CAMERA_FRAME
    assert result.stamp_ns == STAMP_NS


def test_target_and_source_frames_passed_in_order():
    buffer = FakeBuffer(result=_tf_msg(frame_id="frame_a", child_frame_id="frame_b"))
    _lookup(buffer, target_frame="frame_a", source_frame="frame_b")
    target, source, _, _ = buffer.calls[0]
    assert (target, source) == ("frame_a", "frame_b")


def test_requested_timestamp_passed():
    buffer = FakeBuffer(result=_tf_msg())
    _lookup(buffer)
    time = buffer.calls[0][2]
    assert isinstance(time, Time)
    assert time.nanoseconds == STAMP_NS


def test_configured_timeout_passed():
    buffer = FakeBuffer(result=_tf_msg())
    lookup = TfCameraPoseLookup(buffer, timeout_s=TIMEOUT_S)
    lookup.lookup(target_frame=GROUND_FRAME, source_frame=CAMERA_FRAME, stamp_ns=STAMP_NS)
    timeout = buffer.calls[0][3]
    assert isinstance(timeout, Duration)
    assert timeout.nanoseconds == 125_000_000
    assert lookup.timeout == timeout


def test_zero_timeout_allowed():
    buffer = FakeBuffer(result=_tf_msg())
    TfCameraPoseLookup(buffer, timeout_s=0.0).lookup(
        target_frame=GROUND_FRAME, source_frame=CAMERA_FRAME, stamp_ns=STAMP_NS
    )
    assert buffer.calls[0][3].nanoseconds == 0


def test_transform_converted_to_camera_pose():
    result = _lookup(FakeBuffer(result=_tf_msg(t=(1.0, 2.0, 1.5), q=(0.0, 0.0, S, S))))
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(result.geometry.rotation, expected, atol=1e-12)
    np.testing.assert_array_equal(result.geometry.translation, [1.0, 2.0, 1.5])


def test_returned_stamp_is_the_transform_stamp_not_the_request():
    # tf2 may interpolate and return a different stamp; it is carried, not judged.
    result = _lookup(FakeBuffer(result=_tf_msg(stamp_ns=STAMP_NS + 7)))
    assert result.stamp_ns == STAMP_NS + 7


@pytest.mark.parametrize(
    "error_cls",
    [
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
        tf2_ros.InvalidArgumentException,
        tf2_ros.TimeoutException,
        tf2_ros.TransformException,
    ],
)
def test_tf2_exceptions_wrapped(error_cls):
    cause = error_cls("synthetic failure")
    with pytest.raises(TfLookupError) as info:
        _lookup(FakeBuffer(error=cause))
    err = info.value
    assert err.cause is cause
    assert err.__cause__ is cause
    assert err.kind == error_cls.__name__
    assert (err.target_frame, err.source_frame, err.stamp_ns) == (GROUND_FRAME, CAMERA_FRAME, STAMP_NS)
    assert "synthetic failure" in str(err)


def test_non_tf_exceptions_not_swallowed():
    with pytest.raises(KeyError):
        _lookup(FakeBuffer(error=KeyError("bug")))


def test_invalid_pose_from_tf_is_projection_error_not_lookup_error():
    with pytest.raises(ProjectionError, match="translation z"):
        _lookup(FakeBuffer(result=_tf_msg(t=(0.0, 0.0, -1.0))))


def test_frames_returned_differ_from_request_rejected():
    buffer = FakeBuffer(result=_tf_msg(frame_id="other_frame"))
    with pytest.raises(ContractError, match="requested"):
        _lookup(buffer)


@pytest.mark.parametrize("field", ["target_frame", "source_frame"])
@pytest.mark.parametrize("bad", ["", None])
def test_bad_frame_argument_rejected_before_lookup(field, bad):
    buffer = FakeBuffer(result=_tf_msg())
    with pytest.raises(ContractError, match=field):
        _lookup(buffer, **{field: bad})
    assert buffer.calls == []


@pytest.mark.parametrize("stamp_ns", [0, -1, 1.5, True])
def test_bad_stamp_rejected_before_lookup(stamp_ns):
    buffer = FakeBuffer(result=_tf_msg())
    with pytest.raises(ContractError, match="stamp_ns"):
        _lookup(buffer, stamp_ns=stamp_ns)
    assert buffer.calls == []


@pytest.mark.parametrize("timeout_s", [-0.1, math.nan, math.inf, "1.0", None, True])
def test_bad_timeout_rejected(timeout_s):
    with pytest.raises(ContractError, match="timeout_s"):
        TfCameraPoseLookup(FakeBuffer(), timeout_s=timeout_s)


def test_timeout_is_required():
    with pytest.raises(TypeError):
        TfCameraPoseLookup(FakeBuffer())


def test_object_without_lookup_transform_rejected():
    with pytest.raises(ContractError, match="lookup_transform"):
        TfCameraPoseLookup(object(), timeout_s=TIMEOUT_S)


# --- real tf2_ros.Buffer with synthetic transforms ------------------------------


def _real_buffer(*stamps_ns):
    buffer = tf2_ros.Buffer()
    for stamp_ns in stamps_ns:
        buffer.set_transform(_tf_msg(stamp_ns=stamp_ns), "synthetic_test")
    return buffer


def test_real_buffer_successful_lookup():
    buffer = _real_buffer(STAMP_NS, STAMP_NS + 100_000_000)
    result = TfCameraPoseLookup(buffer, timeout_s=0.0).lookup(
        target_frame=GROUND_FRAME, source_frame=CAMERA_FRAME, stamp_ns=STAMP_NS + 50_000_000
    )
    np.testing.assert_allclose(result.geometry.translation, [1.0, 2.0, 1.5])
    assert result.stamp_ns == STAMP_NS + 50_000_000


def test_real_buffer_extrapolation_raises_lookup_error():
    buffer = _real_buffer(STAMP_NS, STAMP_NS + 100_000_000)
    with pytest.raises(TfLookupError) as info:
        TfCameraPoseLookup(buffer, timeout_s=0.0).lookup(
            target_frame=GROUND_FRAME, source_frame=CAMERA_FRAME, stamp_ns=STAMP_NS + 10_000_000_000
        )
    assert isinstance(info.value.cause, tf2_ros.ExtrapolationException)
    assert info.value.kind == "ExtrapolationException"


def test_real_buffer_unknown_frame_raises_lookup_error():
    buffer = _real_buffer(STAMP_NS)
    with pytest.raises(TfLookupError) as info:
        TfCameraPoseLookup(buffer, timeout_s=0.0).lookup(
            target_frame=GROUND_FRAME, source_frame="unknown_frame", stamp_ns=STAMP_NS
        )
    assert isinstance(info.value.cause, tf2_ros.TransformException)


# --- point_cloud_in_frame -------------------------------------------------------

CLOUD_POINTS = np.array([[0.0, 0.0, 2.0], [1.0, -1.0, 3.0]], dtype=np.float32)


def _point_cloud(*, frame_id=CAMERA_FRAME, stamp_ns=STAMP_NS, points=CLOUD_POINTS):
    return PointCloudInput(points=points, stamp_ns=stamp_ns, frame_id=frame_id)


def _expected_in_ground(points):
    # _tf_msg default: 90 deg about z, translation (1, 2, 1.5).
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return points.astype(np.float64) @ rotation.T + [1.0, 2.0, 1.5]


def test_cloud_lookup_uses_cloud_stamp_and_frames():
    buffer = FakeBuffer(result=_tf_msg())
    lookup = TfCameraPoseLookup(buffer, timeout_s=TIMEOUT_S)
    result = point_cloud_in_frame(lookup, _point_cloud(), target_frame=GROUND_FRAME)
    assert len(buffer.calls) == 1
    target, source, time, timeout = buffer.calls[0]
    assert (target, source) == (GROUND_FRAME, CAMERA_FRAME)
    assert time == Time(nanoseconds=STAMP_NS)
    assert time.nanoseconds != 0  # never "latest"
    assert timeout == Duration(nanoseconds=125_000_000)
    assert result.frame_id == GROUND_FRAME
    assert result.stamp_ns == STAMP_NS
    np.testing.assert_allclose(result.points, _expected_in_ground(CLOUD_POINTS), atol=1e-12)


def test_cloud_lookup_empty_cloud():
    lookup = TfCameraPoseLookup(FakeBuffer(result=_tf_msg()), timeout_s=TIMEOUT_S)
    result = point_cloud_in_frame(
        lookup, _point_cloud(points=np.empty((0, 3), dtype=np.float32)), target_frame=GROUND_FRAME
    )
    assert result.points.shape == (0, 3)
    assert result.frame_id == GROUND_FRAME


def test_cloud_lookup_failure_propagates_without_fallback():
    buffer = FakeBuffer(error=tf2_ros.LookupException("no such frame"))
    lookup = TfCameraPoseLookup(buffer, timeout_s=TIMEOUT_S)
    with pytest.raises(TfLookupError):
        point_cloud_in_frame(lookup, _point_cloud(), target_frame=GROUND_FRAME)


def test_cloud_lookup_wrong_frames_from_tf_rejected():
    buffer = FakeBuffer(result=_tf_msg(child_frame_id="other_camera"))
    lookup = TfCameraPoseLookup(buffer, timeout_s=TIMEOUT_S)
    with pytest.raises(ContractError):
        point_cloud_in_frame(lookup, _point_cloud(), target_frame=GROUND_FRAME)


@pytest.mark.parametrize("target_frame", ["", None, 5])
def test_cloud_lookup_requires_target_frame(target_frame):
    buffer = FakeBuffer(result=_tf_msg())
    lookup = TfCameraPoseLookup(buffer, timeout_s=TIMEOUT_S)
    with pytest.raises(ContractError):
        point_cloud_in_frame(lookup, _point_cloud(), target_frame=target_frame)
    assert buffer.calls == []


def test_cloud_lookup_target_frame_is_keyword_only_without_default():
    lookup = TfCameraPoseLookup(FakeBuffer(result=_tf_msg()), timeout_s=TIMEOUT_S)
    with pytest.raises(TypeError):
        point_cloud_in_frame(lookup, _point_cloud())
    with pytest.raises(TypeError):
        point_cloud_in_frame(lookup, _point_cloud(), GROUND_FRAME)


def test_cloud_lookup_rejects_wrong_types():
    lookup = TfCameraPoseLookup(FakeBuffer(result=_tf_msg()), timeout_s=TIMEOUT_S)
    with pytest.raises(ContractError):
        point_cloud_in_frame(FakeBuffer(result=_tf_msg()), _point_cloud(), target_frame=GROUND_FRAME)
    with pytest.raises(ContractError):
        point_cloud_in_frame(lookup, CLOUD_POINTS, target_frame=GROUND_FRAME)


def test_cloud_real_buffer_interpolated_at_cloud_stamp():
    buffer = _real_buffer(STAMP_NS, STAMP_NS + 100_000_000)
    lookup = TfCameraPoseLookup(buffer, timeout_s=0.0)
    cloud = _point_cloud(stamp_ns=STAMP_NS + 50_000_000)
    result = point_cloud_in_frame(lookup, cloud, target_frame=GROUND_FRAME)
    assert result.stamp_ns == STAMP_NS + 50_000_000
    np.testing.assert_allclose(result.points, _expected_in_ground(CLOUD_POINTS), atol=1e-9)


def test_cloud_real_buffer_stamp_outside_tf_history_raises():
    buffer = _real_buffer(STAMP_NS, STAMP_NS + 100_000_000)
    lookup = TfCameraPoseLookup(buffer, timeout_s=0.0)
    cloud = _point_cloud(stamp_ns=STAMP_NS + 10_000_000_000)
    with pytest.raises(TfLookupError):
        point_cloud_in_frame(lookup, cloud, target_frame=GROUND_FRAME)
