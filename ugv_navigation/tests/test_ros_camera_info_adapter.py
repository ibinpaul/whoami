"""Tests for costmap_ros.camera_info_adapter (needs sensor_msgs; skipped without ROS)."""

import math

import pytest

msg_module = pytest.importorskip("sensor_msgs.msg")
CameraInfo = msg_module.CameraInfo

from costmap_core.contracts import CameraIntrinsicsInput, ContractError  # noqa: E402
from costmap_core.projection import ProjectionError  # noqa: E402
from costmap_ros.camera_info_adapter import camera_intrinsics_from_camera_info  # noqa: E402

FRAME = "test_camera_frame"  # synthetic; the real frame name is PENDING


def _info_msg(*, fx=500.0, fy=510.0, cx=320.5, cy=240.25, k=None, width=640, height=480,
              frame_id=FRAME):
    msg = CameraInfo()
    msg.header.stamp.sec = 5
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.k = k if k is not None else [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    return msg


def test_camera_info_converts_to_intrinsics_input():
    result = camera_intrinsics_from_camera_info(_info_msg())
    assert isinstance(result, CameraIntrinsicsInput)
    assert (result.intrinsics.fx, result.intrinsics.fy) == (500.0, 510.0)
    assert (result.intrinsics.cx, result.intrinsics.cy) == (320.5, 240.25)
    assert (result.image_width, result.image_height) == (640, 480)


def test_frame_id_preserved():
    result = camera_intrinsics_from_camera_info(_info_msg(frame_id="some_optical_frame"))
    assert result.frame_id == "some_optical_frame"


def test_distortion_is_ignored_not_applied():
    msg = _info_msg()
    msg.distortion_model = "plumb_bob"
    msg.d = [0.1, -0.2, 0.001, 0.002, 0.0]
    result = camera_intrinsics_from_camera_info(msg)
    assert result.intrinsics.fx == 500.0


@pytest.mark.parametrize("fx, fy", [(0.0, 500.0), (500.0, 0.0), (-1.0, 500.0), (500.0, -1.0)])
def test_non_positive_focal_length_rejected(fx, fy):
    with pytest.raises(ProjectionError):
        camera_intrinsics_from_camera_info(_info_msg(fx=fx, fy=fy))


def test_all_zero_k_rejected():
    with pytest.raises(ContractError, match="all zeros"):
        camera_intrinsics_from_camera_info(_info_msg(k=[0.0] * 9))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_k_rejected(bad):
    with pytest.raises(ContractError, match="finite"):
        camera_intrinsics_from_camera_info(_info_msg(cx=bad))


@pytest.mark.parametrize(
    "k",
    [
        [500.0, 1.5, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],  # skew
        [500.0, 0.0, 320.0, 0.2, 500.0, 240.0, 0.0, 0.0, 1.0],  # K[1][0]
        [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 2.0],  # bottom row
        [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.1, 0.0, 1.0],
    ],
)
def test_non_pinhole_k_rejected(k):
    with pytest.raises(ContractError, match="zero-skew pinhole"):
        camera_intrinsics_from_camera_info(_info_msg(k=k))


@pytest.mark.parametrize("width, height", [(0, 480), (640, 0), (0, 0)])
def test_zero_image_size_rejected(width, height):
    with pytest.raises(ContractError, match="image_"):
        camera_intrinsics_from_camera_info(_info_msg(width=width, height=height))


def test_empty_frame_id_rejected():
    with pytest.raises(ContractError, match="frame_id"):
        camera_intrinsics_from_camera_info(_info_msg(frame_id=""))


def test_wrong_message_type_rejected():
    with pytest.raises(ContractError, match="CameraInfo"):
        camera_intrinsics_from_camera_info(msg_module.Image())


def test_identity_k_placeholder_rejected():
    msg = _info_msg()
    msg.k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ContractError, match="identity"):
        camera_intrinsics_from_camera_info(msg)


def test_unit_focal_length_with_real_principal_point_accepted():
    """Only the exact identity placeholder is refused, not fx = fy = 1 in general."""
    msg = _info_msg()
    msg.k = [1.0, 0.0, 3.5, 0.0, 1.0, 3.5, 0.0, 0.0, 1.0]
    assert camera_intrinsics_from_camera_info(msg).intrinsics.fx == 1.0


def test_camera_info_stamp_does_not_affect_conversion():
    old, new = _info_msg(), _info_msg()
    old.header.stamp.sec, new.header.stamp.sec = 1, 2_000_000_000
    assert camera_intrinsics_from_camera_info(old) == camera_intrinsics_from_camera_info(new)
