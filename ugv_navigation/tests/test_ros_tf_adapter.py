"""Tests for costmap_ros.tf_adapter (needs geometry_msgs; skipped without ROS).

All transforms are synthetic. Real TF (Dev 2 / Dev 5) and frame names are PENDING.
"""

import math

import numpy as np
import pytest

msg_module = pytest.importorskip("geometry_msgs.msg")
TransformStamped = msg_module.TransformStamped

from costmap_core.contracts import CameraGroundInput, ContractError  # noqa: E402
from costmap_core.projection import (  # noqa: E402
    CameraPose,
    ProjectionError,
    camera_pose_from_quaternion,
)
from costmap_ros.tf_adapter import (  # noqa: E402
    camera_ground_input_from_transform,
    camera_pose_from_transform,
)

GROUND_FRAME = "test_ground_frame"  # synthetic; the real frame names are PENDING
CAMERA_FRAME = "test_camera_frame"
S = math.sqrt(0.5)


def _tf_msg(*, t=(0.0, 0.0, 1.0), q=(0.0, 0.0, 0.0, 1.0), frame_id=GROUND_FRAME,
            child_frame_id=CAMERA_FRAME, sec=5, nanosec=250):
    msg = TransformStamped()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    tr, rot = msg.transform.translation, msg.transform.rotation
    tr.x, tr.y, tr.z = t
    rot.x, rot.y, rot.z, rot.w = q
    return msg


def test_identity_transform():
    pose = camera_pose_from_transform(_tf_msg())
    assert isinstance(pose, CameraPose)
    np.testing.assert_array_equal(pose.rotation, np.eye(3))
    np.testing.assert_array_equal(pose.translation, [0.0, 0.0, 1.0])


def test_translation_preserved():
    pose = camera_pose_from_transform(_tf_msg(t=(1.25, -3.5, 0.75)))
    np.testing.assert_array_equal(pose.translation, [1.25, -3.5, 0.75])


def test_known_rotation_yaw_90():
    # +90 deg about z: x -> y, y -> -x.
    pose = camera_pose_from_transform(_tf_msg(q=(0.0, 0.0, S, S)))
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(pose.rotation, expected, atol=1e-12)


def test_known_rotation_pitch_90():
    # +90 deg about y: z -> x (camera optical axis pointing along ground x).
    pose = camera_pose_from_transform(_tf_msg(q=(0.0, S, 0.0, S)))
    np.testing.assert_allclose(pose.rotation @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], atol=1e-12)


def test_quaternion_w_is_last():
    # (x=1, y=0, z=0, w=0) is 180 deg about x. Reading w first would give identity.
    pose = camera_pose_from_transform(_tf_msg(q=(1.0, 0.0, 0.0, 0.0)))
    np.testing.assert_allclose(pose.rotation, np.diag([1.0, -1.0, -1.0]), atol=1e-12)


def test_quaternion_xyz_order_matches_core():
    qx, qy, qz = 0.1, 0.2, 0.3
    qw = math.sqrt(1.0 - qx * qx - qy * qy - qz * qz)
    pose = camera_pose_from_transform(_tf_msg(t=(0.4, 0.5, 0.6), q=(qx, qy, qz, qw)))
    expected = camera_pose_from_quaternion(qx=qx, qy=qy, qz=qz, qw=qw, tx=0.4, ty=0.5, tz=0.6)
    np.testing.assert_array_equal(pose.rotation, expected.rotation)
    np.testing.assert_array_equal(pose.translation, expected.translation)
    swapped = camera_pose_from_quaternion(qx=qz, qy=qy, qz=qx, qw=qw, tx=0.4, ty=0.5, tz=0.6)
    assert not np.allclose(pose.rotation, swapped.rotation)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["tx", "ty", "tz", "qx", "qy", "qz", "qw"])
def test_non_finite_rejected(field, bad):
    t = {"tx": 0.0, "ty": 0.0, "tz": 1.0}
    q = {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    (t if field in t else q)[field] = bad
    msg = _tf_msg(t=tuple(t.values()), q=tuple(q.values()))
    with pytest.raises(ProjectionError, match="finite"):
        camera_pose_from_transform(msg)


@pytest.mark.parametrize("q", [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 2.0), (0.5, 0.5, 0.5, 0.0)])
def test_non_unit_quaternion_rejected(q):
    with pytest.raises(ProjectionError, match="unit norm"):
        camera_pose_from_transform(_tf_msg(q=q))


@pytest.mark.parametrize("tz", [0.0, -1.0])
def test_camera_not_above_ground_rejected(tz):
    with pytest.raises(ProjectionError, match="translation z"):
        camera_pose_from_transform(_tf_msg(t=(0.0, 0.0, tz)))


def test_wrong_message_type_rejected():
    with pytest.raises(ContractError, match="TransformStamped"):
        camera_pose_from_transform(msg_module.Transform())


def test_ground_input_preserves_frames_and_stamp():
    result = camera_ground_input_from_transform(
        _tf_msg(t=(1.0, 2.0, 3.0), frame_id="some_parent", child_frame_id="some_child")
    )
    assert isinstance(result, CameraGroundInput)
    assert result.ground_frame_id == "some_parent"
    assert result.camera_frame_id == "some_child"
    assert result.stamp_ns == 5 * 1_000_000_000 + 250
    np.testing.assert_array_equal(result.geometry.translation, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("frame_id, child_frame_id", [("", CAMERA_FRAME), (GROUND_FRAME, "")])
def test_ground_input_empty_frame_rejected(frame_id, child_frame_id):
    with pytest.raises(ContractError, match="frame_id"):
        camera_ground_input_from_transform(_tf_msg(frame_id=frame_id, child_frame_id=child_frame_id))


def test_ground_input_zero_stamp_rejected():
    with pytest.raises(ContractError, match="stamp_ns"):
        camera_ground_input_from_transform(_tf_msg(sec=0, nanosec=0))
