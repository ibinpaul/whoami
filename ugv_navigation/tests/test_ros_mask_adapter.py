"""Tests for costmap_ros.mask_adapter (needs sensor_msgs; skipped without ROS)."""

import numpy as np
import pytest

msg_module = pytest.importorskip("sensor_msgs.msg")
Image = msg_module.Image

from costmap_core.contracts import ContractError, SemanticMaskInput  # noqa: E402
from costmap_ros.mask_adapter import semantic_mask_from_image, stamp_to_ns  # noqa: E402

FRAME = "test_camera_frame"  # synthetic; the real frame name is PENDING


def _mask_msg(classes, *, encoding="mono8", step=None, sec=12, nanosec=345, frame_id=FRAME):
    classes = np.asarray(classes, dtype=np.uint8)
    msg = Image()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = frame_id
    msg.height, msg.width = classes.shape
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = classes.shape[1] if step is None else step
    msg.data = classes.tobytes()
    return msg


def test_mono8_mask_converts_to_semantic_mask_input():
    classes = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    result = semantic_mask_from_image(_mask_msg(classes), valid=True)
    assert isinstance(result, SemanticMaskInput)
    assert result.shape == (2, 3)
    assert result.classes.dtype == np.uint8
    assert result.valid is True


def test_canonical_values_preserved_pixel_for_pixel():
    classes = np.array([[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]], dtype=np.uint8)
    result = semantic_mask_from_image(_mask_msg(classes), valid=True)
    np.testing.assert_array_equal(result.classes, classes)


def test_all_unknown_mask_is_still_a_valid_sample():
    result = semantic_mask_from_image(_mask_msg(np.zeros((2, 2))), valid=True)
    assert np.all(result.classes == 0)


def test_stamp_and_frame_id_taken_from_header():
    result = semantic_mask_from_image(
        _mask_msg([[1]], sec=1_700_000_000, nanosec=123_456_789, frame_id="some_optical_frame"),
        valid=True,
    )
    assert result.stamp_ns == 1_700_000_000 * 1_000_000_000 + 123_456_789
    assert result.frame_id == "some_optical_frame"


def test_valid_flag_is_passed_through_not_invented():
    result = semantic_mask_from_image(_mask_msg([[1]]), valid=False)
    assert result.valid is False


def test_valid_has_no_default():
    with pytest.raises(TypeError):
        semantic_mask_from_image(_mask_msg([[1]]))


def test_stamp_to_ns():
    msg = _mask_msg([[0]], sec=3, nanosec=7)
    assert stamp_to_ns(msg.header.stamp) == 3_000_000_007


@pytest.mark.parametrize("encoding", ["mono16", "rgb8", "bgr8", "8UC1", "MONO8", ""])
def test_non_mono8_encoding_rejected(encoding):
    with pytest.raises(ContractError, match="encoding"):
        semantic_mask_from_image(_mask_msg([[0, 1]], encoding=encoding), valid=True)


@pytest.mark.parametrize("bad_id", [3, 4, 24, 255])
def test_non_canonical_ids_rejected_not_remapped(bad_id):
    with pytest.raises(ContractError, match="non-canonical"):
        semantic_mask_from_image(_mask_msg([[0, 1], [2, bad_id]]), valid=True)


def test_row_padding_rejected():
    msg = _mask_msg([[0, 1]], step=4)
    msg.data = bytes([0, 1, 0, 0])
    with pytest.raises(ContractError, match="step"):
        semantic_mask_from_image(msg, valid=True)


def test_data_size_mismatch_rejected():
    msg = _mask_msg([[0, 1], [1, 0]])
    msg.data = bytes([0, 1, 1])
    with pytest.raises(ContractError, match="bytes"):
        semantic_mask_from_image(msg, valid=True)


def test_empty_image_rejected():
    msg = Image()
    msg.encoding = "mono8"
    msg.header.stamp.sec = 1
    msg.header.frame_id = FRAME
    with pytest.raises(ContractError, match="size"):
        semantic_mask_from_image(msg, valid=True)


def test_zero_stamp_rejected():
    with pytest.raises(ContractError, match="stamp_ns"):
        semantic_mask_from_image(_mask_msg([[0]], sec=0, nanosec=0), valid=True)


def test_empty_frame_id_rejected():
    with pytest.raises(ContractError, match="frame_id"):
        semantic_mask_from_image(_mask_msg([[0]], frame_id=""), valid=True)


def test_wrong_message_type_rejected():
    with pytest.raises(ContractError, match="Image"):
        semantic_mask_from_image(msg_module.CameraInfo(), valid=True)


def test_result_does_not_alias_message_buffer():
    msg = _mask_msg([[0, 1], [2, 0]])
    result = semantic_mask_from_image(msg, valid=True)
    assert not result.classes.flags.writeable
    msg.data = bytes([2, 2, 2, 2])
    np.testing.assert_array_equal(result.classes, [[0, 1], [2, 0]])
