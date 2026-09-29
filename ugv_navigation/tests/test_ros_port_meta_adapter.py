"""Tests for costmap_ros.port_meta_adapter (needs std_msgs; skipped without ROS)."""

import dataclasses
import math

import pytest

msg_module = pytest.importorskip("std_msgs.msg")
Float64MultiArray = msg_module.Float64MultiArray
MultiArrayDimension = msg_module.MultiArrayDimension
MultiArrayLayout = msg_module.MultiArrayLayout

from costmap_core.contracts import ContractError  # noqa: E402
from costmap_ros.port_meta_adapter import (  # noqa: E402
    PortMetaStopgap,
    port_meta_from_float64_multi_array,
)


def _dev1_meta(data, labels=("valid", "age", "scale"), data_offset=0):
    """Same construction as Dev 1's ugv_perception/node/wire.py (not imported)."""
    msg = Float64MultiArray()
    msg.layout = MultiArrayLayout(
        dim=[
            MultiArrayDimension(label=label, size=1, stride=len(labels) - i)
            for i, label in enumerate(labels)
        ],
        data_offset=data_offset,
    )
    msg.data = [float(x) for x in data]
    return msg


def test_decodes_dev1_published_meta():
    result = port_meta_from_float64_multi_array(_dev1_meta([1.0, 0.01, 1.0]))
    assert result == PortMetaStopgap(valid=True, age_s=0.01, scale=1.0)
    assert result.valid is True


def test_valid_zero_decodes_to_false_not_coerced_to_true():
    result = port_meta_from_float64_multi_array(_dev1_meta([0.0, 0.01, 1.0]))
    assert result.valid is False


def test_zero_age_accepted():
    assert port_meta_from_float64_multi_array(_dev1_meta([1.0, 0.0, 1.0])).age_s == 0.0


def test_stale_age_is_carried_not_judged():
    # age is informational; freshness is decided later from header.stamp.
    result = port_meta_from_float64_multi_array(_dev1_meta([1.0, 5.0, 1.0]))
    assert result.age_s == 5.0


def test_result_has_no_stamp_or_frame():
    fields = {f.name for f in dataclasses.fields(PortMetaStopgap)}
    assert fields == {"valid", "age_s", "scale"}


@pytest.mark.parametrize("raw_valid", [0.5, 2.0, -1.0, 0.999, math.nan, math.inf])
def test_non_boolean_valid_rejected(raw_valid):
    with pytest.raises(ContractError, match="valid"):
        port_meta_from_float64_multi_array(_dev1_meta([raw_valid, 0.01, 1.0]))


@pytest.mark.parametrize("age", [-0.001, math.nan, math.inf, -math.inf])
def test_invalid_age_rejected(age):
    with pytest.raises(ContractError, match="age"):
        port_meta_from_float64_multi_array(_dev1_meta([1.0, age, 1.0]))


@pytest.mark.parametrize("scale", [0.5, 2.0, 0.0, -1.0, math.nan, math.inf])
def test_scale_other_than_one_rejected(scale):
    with pytest.raises(ContractError, match="scale"):
        port_meta_from_float64_multi_array(_dev1_meta([1.0, 0.01, scale]))


@pytest.mark.parametrize("data", [[], [1.0], [1.0, 0.01], [1.0, 0.01, 1.0, 0.0]])
def test_wrong_data_length_rejected(data):
    with pytest.raises(ContractError, match="3 values"):
        port_meta_from_float64_multi_array(_dev1_meta(data))


@pytest.mark.parametrize(
    "labels",
    [
        (),
        ("valid", "age"),
        ("age", "valid", "scale"),
        ("valid", "age", "scale", "adapter_id"),
        ("valid", "age_s", "scale"),
    ],
)
def test_unexpected_layout_labels_rejected(labels):
    msg = _dev1_meta([1.0, 0.01, 1.0], labels=labels)
    msg.data = [1.0, 0.01, 1.0]
    with pytest.raises(ContractError, match="labels"):
        port_meta_from_float64_multi_array(msg)


def test_nonzero_data_offset_rejected():
    with pytest.raises(ContractError, match="data_offset"):
        port_meta_from_float64_multi_array(_dev1_meta([0.0, 1.0, 0.01, 1.0], data_offset=1))


def test_wrong_message_type_rejected():
    with pytest.raises(ContractError, match="Float64MultiArray"):
        port_meta_from_float64_multi_array(msg_module.Bool(data=True))


# --- audit regressions --------------------------------------------------------


def test_exact_dev1_wire_encoding_decodes():
    """Mirror of ugv_perception/node/wire.py field by field (sizes 1, strides 3/2/1)."""
    msg = Float64MultiArray()
    msg.layout = MultiArrayLayout(
        dim=[
            MultiArrayDimension(label="valid", size=1, stride=3),
            MultiArrayDimension(label="age", size=1, stride=2),
            MultiArrayDimension(label="scale", size=1, stride=1),
        ],
        data_offset=0,
    )
    msg.data = [1.0, 0.123, 1.0]
    assert port_meta_from_float64_multi_array(msg) == PortMetaStopgap(valid=True, age_s=0.123, scale=1.0)


def test_negative_zero_valid_decodes_to_false():
    assert port_meta_from_float64_multi_array(_dev1_meta([-0.0, 0.01, 1.0])).valid is False


@pytest.mark.parametrize("scale", [1.0 + 1e-12, 1.0 - 1e-12])
def test_scale_must_be_exactly_one(scale):
    with pytest.raises(ContractError, match="scale"):
        port_meta_from_float64_multi_array(_dev1_meta([1.0, 0.01, scale]))


def test_large_finite_age_carried():
    assert port_meta_from_float64_multi_array(_dev1_meta([1.0, 1e6, 1.0])).age_s == 1e6


@pytest.mark.parametrize(
    "labels",
    [
        ("valid", "valid", "scale"),  # duplicate
        ("Valid", "age", "scale"),  # case
        ("valid ", "age", "scale"),  # whitespace
        ("", "", ""),  # unlabeled
        ("scale", "age", "valid"),  # reversed
    ],
)
def test_more_layout_label_deviations_rejected(labels):
    with pytest.raises(ContractError, match="labels"):
        port_meta_from_float64_multi_array(_dev1_meta([1.0, 0.01, 1.0], labels=labels))


def test_missing_layout_is_rejected_even_with_valid_looking_data():
    msg = Float64MultiArray()
    msg.data = [1.0, 0.01, 1.0]  # right values, but no labelled layout
    with pytest.raises(ContractError, match="labels"):
        port_meta_from_float64_multi_array(msg)


def test_labels_without_data_rejected():
    with pytest.raises(ContractError, match="3 values"):
        port_meta_from_float64_multi_array(_dev1_meta([]))


def test_decoded_value_is_immutable():
    result = port_meta_from_float64_multi_array(_dev1_meta([1.0, 0.01, 1.0]))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.valid = False


def test_decoding_is_stateless_and_does_not_mutate_message():
    """No hidden 'latest meta' state that could be paired with a mask later."""
    a_msg = _dev1_meta([1.0, 0.01, 1.0])
    b_msg = _dev1_meta([0.0, 0.2, 1.0])
    first = port_meta_from_float64_multi_array(a_msg)
    port_meta_from_float64_multi_array(b_msg)
    assert port_meta_from_float64_multi_array(a_msg) == first
    assert list(a_msg.data) == [1.0, 0.01, 1.0]
    assert [d.label for d in a_msg.layout.dim] == ["valid", "age", "scale"]


def test_adapter_exposes_no_pairing_api():
    import costmap_ros.port_meta_adapter as module

    public = {name for name in dir(module) if not name.startswith("_")}
    assert not any(word in name.lower() for name in public for word in ("pair", "match", "latest", "mask"))
