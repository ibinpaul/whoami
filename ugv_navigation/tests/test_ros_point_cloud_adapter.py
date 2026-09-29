"""Tests for costmap_ros.point_cloud_adapter (needs sensor_msgs; skipped without ROS).

All points, stamps and frame names are SYNTHETIC. The layout (x/y/z FLOAT32
at 0/4/8, point_step 12, height 1) mirrors Dev 1's /perception/depth_cloud.
"""

import copy

import numpy as np
import pytest

msg_module = pytest.importorskip("sensor_msgs.msg")
PointCloud2 = msg_module.PointCloud2
PointField = msg_module.PointField

from costmap_core.contracts import ContractError, OccupancyInput, PointCloudInput  # noqa: E402
from costmap_ros.point_cloud_adapter import point_cloud_from_pointcloud2  # noqa: E402

FRAME = "test_camera_optical_frame"  # synthetic; the real frame name is PENDING

POINTS = np.array(
    [[1.0, -2.0, 3.5], [0.25, 0.5, 12.0], [-7.125, 1e-3, 0.75]], dtype=np.float32
)


def _fields(*, x=(0, PointField.FLOAT32, 1), y=(4, PointField.FLOAT32, 1),
            z=(8, PointField.FLOAT32, 1)):
    fields = []
    for name, spec in (("x", x), ("y", y), ("z", z)):
        if spec is None:
            continue
        offset, datatype, count = spec
        fields.append(PointField(name=name, offset=offset, datatype=datatype, count=count))
    return fields


def _cloud_msg(points=POINTS, *, bigendian=False, sec=12, nanosec=345, frame_id=FRAME,
               fields=None, height=1, width=None, point_step=12, row_step=None, data=None):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    msg = PointCloud2()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = frame_id
    msg.height = height
    msg.width = points.shape[0] if width is None else width
    msg.fields = _fields() if fields is None else fields
    msg.is_bigendian = bigendian
    msg.point_step = point_step
    msg.row_step = point_step * msg.width if row_step is None else row_step
    msg.is_dense = False
    if data is None:
        data = points.astype(">f4" if bigendian else "<f4").tobytes()
    msg.data = data
    return msg


# --- valid clouds ------------------------------------------------------------


def test_valid_cloud_converts_to_point_cloud_input():
    result = point_cloud_from_pointcloud2(_cloud_msg())
    assert isinstance(result, PointCloudInput)
    assert result.points.shape == (3, 3)
    assert result.points.dtype == np.float32


def test_xyz_extracted_exactly():
    result = point_cloud_from_pointcloud2(_cloud_msg())
    np.testing.assert_array_equal(result.points, POINTS)


def test_empty_cloud_is_accepted_as_no_observation():
    result = point_cloud_from_pointcloud2(_cloud_msg(np.empty((0, 3)), data=b""))
    assert isinstance(result, PointCloudInput)
    assert result.points.shape == (0, 3)
    # Not free space: the adapter returns points, never an occupancy grid.
    assert not isinstance(result, OccupancyInput)


def test_stamp_converted_to_ns():
    result = point_cloud_from_pointcloud2(
        _cloud_msg(sec=1_700_000_000, nanosec=123_456_789)
    )
    assert result.stamp_ns == 1_700_000_000 * 1_000_000_000 + 123_456_789


def test_frame_id_preserved():
    result = point_cloud_from_pointcloud2(_cloud_msg(frame_id="some_optical_frame"))
    assert result.frame_id == "some_optical_frame"


def test_little_endian_decoded():
    msg = _cloud_msg(bigendian=False)
    assert bytes(msg.data) == POINTS.astype("<f4").tobytes()
    np.testing.assert_array_equal(point_cloud_from_pointcloud2(msg).points, POINTS)


def test_big_endian_decoded_to_same_values():
    msg = _cloud_msg(bigendian=True)
    assert bytes(msg.data) == POINTS.astype(">f4").tobytes()
    result = point_cloud_from_pointcloud2(msg)
    np.testing.assert_array_equal(result.points, POINTS)
    assert result.points.dtype == np.float32
    assert result.points.dtype.isnative


def test_endianness_flag_is_honoured_not_guessed():
    # Little-endian bytes flagged big-endian are decoded big-endian, exactly.
    le_bytes = POINTS.astype("<f4").tobytes()
    expected = np.frombuffer(le_bytes, dtype=">f4").reshape(-1, 3)
    assert np.all(np.isfinite(expected))
    result = point_cloud_from_pointcloud2(_cloud_msg(data=le_bytes, bigendian=True))
    np.testing.assert_array_equal(result.points, expected)
    assert not np.array_equal(result.points, POINTS)


def test_field_order_in_list_does_not_matter():
    fields = list(reversed(_fields()))
    result = point_cloud_from_pointcloud2(_cloud_msg(fields=fields))
    np.testing.assert_array_equal(result.points, POINTS)


def test_message_not_mutated():
    msg = _cloud_msg()
    before = copy.deepcopy(msg)
    result = point_cloud_from_pointcloud2(msg)
    assert msg == before
    assert not result.points.flags.writeable


def test_result_detached_from_message_buffer():
    msg = _cloud_msg()
    result = point_cloud_from_pointcloud2(msg)
    msg.data = b"\x00" * len(msg.data)
    np.testing.assert_array_equal(result.points, POINTS)


# --- rejected layouts --------------------------------------------------------


def test_wrong_message_type_rejected():
    with pytest.raises(ContractError, match="PointCloud2"):
        point_cloud_from_pointcloud2(msg_module.Image())


@pytest.mark.parametrize("missing", ["x", "y", "z"])
def test_missing_field_rejected(missing):
    fields = _fields(**{missing: None})
    with pytest.raises(ContractError, match="fields"):
        point_cloud_from_pointcloud2(_cloud_msg(fields=fields))


def test_extra_field_rejected():
    fields = _fields() + [
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1)
    ]
    with pytest.raises(ContractError, match="fields"):
        point_cloud_from_pointcloud2(_cloud_msg(fields=fields, point_step=16,
                                                data=b"\x00" * 16 * 3))


def test_duplicate_field_rejected():
    fields = _fields()
    fields[1] = PointField(name="x", offset=4, datatype=PointField.FLOAT32, count=1)
    with pytest.raises(ContractError, match="fields"):
        point_cloud_from_pointcloud2(_cloud_msg(fields=fields))


@pytest.mark.parametrize(
    "datatype",
    [PointField.FLOAT64, PointField.INT32, PointField.UINT32, PointField.INT8,
     PointField.UINT8, PointField.INT16, PointField.UINT16, 0, 99],
)
@pytest.mark.parametrize("name", ["x", "y", "z"])
def test_wrong_datatype_rejected(name, datatype):
    offset = {"x": 0, "y": 4, "z": 8}[name]
    fields = _fields(**{name: (offset, datatype, 1)})
    with pytest.raises(ContractError, match="datatype"):
        point_cloud_from_pointcloud2(_cloud_msg(fields=fields))


@pytest.mark.parametrize("count", [0, 2, 3])
def test_wrong_count_rejected(count):
    fields = _fields(x=(0, PointField.FLOAT32, count))
    with pytest.raises(ContractError, match="count"):
        point_cloud_from_pointcloud2(_cloud_msg(fields=fields))


@pytest.mark.parametrize(
    "overrides",
    [
        dict(x=(4, PointField.FLOAT32, 1), y=(0, PointField.FLOAT32, 1)),  # swapped
        dict(z=(12, PointField.FLOAT32, 1)),
        dict(y=(2, PointField.FLOAT32, 1)),
        dict(x=(1, PointField.FLOAT32, 1)),
    ],
)
def test_wrong_offsets_rejected(overrides):
    with pytest.raises(ContractError, match="offset"):
        point_cloud_from_pointcloud2(_cloud_msg(fields=_fields(**overrides)))


@pytest.mark.parametrize("point_step", [0, 8, 11, 13, 16, 32])
def test_invalid_point_step_rejected(point_step):
    msg = _cloud_msg(point_step=point_step, data=b"\x00" * max(point_step, 1) * 3)
    with pytest.raises(ContractError, match="point_step"):
        point_cloud_from_pointcloud2(msg)


@pytest.mark.parametrize("height", [0, 2, 3])
def test_organised_or_zero_height_rejected(height):
    with pytest.raises(ContractError, match="height"):
        point_cloud_from_pointcloud2(_cloud_msg(height=height))


@pytest.mark.parametrize("row_step", [0, 12, 24, 40, 48])
def test_row_step_mismatch_rejected(row_step):
    with pytest.raises(ContractError, match="row_step"):
        point_cloud_from_pointcloud2(_cloud_msg(row_step=row_step))


@pytest.mark.parametrize("width", [0, 2, 4, 100])
def test_width_disagreeing_with_data_rejected(width):
    # row_step follows width, so the data length is what disagrees.
    with pytest.raises(ContractError, match="bytes"):
        point_cloud_from_pointcloud2(_cloud_msg(width=width))


@pytest.mark.parametrize("delta", [-12, -1, 1, 12])
def test_data_length_mismatch_rejected(delta):
    good = POINTS.astype("<f4").tobytes()
    data = good[:delta] if delta < 0 else good + b"\x00" * delta
    with pytest.raises(ContractError, match="bytes"):
        point_cloud_from_pointcloud2(_cloud_msg(data=data))


def test_empty_cloud_with_leftover_bytes_rejected():
    with pytest.raises(ContractError, match="bytes"):
        point_cloud_from_pointcloud2(_cloud_msg(np.empty((0, 3)), data=b"\x00" * 12))


# --- non-finite points and header values ------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("bigendian", [False, True])
def test_non_finite_points_rejected_not_dropped(bad, bigendian):
    points = POINTS.copy()
    points[1, 0] = bad
    with pytest.raises(ContractError, match="finite"):
        point_cloud_from_pointcloud2(_cloud_msg(points, bigendian=bigendian))


def test_empty_frame_id_rejected():
    with pytest.raises(ContractError, match="frame_id"):
        point_cloud_from_pointcloud2(_cloud_msg(frame_id=""))


def test_zero_stamp_rejected():
    with pytest.raises(ContractError, match="stamp"):
        point_cloud_from_pointcloud2(_cloud_msg(sec=0, nanosec=0))


def test_camera_sized_cloud_converts():
    # Synthetic size of the order of one full camera frame of valid points.
    rng = np.random.default_rng(0)
    points = rng.uniform(-50.0, 50.0, size=(640 * 480, 3)).astype(np.float32)
    result = point_cloud_from_pointcloud2(_cloud_msg(points))
    np.testing.assert_array_equal(result.points, points)
