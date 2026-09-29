"""Tests for costmap_ros.costmap_node (needs rclpy/tf2_ros; skipped without ROS).

All messages, TF, frames and parameter values are synthetic; the real ones
are PENDING (Dev 2 / Dev 5 / team decisions).
"""

import math
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
tf2_ros = pytest.importorskip("tf2_ros")

from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import OccupancyGrid  # noqa: E402
from rclpy.exceptions import ParameterUninitializedException  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSCompatibility,
    QoSProfile,
    ReliabilityPolicy,
    qos_check_compatible,
)
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402

from costmap_core.contracts import ContractError  # noqa: E402
from costmap_ros import costmap_node  # noqa: E402
from costmap_ros.costmap_node import (  # noqa: E402
    PARAMETERS,
    CostmapNode,
    CostmapNodeCore,
    config_from_parameters,
)
from costmap_ros.tf_lookup import TfCameraPoseLookup  # noqa: E402

GROUND_FRAME = "test_ground_frame"
CAMERA_FRAME = "test_camera_frame"
MASK_SEC, MASK_NSEC = 7, 500_000_000
MASK_STAMP_NS = MASK_SEC * 1_000_000_000 + MASK_NSEC

# Optical frame (x right, y down, z forward) -> level ground frame
# (x forward, y left, z up): quaternion (x, y, z, w) of R = [[0,0,1],[-1,0,0],[0,-1,0]].
OPTICAL_TO_GROUND_Q = (-0.5, 0.5, -0.5, 0.5)
CAMERA_HEIGHT = 1.0

# 8x8 mask, fx = fy = 4, principal point at the centre: the bottom half of the
# image sees the ground between ~1.1 m and 8 m ahead.
IMG_W = IMG_H = 8


def _params(**overrides):
    values = {
        "mask_topic": "/test/mask",
        "camera_info_topic": "/test/camera_info",
        "costmap_topic": "/test/costmap",
        "target_frame": GROUND_FRAME,
        "grid.resolution": 0.5,
        "grid.origin_x": 0.0,
        "grid.origin_y": -2.0,
        "grid.width": 20,
        "grid.height": 8,
        "tf_timeout_s": 0.05,
        "inflation_radius": 0.0,
        "camera_info_durability": "volatile",
        "mask_max_age_s": 0.5,
        "output_stamp_source": "mask",
        "geometry.enabled": False,
        "fail_safe.roi_min_x": 0.0,
        "fail_safe.roi_max_x": 2.0,
        "fail_safe.roi_min_y": -1.0,
        "fail_safe.roi_max_y": 1.0,
        "fail_safe.check_period_s": 0.1,
        "footprint.enabled": False,
    }
    values.update(overrides)
    return values


def _mask_msg(*, classes=None, encoding="mono8", frame_id=CAMERA_FRAME, sec=MASK_SEC, nanosec=MASK_NSEC):
    if classes is None:
        classes = np.ones((IMG_H, IMG_W), dtype=np.uint8)
        classes[6:, :] = 2
    classes = np.asarray(classes, dtype=np.uint8)
    msg = Image()
    msg.header.stamp.sec, msg.header.stamp.nanosec = sec, nanosec
    msg.header.frame_id = frame_id
    msg.height, msg.width = classes.shape
    msg.encoding = encoding
    msg.step = classes.shape[1]
    msg.data = classes.tobytes()
    return msg


def _info_msg(*, frame_id=CAMERA_FRAME, sec=MASK_SEC, nanosec=MASK_NSEC, fx=4.0, width=IMG_W, height=IMG_H):
    msg = CameraInfo()
    msg.header.stamp.sec, msg.header.stamp.nanosec = sec, nanosec
    msg.header.frame_id = frame_id
    msg.width, msg.height = width, height
    msg.k = [fx, 0.0, 3.5, 0.0, 4.0, 3.5, 0.0, 0.0, 1.0]
    return msg


def _tf_msg(stamp_ns=MASK_STAMP_NS, *, frame_id=GROUND_FRAME, child_frame_id=CAMERA_FRAME):
    msg = TransformStamped()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    msg.transform.translation.z = CAMERA_HEIGHT
    q = msg.transform.rotation
    q.x, q.y, q.z, q.w = OPTICAL_TO_GROUND_Q
    return msg


class FakeBuffer:
    """tf2 Buffer stand-in: echoes the requested stamp, or raises `error`."""

    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def lookup_transform(self, target_frame, source_frame, time, timeout):
        self.calls.append((target_frame, source_frame, time, timeout))
        if self.error is not None:
            raise self.error
        return _tf_msg(time.nanoseconds, frame_id=target_frame, child_frame_id=source_frame)


class RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message, **_):
        self.warnings.append(message)


@pytest.fixture
def pipeline_spy(monkeypatch):
    calls = []
    real = costmap_node.run_costmap_pipeline

    def spy(inputs, **kwargs):
        calls.append((inputs, kwargs))
        return real(inputs, **kwargs)

    monkeypatch.setattr(costmap_node, "run_costmap_pipeline", spy)
    return calls


def _core(*, buffer=None, **param_overrides):
    config = config_from_parameters(_params(**param_overrides))
    buffer = buffer if buffer is not None else FakeBuffer()
    logger = RecordingLogger()
    core = CostmapNodeCore(config, TfCameraPoseLookup(buffer, timeout_s=config.tf_timeout_s), logger)
    return core, buffer, logger


NOW = Time(sec=MASK_SEC, nanosec=MASK_NSEC + 100_000_000)  # mask age 0.1 s


# --- configuration --------------------------------------------------------------


def test_parameters_build_config_and_grid():
    config = config_from_parameters(_params())
    assert (config.mask_topic, config.camera_info_topic, config.costmap_topic) == (
        "/test/mask", "/test/camera_info", "/test/costmap",
    )
    assert config.target_frame == GROUND_FRAME
    g = config.grid.geometry
    assert (g.resolution, g.origin_x, g.origin_y, g.width, g.height) == (0.5, 0.0, -2.0, 20, 8)
    assert config.grid.frame_id == GROUND_FRAME
    assert config.tf_timeout_s == 0.05
    assert config.inflation_radius == 0.0
    assert config.camera_info_durability == "volatile"
    assert config.mask_max_age_s == 0.5
    assert config.output_stamp_source == "mask"


@pytest.mark.parametrize("name", [name for name, _ in PARAMETERS])
def test_every_parameter_is_required(name):
    values = _params()
    del values[name]
    with pytest.raises(ContractError, match="missing"):
        config_from_parameters(values)


def test_unknown_parameter_rejected():
    with pytest.raises(ContractError, match="unknown"):
        config_from_parameters(_params(extra=1))


@pytest.mark.parametrize(
    "override",
    [
        {"output_stamp_source": "latest"},
        {"tf_timeout_s": -0.1},
        {"inflation_radius": math.nan},
        {"camera_info_durability": "latched"},
        {"camera_info_durability": ""},
        {"mask_max_age_s": 0.0},
        {"mask_max_age_s": -0.1},
        {"mask_max_age_s": math.nan},
        {"mask_max_age_s": math.inf},
        {"mask_topic": ""},
        {"target_frame": ""},
    ],
)
def test_invalid_parameter_values_rejected(override):
    with pytest.raises(ContractError):
        config_from_parameters(_params(**override))


def test_invalid_grid_parameter_rejected():
    from costmap_core.grid import GridError

    with pytest.raises(GridError):
        config_from_parameters(_params(**{"grid.resolution": 0.0}))


# --- per-message logic ----------------------------------------------------------


def test_valid_mask_invokes_pipeline_and_returns_grid(pipeline_spy):
    core, _, logger = _core()
    core.handle_camera_info(_info_msg())
    grid = core.handle_mask(_mask_msg(), now_stamp=NOW)
    assert len(pipeline_spy) == 1
    assert isinstance(grid, OccupancyGrid)
    assert logger.warnings == []
    data = np.array(grid.data)
    assert (data == 0).any() and (data == 100).any() and (data == -1).any()



def test_node_supplies_no_footprint_or_padding(pipeline_spy):
    """footprint.enabled is False (Dev 5 footprint PENDING): none is invented."""
    core, _, _ = _core()
    core.handle_camera_info(_info_msg())
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is not None
    inputs, kwargs = pipeline_spy[0]
    assert inputs.footprint is None
    assert inputs.occupancy is None  # geometry.enabled is False: semantics only
    assert "footprint_padding" not in kwargs
    # Only the explicit switch exists; no footprint value is defaulted.
    assert [name for name, _ in PARAMETERS if "footprint" in name] == ["footprint.enabled"]



def test_mask_processed_without_any_port_meta(pipeline_spy):
    """Receipt of a valid mask is the validity signal (temporary, see module docstring)."""
    core, _, _ = _core()
    core.handle_camera_info(_info_msg())
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is not None
    assert not hasattr(core, "handle_port_meta")

def test_mask_is_marked_valid_on_receipt(pipeline_spy):
    core, _, _ = _core()
    core.handle_camera_info(_info_msg())
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    inputs, _ = pipeline_spy[0]
    assert inputs.mask.valid is True


def test_tf_lookup_uses_mask_stamp_frames_and_timeout(pipeline_spy):
    core, buffer, _ = _core(**{"tf_timeout_s": 0.25})
    core.handle_camera_info(_info_msg())
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    target, source, time, timeout = buffer.calls[0]
    assert target == GROUND_FRAME
    assert source == CAMERA_FRAME  # taken from the mask header, not configured
    assert time.nanoseconds == MASK_STAMP_NS
    assert timeout.nanoseconds == 250_000_000
    inputs, _ = pipeline_spy[0]
    assert inputs.camera_ground.stamp_ns == MASK_STAMP_NS
    np.testing.assert_allclose(inputs.camera_ground.geometry.translation, [0.0, 0.0, CAMERA_HEIGHT])


def test_camera_info_converted_into_pipeline_inputs(pipeline_spy):
    core, _, _ = _core()
    core.handle_camera_info(_info_msg(fx=4.5))
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    intrinsics = pipeline_spy[0][0].intrinsics
    assert (intrinsics.intrinsics.fx, intrinsics.intrinsics.fy) == (4.5, 4.0)
    assert (intrinsics.intrinsics.cx, intrinsics.intrinsics.cy) == (3.5, 3.5)
    assert (intrinsics.image_width, intrinsics.image_height) == (IMG_W, IMG_H)
    assert intrinsics.frame_id == CAMERA_FRAME


def test_grid_input_and_inflation_radius_from_parameters(pipeline_spy):
    core, _, _ = _core(**{"grid.resolution": 0.25, "grid.width": 40, "grid.height": 16,
                          "inflation_radius": 0.5})
    core.handle_camera_info(_info_msg())
    grid_msg = core.handle_mask(_mask_msg(), now_stamp=NOW)
    inputs, kwargs = pipeline_spy[0]
    g = inputs.grid.geometry
    assert (g.resolution, g.origin_x, g.origin_y, g.width, g.height) == (0.25, 0.0, -2.0, 40, 16)
    assert inputs.grid.frame_id == GROUND_FRAME
    assert kwargs == {"inflation_radius": 0.5}
    assert (grid_msg.info.width, grid_msg.info.height) == (40, 16)
    assert grid_msg.info.resolution == pytest.approx(0.25)
    assert grid_msg.header.frame_id == GROUND_FRAME


def test_output_stamp_mask(pipeline_spy):
    core, _, _ = _core(output_stamp_source="mask")
    core.handle_camera_info(_info_msg())
    grid = core.handle_mask(_mask_msg(), now_stamp=NOW)
    assert (grid.header.stamp.sec, grid.header.stamp.nanosec) == (MASK_SEC, MASK_NSEC)


def test_output_stamp_now(pipeline_spy):
    core, _, _ = _core(output_stamp_source="now")
    core.handle_camera_info(_info_msg())
    grid = core.handle_mask(_mask_msg(), now_stamp=NOW)
    assert grid.header.stamp == NOW


def test_mask_skipped_without_camera_info(pipeline_spy):
    core, buffer, logger = _core()
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is None
    assert pipeline_spy == [] and buffer.calls == []
    assert "no valid CameraInfo" in logger.warnings[0]


@pytest.mark.parametrize(
    "error",
    [
        tf2_ros.LookupException("synthetic: frame does not exist"),
        tf2_ros.ExtrapolationException("synthetic: extrapolation into the future"),
        tf2_ros.ConnectivityException("synthetic: not connected"),
        tf2_ros.TimeoutException("synthetic: timed out"),
    ],
)
def test_tf_failure_is_logged_and_skipped(pipeline_spy, error):
    core, _, logger = _core(buffer=FakeBuffer(error=error))
    core.handle_camera_info(_info_msg())
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is None
    assert pipeline_spy == []
    assert type(error).__name__ in logger.warnings[0]
    # The node keeps working once TF is available again.
    core._tf_lookup._buffer.error = None
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is not None


@pytest.mark.parametrize(
    "msg",
    [
        _mask_msg(encoding="rgb8"),
        _mask_msg(classes=np.full((IMG_H, IMG_W), 3, dtype=np.uint8)),  # non-canonical id
        _mask_msg(sec=0, nanosec=0),
        _mask_msg(frame_id=""),
    ],
    ids=["encoding", "class_id", "zero_stamp", "empty_frame"],
)
def test_invalid_mask_never_reaches_pipeline(pipeline_spy, msg):
    core, buffer, logger = _core()
    core.handle_camera_info(_info_msg())
    assert core.handle_mask(msg, now_stamp=NOW) is None
    assert pipeline_spy == [] and buffer.calls == []
    assert "ContractError" in logger.warnings[0]


def test_camera_info_frame_mismatch_skipped(pipeline_spy):
    core, _, logger = _core()
    core.handle_camera_info(_info_msg(frame_id="other_camera"))
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is None
    assert pipeline_spy == []
    assert "camera frame ids disagree" in logger.warnings[0]


def test_camera_info_size_mismatch_skipped(pipeline_spy):
    core, _, logger = _core()
    core.handle_camera_info(_info_msg(width=16, height=16))
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is None
    assert pipeline_spy == []
    assert "does not match intrinsics image size" in logger.warnings[0]


def test_camera_info_much_older_than_mask_is_used(pipeline_spy):
    """A latched calibration keeps its own old stamp; it is still valid."""
    core, _, logger = _core()
    core.handle_camera_info(_info_msg(sec=1, nanosec=0))  # ~MASK_SEC seconds older than the mask
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is not None
    assert len(pipeline_spy) == 1 and logger.warnings == []


@pytest.mark.parametrize(
    "sec, nanosec", [(0, 0), (MASK_SEC + 3600, 0), (MASK_SEC, MASK_NSEC + 1)],
    ids=["zero_stamp", "one_hour_newer", "1ns_newer"],
)
def test_camera_info_stamp_is_not_compared_with_mask_stamp(pipeline_spy, sec, nanosec):
    core, _, _ = _core()
    core.handle_camera_info(_info_msg(sec=sec, nanosec=nanosec))
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is not None
    assert len(pipeline_spy) == 1


def test_camera_info_max_stamp_offset_parameter_removed():
    with pytest.raises(ContractError, match="unknown.*camera_info_max_stamp_offset_s"):
        config_from_parameters(_params(camera_info_max_stamp_offset_s=0.1))
    assert "camera_info_max_stamp_offset_s" not in {name for name, _ in PARAMETERS}


def test_matching_frame_camera_info_intrinsics_reach_pipeline(pipeline_spy):
    core, _, _ = _core()
    core.handle_camera_info(_info_msg(fx=5.0))
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is not None
    inputs, _ = pipeline_spy[0]
    assert inputs.intrinsics.frame_id == inputs.mask.frame_id == CAMERA_FRAME
    assert inputs.intrinsics.intrinsics.fx == 5.0


def test_latest_camera_info_replaces_previous(pipeline_spy):
    core, _, _ = _core()
    core.handle_camera_info(_info_msg(fx=4.0, sec=1))
    core.handle_camera_info(_info_msg(fx=6.0, sec=2))
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    assert pipeline_spy[0][0].intrinsics.intrinsics.fx == 6.0


def test_mismatched_frame_after_valid_camera_info_is_skipped(pipeline_spy):
    core, _, logger = _core()
    core.handle_camera_info(_info_msg())
    assert core.handle_mask(_mask_msg(frame_id="other_camera"), now_stamp=NOW) is None
    assert pipeline_spy == []
    assert "camera frame ids disagree" in logger.warnings[0]


@pytest.mark.parametrize("width, height", [(IMG_W + 1, IMG_H), (IMG_W, IMG_H - 1), (IMG_H, IMG_W * 2)])
def test_camera_info_dimension_mismatch_skipped(pipeline_spy, width, height):
    core, _, logger = _core()
    core.handle_camera_info(_info_msg(width=width, height=height))
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is None
    assert pipeline_spy == []
    assert "does not match intrinsics image size" in logger.warnings[0]


@pytest.mark.parametrize(
    "k",
    [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # identity placeholder
        [0.0] * 9,  # uncalibrated
        [-4.0, 0.0, 3.5, 0.0, 4.0, 3.5, 0.0, 0.0, 1.0],  # fx < 0
        [4.0, 0.0, 3.5, 0.0, math.nan, 3.5, 0.0, 0.0, 1.0],  # non-finite
        [4.0, 0.1, 3.5, 0.0, 4.0, 3.5, 0.0, 0.0, 1.0],  # skew
    ],
    ids=["identity", "zeros", "negative_fx", "nan", "skew"],
)
def test_invalid_intrinsics_camera_info_rejected_and_masks_skipped(pipeline_spy, k):
    core, _, logger = _core()
    bad = _info_msg()
    bad.k = k
    core.handle_camera_info(bad)
    assert not core.has_camera_info
    assert "CameraInfo rejected" in logger.warnings[0]
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is None
    assert pipeline_spy == []


def test_invalid_camera_info_clears_stored_intrinsics(pipeline_spy):
    core, _, logger = _core()
    core.handle_camera_info(_info_msg())
    bad = _info_msg()
    bad.k = [0.0] * 9
    core.handle_camera_info(bad)
    assert not core.has_camera_info
    assert "CameraInfo rejected" in logger.warnings[0]
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is None
    assert pipeline_spy == []


def test_invalid_pose_from_tf_skipped(pipeline_spy):
    class BelowGroundBuffer(FakeBuffer):
        def lookup_transform(self, *args):
            msg = super().lookup_transform(*args)
            msg.transform.translation.z = -1.0
            return msg

    core, _, logger = _core(buffer=BelowGroundBuffer())
    core.handle_camera_info(_info_msg())
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is None
    assert pipeline_spy == []
    assert "ProjectionError" in logger.warnings[0]


# --- rclpy node wiring ------------------------------------------------------------


def _overrides(values):
    return [Parameter(name, value=value) for name, value in values.items()]


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_node_uses_configured_topics_and_frame(ros_context):
    node = CostmapNode(parameter_overrides=_overrides(_params()))
    try:
        subs = {s.topic_name for s in node.subscriptions}
        pubs = {p.topic_name for p in node.publishers}
        assert {"/test/mask", "/test/camera_info"} <= subs
        assert "/test/costmap" in pubs
        assert node.config.target_frame == GROUND_FRAME
        assert node.config.grid.geometry.width == 20
        assert node.config.tf_timeout_s == 0.05
        assert "/segmentation/port_meta" not in subs
    finally:
        node.destroy_node()



def test_node_does_not_consume_port_meta(ros_context):
    """PortMeta (Float64MultiArray stopgap) has no header, so it cannot be paired
    with a mask; the node neither subscribes to it nor has a parameter for it."""
    from std_msgs.msg import Float64MultiArray
    from tf2_msgs.msg import TFMessage

    node = CostmapNode(parameter_overrides=_overrides(_params()))
    try:
        assert all(s.msg_type is not Float64MultiArray for s in node.subscriptions)
        # Mask, CameraInfo, and the TF listener's /tf + /tf_static.
        assert {s.msg_type for s in node.subscriptions} == {Image, CameraInfo, TFMessage}
        assert not any("port_meta" in name or "meta" in name for name, _ in PARAMETERS)
    finally:
        node.destroy_node()

@pytest.mark.parametrize("name", ["mask_topic", "target_frame", "grid.resolution", "tf_timeout_s",
                                  "inflation_radius", "output_stamp_source"])
def test_node_refuses_to_start_without_required_parameter(ros_context, name):
    values = _params()
    del values[name]
    with pytest.raises(ParameterUninitializedException):
        CostmapNode(parameter_overrides=_overrides(values))


def test_node_callback_publishes_on_configured_topic(ros_context, pipeline_spy):
    node = CostmapNode(parameter_overrides=_overrides(_params()))
    try:
        node._core._tf_lookup = TfCameraPoseLookup(FakeBuffer(), timeout_s=0.05)
        published = []
        node._pub_costmap.publish = published.append
        sec, nanosec = node.get_clock().now().seconds_nanoseconds()
        node._core.handle_camera_info(_info_msg(sec=sec, nanosec=nanosec))
        node._on_mask(_mask_msg(sec=sec, nanosec=nanosec))  # fresh by the node clock
        assert len(published) == 1
        assert published[0].header.frame_id == GROUND_FRAME
    finally:
        node.destroy_node()


# --- mask freshness (architecture §8.4) -----------------------------------------------


def _now_after_mask(age_ns):
    sec, nanosec = divmod(MASK_STAMP_NS + age_ns, 1_000_000_000)
    return Time(sec=sec, nanosec=nanosec)


def test_fresh_mask_accepted(pipeline_spy):
    core, buffer, logger = _core(mask_max_age_s=0.5)
    core.handle_camera_info(_info_msg())
    assert core.handle_mask(_mask_msg(), now_stamp=_now_after_mask(200_000_000)) is not None
    assert len(pipeline_spy) == 1 and len(buffer.calls) == 1
    assert logger.warnings == []


def test_stale_mask_rejected_before_tf_lookup(pipeline_spy):
    core, buffer, logger = _core(mask_max_age_s=0.5)
    core.handle_camera_info(_info_msg())
    assert core.handle_mask(_mask_msg(), now_stamp=_now_after_mask(800_000_000)) is None
    assert buffer.calls == []  # no TF lookup for a stale mask
    assert pipeline_spy == []
    assert "stale" in logger.warnings[0]
    assert "age = 0.800000 s" in logger.warnings[0]
    assert "mask_max_age_s = 0.5" in logger.warnings[0]


def test_future_stamped_mask_rejected(pipeline_spy):
    core, buffer, logger = _core(mask_max_age_s=0.5)
    core.handle_camera_info(_info_msg())
    assert core.handle_mask(_mask_msg(), now_stamp=_now_after_mask(-1)) is None  # 1 ns ahead
    assert buffer.calls == [] and pipeline_spy == []
    assert "future-stamped" in logger.warnings[0]


def test_zero_age_is_fresh(pipeline_spy):
    core, _, logger = _core(mask_max_age_s=0.5)
    core.handle_camera_info(_info_msg())
    assert core.handle_mask(_mask_msg(), now_stamp=_now_after_mask(0)) is not None
    assert logger.warnings == []


@pytest.mark.parametrize("age_ns, fresh", [(250_000_000, True), (250_000_001, False)])
def test_exact_age_boundary(pipeline_spy, age_ns, fresh):
    # is_fresh: 0 <= age <= mask_max_age_s, so age == limit is still fresh.
    core, buffer, _ = _core(mask_max_age_s=0.25)
    core.handle_camera_info(_info_msg())
    result = core.handle_mask(_mask_msg(), now_stamp=_now_after_mask(age_ns))
    assert (result is not None) is fresh
    assert len(buffer.calls) == (1 if fresh else 0)


def test_freshness_uses_the_given_clock_sample_not_wall_clock(pipeline_spy):
    # The mask is stamped 7.5 s after the epoch, i.e. decades old by wall
    # clock; only the supplied node-clock sample decides freshness.
    core, _, _ = _core(mask_max_age_s=0.5)
    core.handle_camera_info(_info_msg())
    assert core.handle_mask(_mask_msg(), now_stamp=_now_after_mask(100_000_000)) is not None


def test_zero_node_clock_is_logged_and_skipped(pipeline_spy):
    # e.g. use_sim_time before the first /clock message: now = 0 is not a
    # valid stamp for is_fresh, so the mask is skipped, never used.
    core, buffer, logger = _core(mask_max_age_s=0.5)
    core.handle_camera_info(_info_msg())
    assert core.handle_mask(_mask_msg(), now_stamp=Time(sec=0, nanosec=0)) is None
    assert buffer.calls == [] and pipeline_spy == []
    assert "now_ns" in logger.warnings[0]


def test_mask_subscription_queue_depth_is_one(ros_context):
    node = CostmapNode(parameter_overrides=_overrides(_params()))
    try:
        (mask_sub,) = [s for s in node.subscriptions if s.topic_name == "/test/mask"]
        assert mask_sub.qos_profile.depth == 1
        assert node.config.mask_max_age_s == 0.5
    finally:
        node.destroy_node()


def test_node_refuses_to_start_without_mask_max_age(ros_context):
    values = _params()
    del values["mask_max_age_s"]
    with pytest.raises(ParameterUninitializedException):
        CostmapNode(parameter_overrides=_overrides(values))


# --- CameraInfo topic and QoS ------------------------------------------------------

# Dev 1's /segmentation/camera_info republisher: create_publisher(..., 10), i.e.
# RELIABLE + VOLATILE (ugv_perception node/adapter_node.py; copied, not imported).
DEV1_REPUBLISHER_QOS = QoSProfile(depth=10)
# QoS Dev 1 expects from the driver's CameraInfo publisher (its camera_info_qos()).
DRIVER_EXPECTED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


@pytest.mark.parametrize(
    "name, policy",
    [("volatile", DurabilityPolicy.VOLATILE), ("transient_local", DurabilityPolicy.TRANSIENT_LOCAL)],
)
def test_camera_info_qos_profile(name, policy):
    qos = costmap_node.camera_info_qos(name)
    assert qos.durability == policy
    assert qos.reliability == ReliabilityPolicy.RELIABLE
    assert qos.history == HistoryPolicy.KEEP_LAST and qos.depth == 1


@pytest.mark.parametrize("bad", ["", "VOLATILE", "latched", "best_effort", None])
def test_camera_info_qos_rejects_unknown_durability(bad):
    with pytest.raises(ContractError, match="camera_info_durability"):
        costmap_node.camera_info_qos(bad)


def _compatible(pub, sub):
    return qos_check_compatible(pub, sub)[0] != QoSCompatibility.ERROR


def test_volatile_matches_both_candidate_publishers():
    sub = costmap_node.camera_info_qos("volatile")
    assert _compatible(DEV1_REPUBLISHER_QOS, sub)
    assert _compatible(DRIVER_EXPECTED_QOS, sub)


def test_transient_local_cannot_match_dev1_republisher():
    sub = costmap_node.camera_info_qos("transient_local")
    assert not _compatible(DEV1_REPUBLISHER_QOS, sub)
    assert _compatible(DRIVER_EXPECTED_QOS, sub)


@pytest.mark.parametrize("durability", ["volatile", "transient_local"])
def test_node_camera_info_subscription_uses_configured_topic_and_qos(ros_context, durability):
    topic = "/some_ns/cam_info_for_test"
    node = CostmapNode(parameter_overrides=_overrides(
        _params(camera_info_topic=topic, camera_info_durability=durability)))
    try:
        (sub,) = [s for s in node.subscriptions if s.topic_name == topic]
        expected = costmap_node.camera_info_qos(durability)
        assert (sub.qos_profile.durability, sub.qos_profile.reliability, sub.qos_profile.depth) == (
            expected.durability, expected.reliability, 1,
        )
    finally:
        node.destroy_node()


def test_node_refuses_to_start_without_camera_info_durability(ros_context):
    values = _params()
    del values["camera_info_durability"]
    with pytest.raises(ParameterUninitializedException):
        CostmapNode(parameter_overrides=_overrides(values))


def _spin_until(node, predicate, timeout_s):
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        deadline = time.monotonic() + timeout_s
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
    finally:
        executor.remove_node(node)
        executor.shutdown()
    return predicate()


def _latched_camera_info_received(durability, *, timeout_s):
    """A TRANSIENT_LOCAL publisher sends one CameraInfo BEFORE the node exists."""
    topic = "/test/latched_camera_info"
    pub_node = rclpy.create_node("latched_camera_info_publisher")
    node = None
    try:
        pub = pub_node.create_publisher(CameraInfo, topic, DRIVER_EXPECTED_QOS)
        pub.publish(_info_msg(sec=1))  # old calibration stamp
        node = CostmapNode(parameter_overrides=_overrides(
            _params(camera_info_topic=topic, camera_info_durability=durability)))
        return _spin_until(node, lambda: node._core.has_camera_info, timeout_s)
    finally:
        if node is not None:
            node.destroy_node()
        pub_node.destroy_node()


def test_transient_local_subscription_receives_latched_calibration(ros_context):
    assert _latched_camera_info_received("transient_local", timeout_s=5.0)


def test_volatile_subscription_misses_latched_calibration(ros_context):
    assert not _latched_camera_info_received("volatile", timeout_s=1.0)


FORBIDDEN_TOPIC_LITERALS = ("/camera/camera_info", "/segmentation/camera_info", "camera_info")


def test_no_hard_coded_camera_info_topic_in_node_code():
    import ast
    from pathlib import Path

    root = Path(costmap_node.__file__).resolve().parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
        }
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings:
                if n.value in FORBIDDEN_TOPIC_LITERALS or n.value.startswith(FORBIDDEN_TOPIC_LITERALS[:2]):
                    offenders.append(f"{path.name}:{n.lineno}: {n.value!r}")
    assert offenders == []


# --- geometry side-channel (/perception/depth_cloud) ------------------------------
#
# Synthetic geometry: level camera 1.0 m above the ground-frame origin (the
# _tf_msg pose). Synthetic thresholds keep points 0.2 .. 2.0 m high and
# 0.5 .. 20 m from the camera. None of these are real values.

from sensor_msgs.msg import PointCloud2, PointField  # noqa: E402

from costmap_ros.costmap_node import (  # noqa: E402
    DEPTH_CLOUD_QOS,
    GEOMETRY_PARAMETERS,
)

GRID_RES, GRID_OX, GRID_OY, GRID_W = 0.5, 0.0, -2.0, 20
GEOMETRY_MAX_AGE_S = 0.1  # synthetic; the real geometry.max_age_s is PENDING
GEOMETRY_MAX_AGE_NS = 100_000_000
OPTICAL_TO_GROUND_R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _geo_params(**overrides):
    values = _params(**{
        "geometry.enabled": True,
        "geometry.depth_cloud_topic": "/test/depth_cloud",
        "geometry.min_height": 0.2,
        "geometry.max_height": 2.0,
        "geometry.min_range": 0.5,
        "geometry.max_range": 20.0,
        "geometry.max_age_s": GEOMETRY_MAX_AGE_S,
    })
    values.update(overrides)
    return values


def _geo_core(*, buffer=None, **overrides):
    config = config_from_parameters(_geo_params(**overrides))
    buffer = buffer if buffer is not None else FakeBuffer()
    logger = RecordingLogger()
    core = CostmapNodeCore(config, TfCameraPoseLookup(buffer, timeout_s=config.tf_timeout_s), logger)
    core.handle_camera_info(_info_msg())
    return core, buffer, logger


def _ground_to_optical(ground_xyz):
    g = np.asarray(ground_xyz, dtype=np.float64).reshape(-1, 3)
    return (g - [0.0, 0.0, CAMERA_HEIGHT]) @ OPTICAL_TO_GROUND_R  # R^T (g - t), row-wise


def _cloud_msg(ground_points=(), *, stamp_ns=MASK_STAMP_NS, frame_id=CAMERA_FRAME, data=None):
    optical = _ground_to_optical(ground_points).astype("<f4")
    msg = PointCloud2()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
    msg.header.frame_id = frame_id
    msg.height, msg.width = 1, optical.shape[0]
    msg.fields = [
        PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
        for n, o in (("x", 0), ("y", 4), ("z", 8))
    ]
    msg.is_bigendian = False
    msg.point_step, msg.row_step = 12, 12 * optical.shape[0]
    msg.is_dense = False
    msg.data = optical.tobytes() if data is None else data
    return msg


def _cell_center(row, col):
    return (GRID_OX + (col + 0.5) * GRID_RES, GRID_OY + (row + 0.5) * GRID_RES)


def _data(grid_msg):
    return np.array(grid_msg.data, dtype=np.int64).reshape(grid_msg.info.height, grid_msg.info.width)


def _semantic_only():
    core, _, _ = _core()
    core.handle_camera_info(_info_msg())
    return _data(core.handle_mask(_mask_msg(), now_stamp=NOW))


def _free_cell():
    """A cell the semantic mask marks traversable (occupancy 0)."""
    rows, cols = np.nonzero(_semantic_only() == 0)
    return int(rows[0]), int(cols[0])


def _obstacle_point_in(row, col, height=1.0):
    x, y = _cell_center(row, col)
    return [x, y, height]


# configuration


def test_geometry_config_built_from_parameters():
    config = config_from_parameters(_geo_params())
    geo = config.geometry
    assert geo.depth_cloud_topic == "/test/depth_cloud"
    p = geo.filter_params
    assert (p.min_height, p.max_height, p.min_range, p.max_range) == (0.2, 2.0, 0.5, 20.0)
    assert geo.max_age_s == GEOMETRY_MAX_AGE_S


def test_geometry_disabled_has_no_geometry_config():
    assert config_from_parameters(_params()).geometry is None


@pytest.mark.parametrize("name", [name for name, _ in GEOMETRY_PARAMETERS])
def test_every_geometry_parameter_required_when_enabled(name):
    values = _geo_params()
    del values[name]
    with pytest.raises(ContractError, match="missing"):
        config_from_parameters(values)


@pytest.mark.parametrize("name", [name for name, _ in GEOMETRY_PARAMETERS])
def test_geometry_parameter_rejected_when_disabled(name):
    values = _params(**{name: _geo_params()[name]})
    with pytest.raises(ContractError, match="unknown"):
        config_from_parameters(values)


@pytest.mark.parametrize(
    "override",
    [
        {"geometry.enabled": "true"},
        {"geometry.enabled": 1},
        {"geometry.depth_cloud_topic": ""},
        {"geometry.min_height": 2.0},                 # == max_height
        {"geometry.max_height": 0.1},                 # < min_height
        {"geometry.min_range": -0.1},
        {"geometry.max_range": 0.5},                  # == min_range
        {"geometry.min_height": math.nan},
        {"geometry.max_range": math.inf},
        {"geometry.max_age_s": 0.0},
        {"geometry.max_age_s": -0.001},
        {"geometry.max_age_s": math.nan},
        {"geometry.max_age_s": math.inf},
        {"geometry.max_age_s": "0.1"},
        {"geometry.max_age_s": True},
    ],
)
def test_invalid_geometry_configuration_rejected(override):
    with pytest.raises(ContractError):
        config_from_parameters(_geo_params(**override))


def test_depth_cloud_qos_matches_dev1_publisher():
    # Dev 1 adapter_node.py: create_publisher(PointCloud2, ..., 10) (copied, not imported).
    dev1_publisher = QoSProfile(depth=10)
    assert (DEPTH_CLOUD_QOS.history, DEPTH_CLOUD_QOS.depth) == (HistoryPolicy.KEEP_LAST, 1)
    assert DEPTH_CLOUD_QOS.reliability == ReliabilityPolicy.RELIABLE
    assert DEPTH_CLOUD_QOS.durability == DurabilityPolicy.VOLATILE
    assert _compatible(dev1_publisher, DEPTH_CLOUD_QOS)


# fusion


def test_cloud_then_mask_same_stamp_fuses_geometry_lethal():
    row, col = _free_cell()
    core, _, logger = _geo_core()
    assert core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=NOW) is None
    data = _data(core.handle_mask(_mask_msg(), now_stamp=NOW))
    assert data[row, col] == 100  # semantic traversable did not clear geometry lethal
    assert logger.warnings == []


def test_mask_then_cloud_republishes_fused_costmap_for_that_mask(pipeline_spy):
    row, col = _free_cell()
    core, _, _ = _geo_core()
    first = core.handle_mask(_mask_msg(), now_stamp=NOW)
    assert _data(first)[row, col] == 0  # semantic-only until the cloud arrives
    second = core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=NOW)
    assert second is not None
    assert _data(second)[row, col] == 100
    assert (second.header.stamp.sec, second.header.stamp.nanosec) == (MASK_SEC, MASK_NSEC)
    (mask_inputs, _), (cloud_inputs, _) = pipeline_spy[-2:]
    assert mask_inputs.occupancy is None
    assert cloud_inputs.mask is mask_inputs.mask  # same mask, re-fused
    assert cloud_inputs.occupancy.stamp_ns == MASK_STAMP_NS


def test_fused_costmap_only_adds_lethal():
    row, col = _free_cell()
    semantic = _semantic_only()
    core, _, _ = _geo_core()
    core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=NOW)
    fused = _data(core.handle_mask(_mask_msg(), now_stamp=NOW))
    changed = fused != semantic
    assert np.argwhere(changed).tolist() == [[row, col]]
    assert (semantic == 100).sum() > 0 and np.all(fused[semantic == 100] == 100)


def test_ground_point_is_not_lethal():
    row, col = _free_cell()
    core, _, _ = _geo_core()
    core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col, height=0.0)]), now_stamp=NOW)
    np.testing.assert_array_equal(_data(core.handle_mask(_mask_msg(), now_stamp=NOW)), _semantic_only())


def test_no_cloud_is_semantic_only_unchanged(pipeline_spy):
    core, _, _ = _geo_core()
    data = _data(core.handle_mask(_mask_msg(), now_stamp=NOW))
    np.testing.assert_array_equal(data, _semantic_only())
    assert pipeline_spy[0][0].occupancy is None


def test_empty_cloud_clears_nothing(pipeline_spy):
    semantic = _semantic_only()
    core, _, _ = _geo_core()
    core.handle_cloud(_cloud_msg([]), now_stamp=NOW)
    data = _data(core.handle_mask(_mask_msg(), now_stamp=NOW))
    np.testing.assert_array_equal(data, semantic)  # hazards, unknowns kept
    occupancy = pipeline_spy[-1][0].occupancy
    assert occupancy is not None and not occupancy.occupied.any()


def test_empty_cloud_after_mask_republishes_identical_costmap():
    core, _, _ = _geo_core()
    first = _data(core.handle_mask(_mask_msg(), now_stamp=NOW))
    second = core.handle_cloud(_cloud_msg([]), now_stamp=NOW)
    np.testing.assert_array_equal(_data(second), first)


def test_out_of_grid_points_ignored():
    row, col = _free_cell()
    core, _, logger = _geo_core()
    points = [_obstacle_point_in(row, col), [15.0, 0.0, 1.0], [5.0, -9.0, 1.0]]  # 2 outside the grid
    core.handle_cloud(_cloud_msg(points), now_stamp=NOW)
    data = _data(core.handle_mask(_mask_msg(), now_stamp=NOW))
    assert np.argwhere(data != _semantic_only()).tolist() == [[row, col]]
    assert logger.warnings == []


# stamps, frames, TF


def test_cloud_tf_lookup_uses_cloud_stamp_and_frame():
    core, buffer, _ = _geo_core(**{"tf_timeout_s": 0.2})
    cloud_stamp = MASK_STAMP_NS - 33_000_000
    core.handle_cloud(_cloud_msg([[3.0, 0.0, 1.0]], stamp_ns=cloud_stamp), now_stamp=NOW)
    target, source, time, timeout = buffer.calls[-1]
    assert (target, source) == (GROUND_FRAME, CAMERA_FRAME)
    assert time.nanoseconds == cloud_stamp
    assert timeout.nanoseconds == 200_000_000


def test_cloud_frame_mismatch_not_paired():
    row, col = _free_cell()
    core, _, logger = _geo_core()
    core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)], frame_id="other_camera"), now_stamp=NOW)
    data = _data(core.handle_mask(_mask_msg(), now_stamp=NOW))
    np.testing.assert_array_equal(data, _semantic_only())
    assert any("cloud frame 'other_camera'" in w for w in logger.warnings)


def test_later_cloud_for_stale_mask_is_not_republished():
    row, col = _free_cell()
    core, _, _ = _geo_core()
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    late = _now_after_mask(600_000_000)  # mask_max_age_s = 0.5
    assert core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=late) is None


def test_same_cloud_twice_republishes_once():
    row, col = _free_cell()
    core, _, _ = _geo_core()
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    msg = _cloud_msg([_obstacle_point_in(row, col)])
    assert core.handle_cloud(msg, now_stamp=NOW) is not None
    assert core.handle_cloud(msg, now_stamp=NOW) is None


def test_older_cloud_does_not_replace_newer_observation():
    row, col = _free_cell()
    core, _, _ = _geo_core()
    core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=NOW)
    core.handle_cloud(_cloud_msg([], stamp_ns=MASK_STAMP_NS - 1), now_stamp=NOW)
    assert _data(core.handle_mask(_mask_msg(), now_stamp=NOW))[row, col] == 100


def test_cloud_tf_failure_logged_and_semantic_path_continues():
    core, _, logger = _geo_core(buffer=FakeBuffer(error=tf2_ros.LookupException("synthetic")))
    assert core.handle_cloud(_cloud_msg([[3.0, 0.0, 1.0]]), now_stamp=NOW) is None
    assert "cloud skipped: TF lookup failed (LookupException)" in logger.warnings[0]
    core._tf_lookup._buffer.error = None
    np.testing.assert_array_equal(_data(core.handle_mask(_mask_msg(), now_stamp=NOW)), _semantic_only())


def test_malformed_cloud_logged_and_skipped():
    core, buffer, logger = _geo_core()
    bad = _cloud_msg([[3.0, 0.0, 1.0]], data=b"\x00" * 5)
    assert core.handle_cloud(bad, now_stamp=NOW) is None
    assert buffer.calls == []
    assert "cloud skipped: ContractError" in logger.warnings[0]


def test_handle_cloud_refused_when_geometry_disabled():
    core, _, _ = _core()
    with pytest.raises(ContractError, match="geometry disabled"):
        core.handle_cloud(_cloud_msg([]), now_stamp=NOW)


# rclpy wiring


def test_node_subscribes_to_depth_cloud_when_enabled(ros_context):
    node = CostmapNode(parameter_overrides=_overrides(_geo_params()))
    try:
        (sub,) = [s for s in node.subscriptions if s.topic_name == "/test/depth_cloud"]
        assert sub.msg_type is PointCloud2
        q = sub.qos_profile
        assert (q.history, q.depth, q.reliability, q.durability) == (
            HistoryPolicy.KEEP_LAST, 1, ReliabilityPolicy.RELIABLE, DurabilityPolicy.VOLATILE,
        )
    finally:
        node.destroy_node()


def test_node_has_no_depth_cloud_subscription_when_disabled(ros_context):
    node = CostmapNode(parameter_overrides=_overrides(_params()))
    try:
        assert all(s.msg_type is not PointCloud2 for s in node.subscriptions)
    finally:
        node.destroy_node()


@pytest.mark.parametrize("name", [name for name, _ in GEOMETRY_PARAMETERS])
def test_node_refuses_to_start_with_geometry_enabled_but_parameter_missing(ros_context, name):
    values = _geo_params()
    del values[name]
    with pytest.raises(ParameterUninitializedException):
        CostmapNode(parameter_overrides=_overrides(values))


def test_node_refuses_to_start_without_geometry_enabled(ros_context):
    values = _params()
    del values["geometry.enabled"]
    with pytest.raises(ParameterUninitializedException):
        CostmapNode(parameter_overrides=_overrides(values))


def test_node_cloud_callback_publishes_fused_costmap(ros_context):
    node = CostmapNode(parameter_overrides=_overrides(_geo_params()))
    try:
        node._core._tf_lookup = TfCameraPoseLookup(FakeBuffer(), timeout_s=0.05)
        published = []
        node._pub_costmap.publish = published.append
        sec, nanosec = node.get_clock().now().seconds_nanoseconds()
        stamp_ns = sec * 1_000_000_000 + nanosec
        node._core.handle_camera_info(_info_msg(sec=sec, nanosec=nanosec))
        node._on_mask(_mask_msg(sec=sec, nanosec=nanosec))
        node._on_cloud(_cloud_msg([[3.0, 0.0, 1.0]], stamp_ns=stamp_ns))
        assert len(published) == 2
        assert all(g.header.frame_id == GROUND_FRAME for g in published)
    finally:
        node.destroy_node()


# --- geometry freshness relative to the mask (geometry.max_age_s) -----------------
#
# Rule: geometry with stamp G is fused into a mask with stamp M iff
# 0 <= M - G <= geometry.max_age_s (contracts.is_fresh(G, M, max_age)).
# G > M (future geometry) is never fused into that mask. Synthetic stamps.


def _mask_at(stamp_ns, **kwargs):
    sec, nanosec = divmod(stamp_ns, 1_000_000_000)
    return _mask_msg(sec=sec, nanosec=nanosec, **kwargs)


def _now_for(stamp_ns, age_ns=100_000_000):
    sec, nanosec = divmod(stamp_ns + age_ns, 1_000_000_000)
    return Time(sec=sec, nanosec=nanosec)


def _run(order, cloud_msg, *, core=None, mask_stamp_ns=MASK_STAMP_NS):
    """Feed one mask and one cloud in the given arrival order; return the
    last costmap published for that mask (None if nothing was published)."""
    if core is None:
        core, _, _ = _geo_core()
    now = _now_for(mask_stamp_ns)
    if order == "cloud_first":
        assert core.handle_cloud(cloud_msg, now_stamp=now) is None  # no mask yet
        return core.handle_mask(_mask_at(mask_stamp_ns), now_stamp=now)
    first = core.handle_mask(_mask_at(mask_stamp_ns), now_stamp=now)
    second = core.handle_cloud(cloud_msg, now_stamp=now)
    return first if second is None else second


ORDERS = ["cloud_first", "mask_first"]


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize(
    "age_ns",
    [0, 1, 60_000_000, GEOMETRY_MAX_AGE_NS],
    ids=["same_stamp", "1ns_older", "slightly_older", "exactly_max_age"],
)
def test_geometry_within_max_age_is_fused(order, age_ns):
    row, col = _free_cell()
    cloud = _cloud_msg([_obstacle_point_in(row, col)], stamp_ns=MASK_STAMP_NS - age_ns)
    data = _data(_run(order, cloud))
    assert data[row, col] == 100


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize(
    "age_ns",
    [GEOMETRY_MAX_AGE_NS + 1, 400_000_000, 5_000_000_000],
    ids=["1ns_past_max_age", "much_older", "seconds_older"],
)
def test_stale_geometry_is_not_fused(order, age_ns, pipeline_spy):
    row, col = _free_cell()
    cloud = _cloud_msg([_obstacle_point_in(row, col)], stamp_ns=MASK_STAMP_NS - age_ns)
    data = _data(_run(order, cloud))
    semantic = _semantic_only()
    np.testing.assert_array_equal(data, semantic)  # no new obstacle cell, nothing cleared
    assert (semantic == 100).any() and np.all(data[semantic == 100] == 100)  # hazards kept
    assert pipeline_spy[-1][0].occupancy is None


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("ahead_ns", [1, 50_000_000], ids=["1ns_newer", "50ms_newer"])
def test_future_geometry_is_not_fused_into_older_mask(order, ahead_ns, pipeline_spy):
    row, col = _free_cell()
    cloud = _cloud_msg([_obstacle_point_in(row, col)], stamp_ns=MASK_STAMP_NS + ahead_ns)
    data = _data(_run(order, cloud))
    np.testing.assert_array_equal(data, _semantic_only())
    assert pipeline_spy[-1][0].occupancy is None


def test_future_geometry_stays_stored_for_its_own_mask():
    row, col = _free_cell()
    core, _, _ = _geo_core()
    later = MASK_STAMP_NS + 50_000_000
    core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)], stamp_ns=later), now_stamp=NOW)
    assert _data(core.handle_mask(_mask_at(MASK_STAMP_NS), now_stamp=NOW))[row, col] == 0
    assert _data(core.handle_mask(_mask_at(later), now_stamp=_now_for(later)))[row, col] == 100


def test_stale_geometry_is_not_reused_for_later_masks():
    row, col = _free_cell()
    core, _, _ = _geo_core()
    core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=NOW)
    # Same-stamp mask: fused.
    assert _data(core.handle_mask(_mask_at(MASK_STAMP_NS), now_stamp=NOW))[row, col] == 100
    # Mask within max age of the cloud: still fused.
    m2 = MASK_STAMP_NS + GEOMETRY_MAX_AGE_NS
    assert _data(core.handle_mask(_mask_at(m2), now_stamp=_now_for(m2)))[row, col] == 100
    # Mask past max age: not fused, and the stored cloud is discarded.
    m3 = m2 + 1
    assert _data(core.handle_mask(_mask_at(m3), now_stamp=_now_for(m3)))[row, col] == 0
    assert core._geometry is None
    # Later masks never see it again.
    for m in (m3 + 1_000_000, m3 + 10_000_000_000):
        np.testing.assert_array_equal(
            _data(core.handle_mask(_mask_at(m), now_stamp=_now_for(m))), _semantic_only()
        )


def test_rejected_stale_mask_does_not_discard_stored_geometry():
    """A mask rejected as stale by mask_max_age_s does not touch stored geometry."""
    row, col = _free_cell()
    core, _, _ = _geo_core()
    core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=NOW)
    far = MASK_STAMP_NS + 1_000_000_000
    assert core.handle_mask(_mask_at(far), now_stamp=_now_for(far, age_ns=900_000_000)) is None
    assert core._geometry is not None
    assert _data(core.handle_mask(_mask_at(MASK_STAMP_NS), now_stamp=NOW))[row, col] == 100


def test_fresh_empty_cloud_does_not_clear_semantic_hazard(pipeline_spy):
    semantic = _semantic_only()
    for order in ORDERS:
        data = _data(_run(order, _cloud_msg([], stamp_ns=MASK_STAMP_NS - 50_000_000)))
        np.testing.assert_array_equal(data, semantic)
        assert np.all(data[semantic == 100] == 100)
        occupancy = pipeline_spy[-1][0].occupancy
        assert occupancy is not None and not occupancy.occupied.any()  # fused, adds nothing


def test_stale_empty_cloud_is_ignored(pipeline_spy):
    for order in ORDERS:
        stale = _cloud_msg([], stamp_ns=MASK_STAMP_NS - GEOMETRY_MAX_AGE_NS - 1)
        data = _data(_run(order, stale))
        np.testing.assert_array_equal(data, _semantic_only())
        assert pipeline_spy[-1][0].occupancy is None


def test_republish_only_when_newer_geometry_improves_pairing():
    row, col = _free_cell()
    core, _, _ = _geo_core()
    core.handle_mask(_mask_at(MASK_STAMP_NS), now_stamp=NOW)
    point = [_obstacle_point_in(row, col)]
    assert core.handle_cloud(_cloud_msg(point, stamp_ns=MASK_STAMP_NS - 60_000_000), now_stamp=NOW)
    assert core.handle_cloud(_cloud_msg(point, stamp_ns=MASK_STAMP_NS), now_stamp=NOW)
    assert core.handle_cloud(_cloud_msg(point, stamp_ns=MASK_STAMP_NS - 30_000_000), now_stamp=NOW) is None


def test_geometry_max_age_does_not_change_mask_freshness():
    core, buffer, logger = _geo_core(**{"geometry.max_age_s": 100.0})
    # mask_max_age_s = 0.5 still rejects a 0.8 s old mask, before any TF lookup.
    assert core.handle_mask(_mask_msg(), now_stamp=_now_after_mask(800_000_000)) is None
    assert buffer.calls == [] and "stale" in logger.warnings[0]
    # and a tiny geometry.max_age_s does not reject a fresh mask.
    core, _, _ = _geo_core(**{"geometry.max_age_s": 1e-9})
    assert core.handle_mask(_mask_msg(), now_stamp=_now_after_mask(400_000_000)) is not None


def test_no_geometry_available_semantic_only_unchanged(pipeline_spy):
    core, _, _ = _geo_core()
    for m in (MASK_STAMP_NS, MASK_STAMP_NS + 33_000_000):
        np.testing.assert_array_equal(
            _data(core.handle_mask(_mask_at(m), now_stamp=_now_for(m))), _semantic_only()
        )
        assert pipeline_spy[-1][0].occupancy is None


def test_old_symmetric_stamp_tolerance_parameter_removed():
    with pytest.raises(ContractError, match="unknown.*geometry.stamp_tolerance_s"):
        config_from_parameters(_geo_params(**{"geometry.stamp_tolerance_s": 0.05}))
    assert "geometry.stamp_tolerance_s" not in {name for name, _ in GEOMETRY_PARAMETERS}


# --- stale-mask fail-safe (architecture §8.6) ---------------------------------------
#
# Synthetic ROI from _params(): x [0, 2], y [-1, 1] in the 0.5 m test grid
# -> rows 2..5, cols 0..3. OccupancyGrid values: lethal 100, unknown -1.

FAIL_SAFE_ROI = {(r, c) for r in range(2, 6) for c in range(4)}


def _assert_fail_safe_grid(grid_msg, now):
    data = _data(grid_msg)
    roi = np.zeros_like(data, dtype=bool)
    for r, c in FAIL_SAFE_ROI:
        roi[r, c] = True
    assert np.all(data[roi] == 100)
    assert np.all(data[~roi] == -1)
    assert not np.any(data == 0)  # nothing free
    assert grid_msg.header.stamp == now
    assert grid_msg.header.frame_id == GROUND_FRAME


def test_fail_safe_at_startup_before_any_mask():
    core, _, logger = _core()
    now = _now_after_mask(0)
    grid = core.handle_fail_safe_check(now_stamp=now)
    _assert_fail_safe_grid(grid, now)
    assert core.fail_safe_active
    assert "entering stale-mask fail-safe: no mask has produced a costmap" in logger.warnings[0]


def test_fresh_mask_suppresses_fail_safe_and_output_is_unchanged():
    core, _, _ = _core()
    core.handle_camera_info(_info_msg())
    normal = core.handle_mask(_mask_msg(), now_stamp=NOW)
    np.testing.assert_array_equal(_data(normal), _semantic_only())
    assert core.handle_fail_safe_check(now_stamp=NOW) is None
    assert not core.fail_safe_active


@pytest.mark.parametrize("age_ns, fail_safe", [(500_000_000, False), (500_000_001, True)])
def test_fail_safe_mask_age_boundary(age_ns, fail_safe):
    # Same rule as mask freshness: 0 <= age <= mask_max_age_s (0.5 s) is fresh.
    core, _, _ = _core()
    core.handle_camera_info(_info_msg())
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    result = core.handle_fail_safe_check(now_stamp=_now_after_mask(age_ns))
    assert (result is not None) is fail_safe
    assert core.fail_safe_active is fail_safe


def test_fresh_to_stale_transition_replaces_normal_output():
    core, _, logger = _core()
    core.handle_camera_info(_info_msg())
    normal = _data(core.handle_mask(_mask_msg(), now_stamp=NOW))
    assert (normal == 0).any()  # the normal costmap has free (traversable) cells
    later = _now_after_mask(800_000_000)
    grid = core.handle_fail_safe_check(now_stamp=later)
    _assert_fail_safe_grid(grid, later)
    assert "last mask age 0.800000 s" in logger.warnings[-1]


def test_stale_mask_arrival_still_skipped_and_fail_safe_follows():
    core, buffer, _ = _core()
    core.handle_camera_info(_info_msg())
    late = _now_after_mask(800_000_000)
    assert core.handle_mask(_mask_msg(), now_stamp=late) is None  # unchanged behaviour
    assert buffer.calls == []
    _assert_fail_safe_grid(core.handle_fail_safe_check(now_stamp=late), late)


def test_repeated_fail_safe_checks_do_not_compound():
    core, _, logger = _core()
    grids = [core.handle_fail_safe_check(now_stamp=_now_after_mask(k * 100_000_000)) for k in range(1, 6)]
    first = _data(grids[0])
    for grid in grids[1:]:
        np.testing.assert_array_equal(_data(grid), first)
    assert sum("entering stale-mask fail-safe" in w for w in logger.warnings) == 1  # logged once


def test_new_fresh_mask_exits_fail_safe():
    core, _, logger = _core()
    core.handle_camera_info(_info_msg())
    core.handle_fail_safe_check(now_stamp=NOW)
    assert core.fail_safe_active
    normal = core.handle_mask(_mask_msg(), now_stamp=NOW)
    np.testing.assert_array_equal(_data(normal), _semantic_only())
    assert not core.fail_safe_active
    assert "leaving stale-mask fail-safe" in logger.warnings[-1]
    assert core.handle_fail_safe_check(now_stamp=NOW) is None


def test_stale_then_fresh_then_stale_again():
    core, _, _ = _core()
    core.handle_camera_info(_info_msg())
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    assert core.handle_fail_safe_check(now_stamp=_now_after_mask(600_000_000)) is not None
    m2 = MASK_STAMP_NS + 1_000_000_000
    assert core.handle_mask(_mask_at(m2), now_stamp=_now_for(m2)) is not None
    assert core.handle_fail_safe_check(now_stamp=_now_for(m2)) is None
    assert core.handle_fail_safe_check(now_stamp=_now_for(m2, age_ns=600_000_000)) is not None


def test_fresh_mask_that_fails_processing_does_not_exit_fail_safe():
    core, _, _ = _core(buffer=FakeBuffer(error=tf2_ros.LookupException("synthetic")))
    core.handle_camera_info(_info_msg())
    core.handle_fail_safe_check(now_stamp=NOW)
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is None
    assert core.fail_safe_active
    assert core.handle_fail_safe_check(now_stamp=NOW) is not None


def test_zero_clock_fail_safe_check_skipped():
    core, _, logger = _core()
    assert core.handle_fail_safe_check(now_stamp=Time(sec=0, nanosec=0)) is None
    assert "node clock is 0" in logger.warnings[0]


def test_no_geometry_stale_mask_is_fail_safe():
    core, _, _ = _geo_core()
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    later = _now_after_mask(700_000_000)
    _assert_fail_safe_grid(core.handle_fail_safe_check(now_stamp=later), later)


def test_geometry_arriving_while_stale_does_not_bypass_fail_safe():
    row, col = _free_cell()
    core, _, _ = _geo_core()
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    later = _now_after_mask(700_000_000)
    reference = _data(core.handle_fail_safe_check(now_stamp=later))
    # Same-stamp cloud for the (now stale) mask: no republish of that mask.
    assert core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=later) is None
    # A newer cloud with no fresh mask to pair with: nothing published either.
    newer = MASK_STAMP_NS + 600_000_000
    assert core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)], stamp_ns=newer),
                             now_stamp=later) is None
    np.testing.assert_array_equal(_data(core.handle_fail_safe_check(now_stamp=later)), reference)
    assert core.fail_safe_active


def test_cloud_first_then_fresh_mask_exits_fail_safe_with_geometry():
    row, col = _free_cell()
    core, _, _ = _geo_core()
    core.handle_fail_safe_check(now_stamp=NOW)
    assert core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=NOW) is None
    data = _data(core.handle_mask(_mask_msg(), now_stamp=NOW))
    assert not core.fail_safe_active
    assert data[row, col] == 100


def test_mask_first_cloud_after_mask_went_stale_is_not_republished():
    row, col = _free_cell()
    core, _, _ = _geo_core()
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    later = _now_after_mask(600_000_000)
    assert core.handle_fail_safe_check(now_stamp=later) is not None
    assert core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=later) is None


@pytest.mark.parametrize(
    "override",
    [
        {"fail_safe.roi_min_x": 2.0},                 # == max_x
        {"fail_safe.roi_max_y": -1.5},                # < min_y
        {"fail_safe.roi_min_x": math.nan},
        {"fail_safe.roi_max_x": math.inf},
        {"fail_safe.roi_min_x": -0.5},                # outside the grid (x starts at 0)
        {"fail_safe.roi_max_y": 2.5},                 # outside the grid (y ends at 2)
        {"fail_safe.check_period_s": 0.0},
        {"fail_safe.check_period_s": -0.1},
        {"fail_safe.check_period_s": math.nan},
        {"fail_safe.check_period_s": math.inf},
    ],
)
def test_invalid_fail_safe_configuration_rejected(override):
    with pytest.raises(ContractError):
        config_from_parameters(_params(**override))


def test_fail_safe_config_built_from_parameters():
    fs = config_from_parameters(_params()).fail_safe
    r = fs.region
    assert (r.min_x, r.max_x, r.min_y, r.max_y) == (0.0, 2.0, -1.0, 1.0)
    assert fs.check_period_s == 0.1


def test_node_creates_fail_safe_timer_with_configured_period(ros_context):
    node = CostmapNode(parameter_overrides=_overrides(_params(**{"fail_safe.check_period_s": 0.25})))
    try:
        assert node._fail_safe_timer.timer_period_ns == 250_000_000
    finally:
        node.destroy_node()


def test_node_fail_safe_callback_publishes_on_costmap_topic(ros_context):
    node = CostmapNode(parameter_overrides=_overrides(_params()))
    try:
        published = []
        node._pub_costmap.publish = published.append
        node._on_fail_safe_check()  # no mask yet
        assert len(published) == 1
        data = _data(published[0])
        assert (data == 100).sum() == len(FAIL_SAFE_ROI) and not (data == 0).any()
    finally:
        node.destroy_node()


# --- footprint (Dev 5 owns the real values; everything here is synthetic) ------------

from costmap_ros.costmap_node import FOOTPRINT_PARAMETERS  # noqa: E402

# Synthetic 1.0 m x 0.6 m rectangle, NOT the robot's footprint.
SYNTHETIC_FOOTPRINT_FLAT = [0.5, 0.3, -0.5, 0.3, -0.5, -0.3, 0.5, -0.3]
SYNTHETIC_ROBOT_FRAME = "test_robot_frame"


def _fp_params(**overrides):
    values = _params(**{
        "footprint.enabled": True,
        "footprint.vertices": list(SYNTHETIC_FOOTPRINT_FLAT),
        "footprint.frame_id": SYNTHETIC_ROBOT_FRAME,
        "footprint.padding": 0.1,
    })
    values.update(overrides)
    return values


@pytest.fixture
def result_spy(monkeypatch):
    results = []
    real = costmap_node.run_costmap_pipeline

    def spy(inputs, **kwargs):
        result = real(inputs, **kwargs)
        results.append((inputs, kwargs, result))
        return result

    monkeypatch.setattr(costmap_node, "run_costmap_pipeline", spy)
    return results


def _fp_core(base_params, *, buffer=None):
    config = config_from_parameters(base_params)
    buffer = buffer if buffer is not None else FakeBuffer()
    core = CostmapNodeCore(config, TfCameraPoseLookup(buffer, timeout_s=config.tf_timeout_s),
                           RecordingLogger())
    core.handle_camera_info(_info_msg())
    return core


# configuration


def test_footprint_config_built_from_parameters():
    fp = config_from_parameters(_fp_params()).footprint
    assert fp.footprint.vertices == ((0.5, 0.3), (-0.5, 0.3), (-0.5, -0.3), (0.5, -0.3))
    assert fp.footprint.frame_id == SYNTHETIC_ROBOT_FRAME
    assert fp.padding == 0.1


def test_footprint_disabled_has_no_footprint_config():
    assert config_from_parameters(_params()).footprint is None


def test_footprint_enabled_is_required():
    values = _params()
    del values["footprint.enabled"]
    with pytest.raises(ContractError, match="missing.*footprint.enabled"):
        config_from_parameters(values)


@pytest.mark.parametrize("name", [name for name, _ in FOOTPRINT_PARAMETERS])
def test_every_footprint_parameter_required_when_enabled(name):
    values = _fp_params()
    del values[name]
    with pytest.raises(ContractError, match="missing"):
        config_from_parameters(values)


@pytest.mark.parametrize("name", [name for name, _ in FOOTPRINT_PARAMETERS])
def test_footprint_parameter_rejected_when_disabled(name):
    with pytest.raises(ContractError, match="unknown"):
        config_from_parameters(_params(**{name: _fp_params()[name]}))


@pytest.mark.parametrize(
    "vertices",
    [
        [0.5, 0.3, -0.5, 0.3, -0.5],                          # odd length
        [],                                                   # no vertices
        [0.0, 0.0, 1.0, 0.0],                                 # 2 vertices
        [0.0, 0.0, 1.0, 0.0, 2.0, 0.0],                       # collinear, zero area
        [0.5, 0.3, -0.5, math.nan, -0.5, -0.3, 0.5, -0.3],    # NaN
        [0.5, 0.3, -0.5, 0.3, -math.inf, -0.3, 0.5, -0.3],    # Inf
        [0.0, 0.0, 2.0, 0.0, 1.0, 0.5, 2.0, 2.0, 0.0, 2.0],   # concave
        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0],             # self-intersecting
        [0.5, 0.3, -0.5, 0.3, -0.5, -0.3, 0.5, -0.3, 0.5, 0.3],  # explicitly closed ring
        [0.0, 0.0, 10 ** 400, 0.0, 0.0, 1.0],                 # int beyond float range
        [0.0, 0.0, 1e160, 1e160, 2e160, 2e160],               # overflowing area
        "0.5, 0.3, -0.5, 0.3",                                # not a list
    ],
    ids=["odd", "empty", "two_vertices", "zero_area", "nan", "inf", "concave",
         "self_intersecting", "closed_ring", "huge_int", "overflow", "string"],
)
def test_invalid_footprint_vertices_rejected(vertices):
    with pytest.raises(ContractError):
        config_from_parameters(_fp_params(**{"footprint.vertices": vertices}))


@pytest.mark.parametrize("padding", [-0.01, math.nan, math.inf, 10 ** 400, True, "0.1"])
def test_invalid_footprint_padding_rejected(padding):
    with pytest.raises(ContractError):
        config_from_parameters(_fp_params(**{"footprint.padding": padding}))


@pytest.mark.parametrize("override", [{"footprint.frame_id": ""}, {"footprint.enabled": "yes"}])
def test_invalid_footprint_frame_or_switch_rejected(override):
    with pytest.raises(ContractError):
        config_from_parameters(_fp_params(**override))


def test_zero_padding_is_allowed():
    assert config_from_parameters(_fp_params(**{"footprint.padding": 0.0})).footprint.padding == 0.0


# runtime pass-through


def test_footprint_is_carried_and_padded_through_pipeline(result_spy):
    core = _fp_core(_fp_params())
    assert core.handle_mask(_mask_msg(), now_stamp=NOW) is not None
    inputs, kwargs, result = result_spy[-1]
    assert inputs.footprint.frame_id == SYNTHETIC_ROBOT_FRAME
    assert kwargs == {"inflation_radius": 0.0, "footprint_padding": 0.1}
    flat = [coord for vertex in result.footprint for coord in vertex]
    assert flat == pytest.approx([0.6, 0.4, -0.6, 0.4, -0.6, -0.4, 0.6, -0.4])
    assert isinstance(result.footprint, tuple)  # immutable


def test_footprint_padding_is_deterministic_across_updates(result_spy):
    core = _fp_core(_fp_params())
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    assert result_spy[-1][2].footprint == result_spy[-2][2].footprint


def test_footprint_does_not_change_published_costmap():
    semantic = _semantic_only()
    core = _fp_core(_fp_params(**{"footprint.padding": 0.3}))
    np.testing.assert_array_equal(_data(core.handle_mask(_mask_msg(), now_stamp=NOW)), semantic)


def test_footprint_does_not_change_geometry_fusion():
    row, col = _free_cell()
    cloud = _cloud_msg([_obstacle_point_in(row, col)])
    outputs = []
    for base in (_geo_params(), _geo_params(**{
        "footprint.enabled": True,
        "footprint.vertices": list(SYNTHETIC_FOOTPRINT_FLAT),
        "footprint.frame_id": SYNTHETIC_ROBOT_FRAME,
        "footprint.padding": 0.1,
    })):
        core = _fp_core(base)
        core.handle_cloud(cloud, now_stamp=NOW)
        outputs.append(_data(core.handle_mask(_mask_msg(), now_stamp=NOW)))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert outputs[1][row, col] == 100


def test_footprint_does_not_change_fail_safe():
    later = _now_after_mask(900_000_000)
    without = _core()[0].handle_fail_safe_check(now_stamp=later)
    with_fp = _fp_core(_fp_params()).handle_fail_safe_check(now_stamp=later)
    np.testing.assert_array_equal(_data(with_fp), _data(without))
    _assert_fail_safe_grid(with_fp, later)


def test_footprint_mask_then_cloud_refusion_carries_footprint(result_spy):
    row, col = _free_cell()
    core = _fp_core(_geo_params(**{
        "footprint.enabled": True,
        "footprint.vertices": list(SYNTHETIC_FOOTPRINT_FLAT),
        "footprint.frame_id": SYNTHETIC_ROBOT_FRAME,
        "footprint.padding": 0.0,
    }))
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    assert core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=NOW) is not None
    inputs, kwargs, _ = result_spy[-1]
    assert inputs.occupancy is not None and inputs.footprint is not None
    assert kwargs["footprint_padding"] == 0.0


# rclpy wiring


def test_node_accepts_footprint_parameters(ros_context):
    node = CostmapNode(parameter_overrides=_overrides(_fp_params()))
    try:
        assert node.config.footprint.footprint.vertices[0] == (0.5, 0.3)
        assert node.config.footprint.padding == 0.1
    finally:
        node.destroy_node()


@pytest.mark.parametrize("name", [name for name, _ in FOOTPRINT_PARAMETERS])
def test_node_refuses_to_start_with_footprint_enabled_but_parameter_missing(ros_context, name):
    values = _fp_params()
    del values[name]
    with pytest.raises(ParameterUninitializedException):
        CostmapNode(parameter_overrides=_overrides(values))


def test_node_refuses_to_start_with_invalid_footprint(ros_context):
    with pytest.raises(ContractError):
        CostmapNode(parameter_overrides=_overrides(
            _fp_params(**{"footprint.vertices": [0.0, 0.0, 1.0, 0.0, 2.0, 0.0]})))


# --- regression: a future-stamped cloud must not shadow later valid clouds ---------


def test_cloud_stamped_after_node_clock_is_rejected_and_not_stored():
    core, buffer, logger = _geo_core()
    ahead = stamp_to_ns_(NOW) + 1
    assert core.handle_cloud(_cloud_msg([[3.0, 0.0, 1.0]], stamp_ns=ahead), now_stamp=NOW) is None
    assert core._geometry is None
    assert buffer.calls == []  # rejected before any TF lookup
    assert "cloud skipped: future-stamped" in logger.warnings[0]


def test_cloud_stamped_exactly_at_node_clock_is_accepted():
    core, _, logger = _geo_core()
    core.handle_cloud(_cloud_msg([], stamp_ns=stamp_to_ns_(NOW)), now_stamp=NOW)
    assert core._geometry is not None and logger.warnings == []


def test_backwards_clock_jump_does_not_block_cloud_first_geometry():
    """A cloud stored before a backwards clock jump (bag loop, sim reset) is
    ahead of the new clock; it must not shadow every later valid cloud."""
    row, col = _free_cell()
    core, _, _ = _geo_core()
    before_jump_ns = MASK_STAMP_NS + 3600 * 1_000_000_000
    before_now = _now_for(before_jump_ns)
    core.handle_cloud(_cloud_msg([], stamp_ns=before_jump_ns), now_stamp=before_now)
    assert core._geometry.occupancy.stamp_ns == before_jump_ns
    # Clock jumps back: same-stamp obstacle cloud, then its mask (cloud-first).
    core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=NOW)
    assert core._geometry.occupancy.stamp_ns == MASK_STAMP_NS
    assert _data(core.handle_mask(_mask_msg(), now_stamp=NOW))[row, col] == 100


def stamp_to_ns_(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


# --- output stamp contract (every publish path) --------------------------------------
#
# normal mask output   : mask header stamp ("mask") or node clock ("now")
# re-fused republish   : the SAME mask's header stamp ("mask") or node clock ("now")
# fail-safe grid       : always node clock (no current mask exists)


def _stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def test_refused_republish_stamp_follows_now_policy():
    row, col = _free_cell()
    core, _, _ = _geo_core(output_stamp_source="now")
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    later = _now_after_mask(150_000_000)
    grid = core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=later)
    assert grid is not None and grid.header.stamp == later


def test_fail_safe_stamp_is_node_clock_under_now_policy():
    core, _, _ = _core(output_stamp_source="now")
    later = _now_after_mask(900_000_000)
    assert core.handle_fail_safe_check(now_stamp=later).header.stamp == later


def test_mask_policy_recovery_stamp_can_precede_last_fail_safe_stamp():
    """Documented contract: with output_stamp_source == "mask", stamps are NOT
    monotonic across fail-safe -> recovery. Consumers must take the latest
    ARRIVAL, not the largest stamp."""
    core, _, _ = _core(output_stamp_source="mask")
    core.handle_camera_info(_info_msg())
    fail_safe_at = _now_after_mask(300_000_000)          # node clock, no mask yet
    fail_safe = core.handle_fail_safe_check(now_stamp=fail_safe_at)
    recovery = core.handle_mask(_mask_msg(), now_stamp=fail_safe_at)  # mask is 0.3 s old: fresh
    assert _stamp_ns(recovery.header.stamp) == MASK_STAMP_NS
    assert _stamp_ns(recovery.header.stamp) < _stamp_ns(fail_safe.header.stamp)


# --- live DDS: Dev 1-shaped mask + cloud through real topics -> published costmap -------


def test_live_dds_depth_cloud_reaches_published_costmap(ros_context):
    """Real publishers/subscriptions (not direct handler calls). The cloud layout
    and QoS copy Dev 1 node/cloud.py + adapter_node.py (create_publisher(..., 10));
    turing/ is not imported. TF is a fake buffer; all values are synthetic."""
    row, col = _free_cell()
    node = CostmapNode(parameter_overrides=_overrides(_geo_params()))
    peer = rclpy.create_node("dev1_stand_in")
    try:
        node._core._tf_lookup = TfCameraPoseLookup(FakeBuffer(), timeout_s=0.05)
        pub_info = peer.create_publisher(CameraInfo, "/test/camera_info", 10)
        pub_mask = peer.create_publisher(Image, "/test/mask", 10)
        pub_cloud = peer.create_publisher(PointCloud2, "/test/depth_cloud", 10)
        received = []
        peer.create_subscription(OccupancyGrid, "/test/costmap", received.append, 10)

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(peer)

        def spin_until(predicate, timeout_s=5.0):
            deadline = time.monotonic() + timeout_s
            while not predicate() and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.05)
            return predicate()

        sec, nanosec = node.get_clock().now().seconds_nanoseconds()
        stamp_ns = sec * 1_000_000_000 + nanosec
        assert spin_until(lambda: pub_info.get_subscription_count() > 0
                          and pub_mask.get_subscription_count() > 0
                          and pub_cloud.get_subscription_count() > 0)
        pub_info.publish(_info_msg(sec=sec, nanosec=nanosec))
        assert spin_until(lambda: node._core.has_camera_info)

        def normal_grids():  # fail-safe ticks are node-clock stamped; keep this mask's grids
            return [g for g in received if (g.header.stamp.sec, g.header.stamp.nanosec) == (sec, nanosec)]

        pub_mask.publish(_mask_msg(sec=sec, nanosec=nanosec))            # Dev 1 order: mask first
        assert spin_until(lambda: len(normal_grids()) >= 1)
        assert _data(normal_grids()[0])[row, col] == 0                    # semantic-only so far
        pub_cloud.publish(_cloud_msg([_obstacle_point_in(row, col)], stamp_ns=stamp_ns))
        assert spin_until(lambda: len(normal_grids()) >= 2)
        fused = _data(normal_grids()[1])
        assert fused[row, col] == 100                                     # geometry lethal fused
        assert normal_grids()[1].header.frame_id == GROUND_FRAME
        executor.remove_node(node)
        executor.remove_node(peer)
        executor.shutdown()
    finally:
        peer.destroy_node()
        node.destroy_node()


# --- published output is the fused, NON-inflated costmap (Nav2 owns inflation) --------

from costmap_ros.occupancy_grid import costs_to_occupancy  # noqa: E402

SYNTHETIC_INFLATION_RADIUS = 3.0  # synthetic; large enough that Dev 3 inflation changes free cells here


def _assert_published_is_fused_not_final(grid_msg, result):
    data = _data(grid_msg)
    np.testing.assert_array_equal(data, costs_to_occupancy(result.fused))
    assert set(np.unique(data).tolist()) <= {0, 100, -1}  # no inflated 1..99
    assert not np.array_equal(result.final, result.fused)  # inflation was computed...
    assert not np.array_equal(data, costs_to_occupancy(result.final))  # ...but not published


def test_published_mask_costmap_is_not_inflated(result_spy):
    core, _, _ = _core(inflation_radius=SYNTHETIC_INFLATION_RADIUS)
    core.handle_camera_info(_info_msg())
    grid = core.handle_mask(_mask_msg(), now_stamp=NOW)
    _, kwargs, result = result_spy[-1]
    assert kwargs["inflation_radius"] == SYNTHETIC_INFLATION_RADIUS
    _assert_published_is_fused_not_final(grid, result)


def test_published_refused_costmap_is_not_inflated(result_spy):
    row, col = _free_cell()
    core, _, _ = _geo_core(inflation_radius=SYNTHETIC_INFLATION_RADIUS)
    core.handle_mask(_mask_msg(), now_stamp=NOW)
    grid = core.handle_cloud(_cloud_msg([_obstacle_point_in(row, col)]), now_stamp=NOW)
    _, _, result = result_spy[-1]
    _assert_published_is_fused_not_final(grid, result)
    data = _data(grid)
    assert data[row, col] == 100                      # geometry lethal still fused
    neighbours = data[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2]
    assert not np.any((neighbours > 0) & (neighbours < 100))  # no Dev 3 inflation ring


def test_inflation_radius_does_not_change_published_mask_costmap():
    grids = []
    for radius in (0.0, 1.0, SYNTHETIC_INFLATION_RADIUS):
        core, _, _ = _core(inflation_radius=radius)
        core.handle_camera_info(_info_msg())
        grids.append(_data(core.handle_mask(_mask_msg(), now_stamp=NOW)))
    for other in grids[1:]:
        np.testing.assert_array_equal(other, grids[0])
