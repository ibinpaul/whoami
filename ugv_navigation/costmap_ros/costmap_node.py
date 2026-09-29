"""Dev 3 costmap ROS 2 node skeleton (semantic input + optional geometry).

    mask (/segmentation/mask)  --mask_adapter-->        SemanticMaskInput
    latest CameraInfo          --camera_info_adapter--> CameraIntrinsicsInput
    TF target_frame <- mask.frame_id at mask stamp --tf_lookup--> CameraGroundInput
    ROS parameters             --grid_params-->         GridInput
    optional depth cloud       --see "Geometry" below-> OccupancyInput
        -> CostmapCoreInputs -> run_costmap_pipeline() -> OccupancyGrid (temporary)

Published costmap = CostmapPipelineResult.fused: semantic + geometry after
geometry-lethal precedence, NOT inflated. Obstacle inflation is Nav2's
(downstream). result.final (Dev 3's own inflation) is still computed by the
pipeline but not published; inflation.py is kept for independent use.

Every real-world value (topics, frames, grid, timeout, inflation radius,
mask max age, CameraInfo durability, output stamp) is a required ROS
parameter with no default.
Frame names are never hard-coded: the camera frame is the mask's
header.frame_id, the grid/ground frame is `target_frame`.

Freshness (architecture §8.4, "consumer rejects if age > max age"): a mask
is used only if contracts.is_fresh(mask stamp, now, mask_max_age_s), where
now is the node's ROS clock (system time, or /clock under use_sim_time) --
the clock the header stamps are expressed in. Stale and future-stamped
masks are rejected before the TF lookup. The limit is a required parameter;
Dev 3's value is PENDING (Dev 1's own perception_max_age is not reused).

CameraInfo pairing: the latest valid CameraInfo is used with each mask.
Its header.stamp is NOT compared with the mask stamp. CameraInfo keeps its
own stamp (Dev 1 republishes the last received message unchanged, after
the mask; a latched TRANSIENT_LOCAL calibration can be arbitrarily old),
and no project document defines a calibration lifetime or a CameraInfo
stamp tolerance. Safety comes from content checks instead, per mask:
CameraInfo received and valid (finite, non-placeholder, zero-skew K with
fx, fy > 0), frame_id == mask frame_id, and width/height == mask size
(CostmapCoreInputs). An invalid CameraInfo clears the stored one.

CameraInfo QoS: RELIABLE, KEEP_LAST depth 1, and a required durability
(CAMERA_INFO_DURABILITIES) because the right value depends on the PENDING
topic choice: Dev 1's /segmentation/camera_info republisher is VOLATILE,
which a TRANSIENT_LOCAL subscription cannot match, while the driver topic
is expected to be TRANSIENT_LOCAL (Dev 1 subscribes to it that way), and
only a TRANSIENT_LOCAL subscription receives its latched calibration.

Geometry (optional, `geometry.enabled`): each PointCloud2 on
geometry.depth_cloud_topic (Dev 1 /perception/depth_cloud) goes through
point_cloud_adapter -> TF target_frame <- cloud.frame_id at the cloud's own
stamp (tf_lookup) -> pipeline.point_cloud_to_occupancy (transform, obstacle
filter, rasterise). The resulting OccupancyInput is paired with a mask only
if both come from the same camera frame and the geometry is fresh relative
to the mask -- actual header stamps, not arrival order:

    0 <= mask stamp - cloud stamp <= geometry.max_age_s
    (contracts.is_fresh(cloud stamp, mask stamp, geometry.max_age_s))

A cloud stamped after the mask (future geometry) is never fused into that
mask: Dev 1 stamps the cloud with the image capture time of the same tick,
so for a given mask its cloud has the same stamp, never a later one; a
later stamp belongs to a later image. Such a cloud stays stored and may
pair with a later mask. This is a separate check from mask freshness
(mask stamp vs node clock, mask_max_age_s), which is unchanged.

The latest (newest-stamped) observation is kept. Once a fresh mask finds it
stale (mask stamp - cloud stamp > geometry.max_age_s) it is discarded, so it
can never be selected for a later mask. Pairing works in either order:
a mask uses the kept observation if it pairs; a cloud that pairs with the
last processed mask better than what that mask used re-runs the pipeline
for that mask (if it is still fresh) and publishes again. Dev 1 publishes
the cloud after the mask of the same image, so this second publication is
how a same-stamp cloud reaches the costmap.
Geometry only adds lethal cells through the existing fusion (geometry
lethal wins; nothing clears a semantic obstacle). No cloud, a cloud that
does not pair, a failed cloud, or an empty cloud all leave the semantic
result unchanged: dropped or missing points are never free space.

Stale-mask fail-safe (architecture §8.6: "Stale or adapter failure ->
/ugv/perception_degraded + front ROI lethal/max-inflate + safety hold"):
every fail_safe.check_period_s a timer checks whether the last mask that
produced a costmap is still fresh (is_fresh(mask stamp, now,
mask_max_age_s) -- the same rule as mask freshness). If there is none, or it
is stale, the node is in fail-safe and publishes
fail_safe.build_fail_safe_costmap on every tick: the fail_safe.roi_* cells
lethal (inflated with inflation_radius), every other cell unknown, never
free, stamped with the node clock (there is no current mask stamp). A timer
is needed because on a stale frame or model failure Dev 1 publishes no mask
at all. Geometry does not bypass it: geometry freshness is defined relative
to a fresh mask, so no cloud is fused while in fail-safe (clouds are still
stored for the next fresh mask). The next mask that produces a costmap
leaves fail-safe; its normal output is published as before. The
perception_degraded flag (Dev 1) and the safety hold (Dev 5) are not this
node's. ROI extent/frame semantics, the check period and the owner of this
costmap behaviour are not defined by the architecture: required parameters.

Footprint (`footprint.enabled`): the robot footprint is owned by Dev 5
(dev.md: primary + secondary footprint YAMLs) and not available yet. When
enabled, footprint.vertices / frame_id / padding are required, validated at
startup (footprint.validate_footprint via FootprintInput, pad_footprint) and
carried into every pipeline run (CostmapCoreInputs.footprint,
footprint_padding) -- which returns it in result.footprint and does NOT
apply it to any costmap cell: no project document defines a footprint
rasterisation, and which Dev 5 footprint Dev 3 uses and how it relates to
inflation are not defined either. So the published costmap, geometry fusion
and the stale-mask fail-safe are identical with or without a footprint.
No footprint is ever defaulted; footprint.enabled must be set explicitly.

A mask or cloud that cannot be processed is logged and skipped; nothing is
published for it.

TODO(PortMeta): mask validity. `/segmentation/port_meta` is NOT subscribed:
its format and mask pairing (no header today, arrival order only) are
PENDING with Dev 1. For this first node, receipt of a mask is the validity
signal -- Dev 1 publishes a mask only for a fresh frame with no model
failure -- so every received mask is converted with valid=True. This is a
temporary boundary, not the final PortMeta contract; revisit it when a
timestamped PortMeta is finalised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import rclpy
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from costmap_core.class_to_cost import InvalidSemanticClassError
from costmap_core.contracts import (
    CameraGroundInput,
    CameraIntrinsicsInput,
    ContractError,
    CostmapCoreInputs,
    FootprintInput,
    GridInput,
    OccupancyInput,
    SemanticMaskInput,
    age_s,
    is_fresh,
)
from costmap_core.fail_safe import FailSafeRegion, build_fail_safe_costmap, fail_safe_region_mask
from costmap_core.footprint import FootprintError, pad_footprint, validate_padding
from costmap_core.grid import GridError
from costmap_core.inflation import InflationError
from costmap_core.pipeline import PipelineError, point_cloud_to_occupancy, run_costmap_pipeline
from costmap_core.point_cloud_filter import ObstacleFilterParams
from costmap_core.projection import ProjectionError
from costmap_ros.camera_info_adapter import camera_intrinsics_from_camera_info
from costmap_ros.grid_params import grid_input_from_parameters
from costmap_ros.mask_adapter import semantic_mask_from_image, stamp_to_ns
from costmap_ros.occupancy_grid import occupancy_grid_from_costmap
from costmap_ros.point_cloud_adapter import point_cloud_from_pointcloud2
from costmap_ros.tf_lookup import TfCameraPoseLookup, TfLookupError

NODE_NAME = "costmap_node"

# Output stamp policy is PENDING (CLAUDE.md), so it is a required choice.
# "mask": the mask's header.stamp (image capture time), carried unchanged.
# "now": the node clock when the costmap is built.
OUTPUT_STAMP_SOURCES = ("mask", "now")

# CameraInfo subscription durability: must match the (PENDING) topic's
# publisher, so it is a required choice. See the module docstring.
CAMERA_INFO_DURABILITIES = {
    "volatile": DurabilityPolicy.VOLATILE,
    "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
}


def camera_info_qos(durability: str) -> QoSProfile:
    """CameraInfo subscription QoS: RELIABLE, KEEP_LAST depth 1 (only the
    latest calibration is used), with the configured durability."""
    if durability not in CAMERA_INFO_DURABILITIES:
        raise ContractError(
            f"camera_info_durability must be one of {tuple(CAMERA_INFO_DURABILITIES)}, "
            f"got {durability!r}"
        )
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=CAMERA_INFO_DURABILITIES[durability],
    )

# Dev 1's /perception/depth_cloud publisher is create_publisher(..., 10):
# KEEP_LAST 10, RELIABLE, VOLATILE. The subscription keeps its reliability and
# durability (so it stays compatible) but a history depth of 1, like the mask
# subscription: only the newest cloud matters. A deeper queue lets clouds lag
# behind masks under load until they exceed geometry.max_age_s and are dropped.
DEPTH_CLOUD_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

# Errors that mean "this mask cannot be turned into a costmap": logged and
# skipped. Anything else is a bug and propagates.
SKIPPED_ERRORS = (
    ContractError,
    ProjectionError,
    GridError,
    InflationError,
    PipelineError,
    InvalidSemanticClassError,
    TfLookupError,
)


def _require_topic(value: object, *, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ContractError(f"{name} must be a non-empty str, got {value!r}")
    return value


def _require_non_negative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a real number, got {value!r}")
    if not math.isfinite(value) or value < 0.0:
        raise ContractError(f"{name} must be finite and >= 0, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class GeometryConfig:
    """Geometry side-channel configuration. Every field is required.

    depth_cloud_topic: PointCloud2 topic (Dev 1: /perception/depth_cloud).
    filter_params: obstacle height/range thresholds. PENDING team values.
    max_age_s: largest allowed geometry age relative to the mask,
        mask stamp - cloud stamp, finite and > 0 as contracts.is_fresh
        requires. PENDING team value.
    """

    depth_cloud_topic: str
    filter_params: ObstacleFilterParams
    max_age_s: float

    def __post_init__(self) -> None:
        _require_topic(self.depth_cloud_topic, name="depth_cloud_topic")
        if not isinstance(self.filter_params, ObstacleFilterParams):
            raise ContractError(
                f"filter_params must be an ObstacleFilterParams, got {type(self.filter_params).__name__}"
            )
        max_age = _require_non_negative(self.max_age_s, name="max_age_s")
        if max_age == 0.0:
            raise ContractError("max_age_s must be > 0, got 0.0")
        object.__setattr__(self, "max_age_s", max_age)


@dataclass(frozen=True)
class FailSafeConfig:
    """Stale-mask fail-safe configuration. Every field is required.

    region: ROI made lethal while no fresh mask exists, in target_frame
        metres; must lie inside the grid. PENDING team values.
    check_period_s: fail-safe timer period, finite and > 0. It bounds how
        long the last normal costmap can outlive its mask. PENDING value.
    """

    region: FailSafeRegion
    check_period_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.region, FailSafeRegion):
            raise ContractError(f"region must be a FailSafeRegion, got {type(self.region).__name__}")
        period = _require_non_negative(self.check_period_s, name="check_period_s")
        if period == 0.0:
            raise ContractError("check_period_s must be > 0, got 0.0")
        object.__setattr__(self, "check_period_s", period)


@dataclass(frozen=True)
class FootprintConfig:
    """Robot footprint to carry through the pipeline. Every field is required.

    footprint: validated FootprintInput (convex polygon in its robot frame).
        Values come from Dev 5; none exist yet.
    padding: footprint_padding for run_costmap_pipeline, metres, finite and
        >= 0 (footprint.validate_padding). PENDING value.
    """

    footprint: FootprintInput
    padding: float

    def __post_init__(self) -> None:
        if not isinstance(self.footprint, FootprintInput):
            raise ContractError(
                f"footprint must be a FootprintInput, got {type(self.footprint).__name__}"
            )
        try:
            padding = validate_padding(self.padding)
            pad_footprint(self.footprint.vertices, padding)  # rejects overflowing padding now
        except FootprintError as exc:
            raise ContractError(f"invalid footprint padding: {exc}") from exc
        object.__setattr__(self, "padding", padding)


@dataclass(frozen=True)
class CostmapNodeConfig:
    """Validated node configuration. Every field is required.

    target_frame: TF target frame; also the grid and output frame.
    camera_info_durability: CameraInfo subscription durability, a key of
        CAMERA_INFO_DURABILITIES; must match the chosen topic's publisher.
    mask_max_age_s: largest allowed mask age (now - mask stamp) in seconds,
        finite and > 0 as contracts.is_fresh requires. PENDING team value.
    geometry: GeometryConfig, or None when geometry.enabled is false.
    fail_safe: FailSafeConfig (always required: architecture §8.6).
    footprint: FootprintConfig, or None when footprint.enabled is false.
    """

    mask_topic: str
    camera_info_topic: str
    costmap_topic: str
    target_frame: str
    grid: GridInput
    tf_timeout_s: float
    inflation_radius: float
    camera_info_durability: str
    mask_max_age_s: float
    output_stamp_source: str
    geometry: Optional[GeometryConfig]
    fail_safe: FailSafeConfig
    footprint: Optional[FootprintConfig]

    def __post_init__(self) -> None:
        for name in ("mask_topic", "camera_info_topic", "costmap_topic", "target_frame"):
            _require_topic(getattr(self, name), name=name)
        if not isinstance(self.grid, GridInput):
            raise ContractError(f"grid must be a GridInput, got {type(self.grid).__name__}")
        if self.grid.frame_id != self.target_frame:
            raise ContractError(
                f"grid.frame_id {self.grid.frame_id!r} must equal target_frame {self.target_frame!r}"
            )
        for name in ("tf_timeout_s", "inflation_radius", "mask_max_age_s"):
            object.__setattr__(self, name, _require_non_negative(getattr(self, name), name=name))
        camera_info_qos(self.camera_info_durability)  # validates the choice
        if self.mask_max_age_s == 0.0:
            raise ContractError("mask_max_age_s must be > 0, got 0.0")
        if self.output_stamp_source not in OUTPUT_STAMP_SOURCES:
            raise ContractError(
                f"output_stamp_source must be one of {OUTPUT_STAMP_SOURCES}, "
                f"got {self.output_stamp_source!r}"
            )
        if self.geometry is not None and not isinstance(self.geometry, GeometryConfig):
            raise ContractError(
                f"geometry must be a GeometryConfig or None, got {type(self.geometry).__name__}"
            )
        if not isinstance(self.fail_safe, FailSafeConfig):
            raise ContractError(
                f"fail_safe must be a FailSafeConfig, got {type(self.fail_safe).__name__}"
            )
        fail_safe_region_mask(self.fail_safe.region, self.grid.geometry)  # inside the grid
        if self.footprint is not None and not isinstance(self.footprint, FootprintConfig):
            raise ContractError(
                f"footprint must be a FootprintConfig or None, got {type(self.footprint).__name__}"
            )


# (ROS parameter name, type). All required, no defaults.
PARAMETERS = (
    ("mask_topic", Parameter.Type.STRING),
    ("camera_info_topic", Parameter.Type.STRING),
    ("costmap_topic", Parameter.Type.STRING),
    ("target_frame", Parameter.Type.STRING),
    ("grid.resolution", Parameter.Type.DOUBLE),
    ("grid.origin_x", Parameter.Type.DOUBLE),
    ("grid.origin_y", Parameter.Type.DOUBLE),
    ("grid.width", Parameter.Type.INTEGER),
    ("grid.height", Parameter.Type.INTEGER),
    ("tf_timeout_s", Parameter.Type.DOUBLE),
    ("inflation_radius", Parameter.Type.DOUBLE),
    ("camera_info_durability", Parameter.Type.STRING),
    ("mask_max_age_s", Parameter.Type.DOUBLE),
    ("output_stamp_source", Parameter.Type.STRING),
    ("geometry.enabled", Parameter.Type.BOOL),
    ("fail_safe.roi_min_x", Parameter.Type.DOUBLE),
    ("fail_safe.roi_max_x", Parameter.Type.DOUBLE),
    ("fail_safe.roi_min_y", Parameter.Type.DOUBLE),
    ("fail_safe.roi_max_y", Parameter.Type.DOUBLE),
    ("fail_safe.check_period_s", Parameter.Type.DOUBLE),
    ("footprint.enabled", Parameter.Type.BOOL),
)

# Required if and only if geometry.enabled is true. No defaults.
GEOMETRY_PARAMETERS = (
    ("geometry.depth_cloud_topic", Parameter.Type.STRING),
    ("geometry.min_height", Parameter.Type.DOUBLE),
    ("geometry.max_height", Parameter.Type.DOUBLE),
    ("geometry.min_range", Parameter.Type.DOUBLE),
    ("geometry.max_range", Parameter.Type.DOUBLE),
    ("geometry.max_age_s", Parameter.Type.DOUBLE),
)

# Required if and only if footprint.enabled is true. No defaults.
# footprint.vertices is flat: [x0, y0, x1, y1, ...], metres, in footprint.frame_id.
FOOTPRINT_PARAMETERS = (
    ("footprint.vertices", Parameter.Type.DOUBLE_ARRAY),
    ("footprint.frame_id", Parameter.Type.STRING),
    ("footprint.padding", Parameter.Type.DOUBLE),
)


def _require_switch(values: dict, name: str) -> bool:
    value = values.get(name)
    if name in values and not isinstance(value, bool):
        raise ContractError(f"{name} must be a bool, got {value!r}")
    return bool(value)


def config_from_parameters(values: dict) -> CostmapNodeConfig:
    """Build the config from {parameter name: value}.

    Names as in PARAMETERS, plus GEOMETRY_PARAMETERS iff geometry.enabled is
    true and FOOTPRINT_PARAMETERS iff footprint.enabled is true. Parameters
    of a disabled group are rejected as unknown rather than silently ignored.
    """
    enabled = _require_switch(values, "geometry.enabled")
    footprint_enabled = _require_switch(values, "footprint.enabled")
    expected = (
        list(PARAMETERS)
        + (list(GEOMETRY_PARAMETERS) if enabled else [])
        + (list(FOOTPRINT_PARAMETERS) if footprint_enabled else [])
    )
    missing = [name for name, _ in expected if name not in values]
    unknown = sorted(set(values) - {name for name, _ in expected})
    if missing or unknown:
        raise ContractError(f"node parameters: missing {missing}, unknown {unknown}")
    grid = grid_input_from_parameters(
        {
            "resolution": values["grid.resolution"],
            "origin_x": values["grid.origin_x"],
            "origin_y": values["grid.origin_y"],
            "width": values["grid.width"],
            "height": values["grid.height"],
            "frame_id": values["target_frame"],
        }
    )
    return CostmapNodeConfig(
        mask_topic=values["mask_topic"],
        camera_info_topic=values["camera_info_topic"],
        costmap_topic=values["costmap_topic"],
        target_frame=values["target_frame"],
        grid=grid,
        tf_timeout_s=values["tf_timeout_s"],
        inflation_radius=values["inflation_radius"],
        camera_info_durability=values["camera_info_durability"],
        mask_max_age_s=values["mask_max_age_s"],
        output_stamp_source=values["output_stamp_source"],
        geometry=_geometry_config(values) if enabled else None,
        fail_safe=FailSafeConfig(
            region=FailSafeRegion(
                min_x=values["fail_safe.roi_min_x"],
                max_x=values["fail_safe.roi_max_x"],
                min_y=values["fail_safe.roi_min_y"],
                max_y=values["fail_safe.roi_max_y"],
            ),
            check_period_s=values["fail_safe.check_period_s"],
        ),
        footprint=_footprint_config(values) if footprint_enabled else None,
    )


def _footprint_config(values: dict) -> FootprintConfig:
    flat = values["footprint.vertices"]
    if isinstance(flat, (str, bytes)) or not isinstance(flat, (list, tuple)):
        raise ContractError(f"footprint.vertices must be a list of numbers, got {flat!r}")
    if len(flat) % 2 != 0:
        raise ContractError(
            f"footprint.vertices must be flat [x0, y0, x1, y1, ...] (even length), got {len(flat)} values"
        )
    vertices = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    return FootprintConfig(
        footprint=FootprintInput(vertices=vertices, frame_id=values["footprint.frame_id"]),
        padding=values["footprint.padding"],
    )


def _geometry_config(values: dict) -> GeometryConfig:
    return GeometryConfig(
        depth_cloud_topic=values["geometry.depth_cloud_topic"],
        filter_params=ObstacleFilterParams(
            min_height=values["geometry.min_height"],
            max_height=values["geometry.max_height"],
            min_range=values["geometry.min_range"],
            max_range=values["geometry.max_range"],
        ),
        max_age_s=values["geometry.max_age_s"],
    )


@dataclass(frozen=True, eq=False)
class _GeometryObservation:
    """One rasterised cloud and the camera frame it was observed from."""

    occupancy: OccupancyInput
    camera_frame_id: str


@dataclass(eq=False)
class _ProcessedMask:
    """Inputs of the last mask that produced a costmap, kept so a cloud that
    arrives later can be fused with that same mask."""

    stamp: object  # builtin_interfaces/Time from the mask header
    mask: SemanticMaskInput
    intrinsics: CameraIntrinsicsInput
    camera_ground: CameraGroundInput
    geometry_stamp_ns: Optional[int]  # stamp of the fused observation, None if none


class CostmapNodeCore:
    """Per-message logic of the node, free of rclpy.Node so it can be tested
    with plain messages, a fake TF lookup and a fake logger."""

    def __init__(self, config: CostmapNodeConfig, tf_lookup: TfCameraPoseLookup, logger) -> None:
        self._config = config
        self._tf_lookup = tf_lookup
        self._logger = logger
        self._intrinsics: Optional[CameraIntrinsicsInput] = None
        self._geometry: Optional[_GeometryObservation] = None
        self._last_mask: Optional[_ProcessedMask] = None
        self._fail_safe_active = False
        # Pure function of the config: built once, identical on every tick.
        self._fail_safe_costmap = build_fail_safe_costmap(
            config.fail_safe.region, config.grid.geometry, inflation_radius=config.inflation_radius
        )

    @property
    def fail_safe_active(self) -> bool:
        return self._fail_safe_active

    def handle_fail_safe_check(self, *, now_stamp):
        """Timer tick: return the fail-safe OccupancyGrid if no fresh mask
        has produced a costmap, else None (normal output is mask-driven).

        Fresh means is_fresh(last mask stamp, now, mask_max_age_s). No mask
        yet, a stale one, or one stamped after now all mean fail-safe.
        """
        now_ns = stamp_to_ns(now_stamp)
        if now_ns <= 0:
            self._logger.warning("fail-safe check skipped: node clock is 0 (no time yet)")
            return None
        last = self._last_mask
        if last is not None and is_fresh(last.mask.stamp_ns, now_ns, self._config.mask_max_age_s):
            return None
        if not self._fail_safe_active:
            reason = "no mask has produced a costmap" if last is None else (
                f"last mask age {age_s(last.mask.stamp_ns, now_ns):.6f} s outside 0 .. "
                f"mask_max_age_s = {self._config.mask_max_age_s}"
            )
            self._logger.warning(f"entering stale-mask fail-safe: {reason}")
        self._fail_safe_active = True
        return occupancy_grid_from_costmap(self._fail_safe_costmap, self._config.grid, stamp=now_stamp)

    @property
    def has_camera_info(self) -> bool:
        return self._intrinsics is not None

    def handle_camera_info(self, msg) -> None:
        """Keep the latest valid CameraInfo. An invalid one clears the stored
        one, so masks are skipped until a valid CameraInfo arrives again.
        header.stamp is not used (see the module docstring)."""
        try:
            intrinsics = camera_intrinsics_from_camera_info(msg)
        except SKIPPED_ERRORS as exc:
            self._intrinsics = None
            self._logger.warning(f"CameraInfo rejected, cleared stored intrinsics: {exc}")
            return
        self._intrinsics = intrinsics

    def handle_mask(self, msg, *, now_stamp):
        """Return an OccupancyGrid for this mask, or None if it was skipped.

        now_stamp: builtin_interfaces/Time of the node clock. It is the "now"
            of the freshness check and, when output_stamp_source == "now",
            the output stamp.
        """
        config = self._config
        if self._intrinsics is None:
            self._logger.warning("mask skipped: no valid CameraInfo received yet")
            return None
        try:
            # TODO(PortMeta): receipt of a mask is the validity signal until a
            # timestamped PortMeta pairing is agreed (see module docstring).
            mask = semantic_mask_from_image(msg, valid=True)

            now_ns = stamp_to_ns(now_stamp)
            if not is_fresh(mask.stamp_ns, now_ns, config.mask_max_age_s):
                age = age_s(mask.stamp_ns, now_ns)
                reason = "future-stamped" if age < 0.0 else "stale"
                self._logger.warning(
                    f"mask skipped: {reason}, age = {age:.6f} s "
                    f"(allowed 0 .. mask_max_age_s = {config.mask_max_age_s})"
                )
                return None

            # CameraInfo frame and size are checked against the mask by
            # CostmapCoreInputs (ContractError -> skipped).
            camera_ground = self._tf_lookup.lookup(
                target_frame=config.target_frame,
                source_frame=mask.frame_id,
                stamp_ns=mask.stamp_ns,
            )
            geometry = self._geometry_for(mask)
            result = self._run_pipeline(
                mask, self._intrinsics, camera_ground,
                None if geometry is None else geometry.occupancy,
            )
        except TfLookupError as exc:
            self._logger.warning(f"mask skipped: TF lookup failed ({exc.kind}): {exc}")
            return None
        except SKIPPED_ERRORS as exc:
            self._logger.warning(f"mask skipped: {type(exc).__name__}: {exc}")
            return None

        if self._fail_safe_active:
            self._logger.warning("leaving stale-mask fail-safe: fresh mask processed")
            self._fail_safe_active = False
        self._last_mask = _ProcessedMask(
            stamp=msg.header.stamp,
            mask=mask,
            intrinsics=self._intrinsics,
            camera_ground=camera_ground,
            geometry_stamp_ns=None if geometry is None else geometry.occupancy.stamp_ns,
        )
        stamp = msg.header.stamp if config.output_stamp_source == "mask" else now_stamp
        return occupancy_grid_from_costmap(result.fused, config.grid, stamp=stamp)

    def handle_cloud(self, msg, *, now_stamp):
        """Rasterise one depth cloud; return a re-fused OccupancyGrid for the
        last mask if the cloud pairs with it better than before, else None.

        now_stamp: node clock sample, used to re-check that mask's freshness
            and as the output stamp when output_stamp_source == "now".
        """
        config = self._config
        if config.geometry is None:
            raise ContractError("handle_cloud called with geometry disabled")
        try:
            cloud = point_cloud_from_pointcloud2(msg)
            now_ns = stamp_to_ns(now_stamp)
            if cloud.stamp_ns > now_ns:
                # Same convention as masks: a stamp after the node clock is not
                # a real observation yet. Stored, it would shadow every later
                # valid cloud (older stamps) until the clock caught up.
                self._logger.warning(
                    f"cloud skipped: future-stamped ({cloud.stamp_ns} ns > node clock {now_ns} ns)"
                )
                return None
            # Always at the cloud's own stamp, never "latest"; no fallback.
            camera_ground = self._tf_lookup.lookup(
                target_frame=config.target_frame,
                source_frame=cloud.frame_id,
                stamp_ns=cloud.stamp_ns,
            )
            occupancy = point_cloud_to_occupancy(
                cloud, camera_ground, config.grid, config.geometry.filter_params
            )
        except TfLookupError as exc:
            self._logger.warning(f"cloud skipped: TF lookup failed ({exc.kind}): {exc}")
            return None
        except SKIPPED_ERRORS as exc:
            self._logger.warning(f"cloud skipped: {type(exc).__name__}: {exc}")
            return None

        observation = _GeometryObservation(occupancy=occupancy, camera_frame_id=cloud.frame_id)
        stored = self._geometry
        # Keep the newest-stamped observation; one stamped after the node clock
        # (e.g. from before a backwards clock jump) is always replaced.
        if stored is None or occupancy.stamp_ns >= stored.occupancy.stamp_ns or stored.occupancy.stamp_ns > now_ns:
            self._geometry = observation
        return self._fuse_into_last_mask(observation, now_stamp)

    def _run_pipeline(self, mask, intrinsics, camera_ground, occupancy):
        """One run_costmap_pipeline call with the configured grid, inflation
        and (if footprint.enabled) footprint + footprint_padding."""
        config = self._config
        footprint = config.footprint
        inputs = CostmapCoreInputs(
            mask=mask,
            intrinsics=intrinsics,
            camera_ground=camera_ground,
            grid=config.grid,
            occupancy=occupancy,
            footprint=None if footprint is None else footprint.footprint,
        )
        if footprint is None:
            return run_costmap_pipeline(inputs, inflation_radius=config.inflation_radius)
        return run_costmap_pipeline(
            inputs, inflation_radius=config.inflation_radius, footprint_padding=footprint.padding
        )

    def _geometry_for(self, mask: SemanticMaskInput) -> Optional[_GeometryObservation]:
        """The stored observation if it pairs with this (fresh) mask, else None.
        A stored observation that is stale for this mask is discarded."""
        geometry = self._geometry
        if geometry is None:
            return None
        # Same arithmetic as is_fresh, so "stale" and "not fresh" agree exactly.
        if age_s(geometry.occupancy.stamp_ns, mask.stamp_ns) > self._config.geometry.max_age_s:
            self._geometry = None  # stale: never reused for this or a later mask
            return None
        return geometry if self._pairs(geometry, mask) else None

    def _pairs(self, geometry: _GeometryObservation, mask: SemanticMaskInput) -> bool:
        """Same camera frame and 0 <= mask stamp - cloud stamp <= geometry.max_age_s."""
        if geometry.camera_frame_id != mask.frame_id:
            self._logger.warning(
                f"geometry not used: cloud frame {geometry.camera_frame_id!r} "
                f"!= mask frame {mask.frame_id!r}"
            )
            return False
        return is_fresh(geometry.occupancy.stamp_ns, mask.stamp_ns, self._config.geometry.max_age_s)

    def _fuse_into_last_mask(self, geometry: _GeometryObservation, now_stamp):
        last = self._last_mask
        if last is None or not self._pairs(geometry, last.mask):
            return None
        age_ns = last.mask.stamp_ns - geometry.occupancy.stamp_ns  # >= 0 once paired
        if last.geometry_stamp_ns is not None and last.mask.stamp_ns - last.geometry_stamp_ns <= age_ns:
            return None  # the published costmap already used an equally recent cloud
        config = self._config
        try:
            if not is_fresh(last.mask.stamp_ns, stamp_to_ns(now_stamp), config.mask_max_age_s):
                return None  # too late to republish that mask
            result = self._run_pipeline(
                last.mask, last.intrinsics, last.camera_ground, geometry.occupancy
            )
        except SKIPPED_ERRORS as exc:
            self._logger.warning(f"cloud not fused: {type(exc).__name__}: {exc}")
            return None
        last.geometry_stamp_ns = geometry.occupancy.stamp_ns
        stamp = last.stamp if config.output_stamp_source == "mask" else now_stamp
        return occupancy_grid_from_costmap(result.fused, config.grid, stamp=stamp)


class CostmapNode(Node):
    """rclpy wiring only: parameters, TF buffer/listener, subscriptions, publisher."""

    def __init__(self, **kwargs) -> None:
        super().__init__(NODE_NAME, **kwargs)
        for name, param_type in PARAMETERS:
            self.declare_parameter(name, param_type)
        declared = list(PARAMETERS)
        # Raises ParameterUninitializedException for any unset parameter.
        for switch, group in (("geometry.enabled", GEOMETRY_PARAMETERS),
                              ("footprint.enabled", FOOTPRINT_PARAMETERS)):
            if self.get_parameter(switch).value is True:
                for name, param_type in group:
                    self.declare_parameter(name, param_type)
                declared += group
        self.config = config_from_parameters(
            {name: self.get_parameter(name).value for name, _ in declared}
        )
        config = self.config
        if config.footprint is None:
            self.get_logger().info(
                "footprint.enabled is false: no robot footprint is carried (Dev 5 footprint pending)"
            )

        # A lookup with a timeout blocks the mask callback (tf2 polls), so TF
        # must be received on another thread: the listener's callbacks are in
        # its own ReentrantCallbackGroup and main() spins a
        # MultiThreadedExecutor. No private listener thread (spin_thread=True
        # would add this whole node to a second executor).
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self, spin_thread=False)
        tf_lookup = TfCameraPoseLookup(self._tf_buffer, timeout_s=config.tf_timeout_s)
        self._core = CostmapNodeCore(config, tf_lookup, self.get_logger())

        # Mask, CameraInfo, cloud and fail-safe timer callbacks never run
        # concurrently, so the stored CameraInfo, geometry, last mask and
        # fail-safe state need no lock.
        inputs_group = MutuallyExclusiveCallbackGroup()
        self._pub_costmap = self.create_publisher(OccupancyGrid, config.costmap_topic, 10)
        self._sub_info = self.create_subscription(
            CameraInfo, config.camera_info_topic, self._core.handle_camera_info,
            camera_info_qos(config.camera_info_durability), callback_group=inputs_group,
        )
        # Depth 1: only the newest mask matters; older queued ones would be stale.
        self._sub_mask = self.create_subscription(
            Image, config.mask_topic, self._on_mask, 1, callback_group=inputs_group
        )
        self._fail_safe_timer = self.create_timer(
            config.fail_safe.check_period_s, self._on_fail_safe_check, callback_group=inputs_group
        )
        self._sub_cloud = None
        if config.geometry is not None:
            self._sub_cloud = self.create_subscription(
                PointCloud2, config.geometry.depth_cloud_topic, self._on_cloud,
                DEPTH_CLOUD_QOS, callback_group=inputs_group,
            )

    def _on_mask(self, msg: Image) -> None:
        grid = self._core.handle_mask(msg, now_stamp=self.get_clock().now().to_msg())
        if grid is not None:
            self._pub_costmap.publish(grid)

    def _on_fail_safe_check(self) -> None:
        grid = self._core.handle_fail_safe_check(now_stamp=self.get_clock().now().to_msg())
        if grid is not None:
            self._pub_costmap.publish(grid)

    def _on_cloud(self, msg: PointCloud2) -> None:
        grid = self._core.handle_cloud(msg, now_stamp=self.get_clock().now().to_msg())
        if grid is not None:
            self._pub_costmap.publish(grid)

    def destroy_node(self) -> None:
        # May run after a failed __init__ (e.g. a missing parameter).
        listener = getattr(self, "_tf_listener", None)
        if listener is not None:
            listener.unregister()
            self._tf_listener = None
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CostmapNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
