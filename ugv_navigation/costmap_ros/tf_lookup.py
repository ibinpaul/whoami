"""tf2_ros.Buffer lookup -> CameraGroundInput (CameraPose + frames + stamp).

One lookup at an explicit timestamp with an explicit timeout. No frame
name, timeout or stamp tolerance is defaulted here: the TF tree (Dev 2 /
Dev 5), frame names and Dev 3's stale/extrapolation policy are all PENDING.

point_cloud_in_frame applies such a lookup, taken at the cloud's own stamp,
to a PointCloudInput via costmap_core.point_cloud_transform.

The Buffer is supplied by the caller (the future node owns it and its
TransformListener). This module does not subscribe, publish or spin.

Failures are split so the future node can apply its own policy:
- TfLookupError: tf2 could not produce a transform (unknown frame, no
  connection, extrapolation into past/future, timeout, invalid argument).
  The tf2 exception is kept as `cause` and __cause__; `kind` is its class
  name. Never retried or replaced by a fallback pose here.
- ProjectionError / ContractError from tf_adapter: tf2 returned a transform
  but it is not a valid camera pose (e.g. camera not above the ground plane).
"""

from __future__ import annotations

import math

import tf2_ros
from rclpy.duration import Duration
from rclpy.time import Time

from costmap_core.contracts import (
    CameraGroundInput,
    ContractError,
    PointCloudInput,
    _require_frame_id,
    _require_stamp,
    _require_type,
)
from costmap_core.point_cloud_transform import transform_point_cloud
from costmap_ros.tf_adapter import camera_ground_input_from_transform


class TfLookupError(RuntimeError):
    """tf2 could not provide target_frame <- source_frame at stamp_ns."""

    def __init__(self, *, target_frame: str, source_frame: str, stamp_ns: int,
                 cause: tf2_ros.TransformException) -> None:
        self.target_frame = target_frame
        self.source_frame = source_frame
        self.stamp_ns = stamp_ns
        self.cause = cause
        self.kind = type(cause).__name__
        super().__init__(
            f"TF lookup {target_frame!r} <- {source_frame!r} at {stamp_ns} ns failed "
            f"({self.kind}): {cause}"
        )


class TfCameraPoseLookup:
    """Looks up the camera optical frame in the ground frame via a tf2 Buffer.

    buffer: a tf2_ros.Buffer (or anything with the same lookup_transform).
    timeout_s: how long lookup_transform may wait, in seconds (>= 0,
        finite). Required; the real value is PENDING.
    """

    def __init__(self, buffer: tf2_ros.Buffer, *, timeout_s: float) -> None:
        if not callable(getattr(buffer, "lookup_transform", None)):
            raise ContractError(
                f"buffer must provide lookup_transform, got {type(buffer).__name__}"
            )
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ContractError(f"timeout_s must be a real number, got {timeout_s!r}")
        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ContractError(f"timeout_s must be finite and >= 0, got {timeout_s!r}")
        self._buffer = buffer
        self._timeout = Duration(nanoseconds=round(float(timeout_s) * 1e9))

    @property
    def timeout(self) -> Duration:
        return self._timeout

    def lookup(self, *, target_frame: str, source_frame: str, stamp_ns: int) -> CameraGroundInput:
        """Transform of source_frame (camera optical) into target_frame (ground) at stamp_ns.

        stamp_ns must be > 0: tf2's "time 0 = latest available" is a
        staleness policy and is not chosen here. The returned
        CameraGroundInput carries the frames and stamp of the transform
        tf2 returned, unchanged.

        Raises TfLookupError for any tf2 TransformException, ContractError
        if tf2 answers for different frames than requested, and the
        tf_adapter errors for an invalid pose.
        """
        _require_frame_id(target_frame, name="target_frame")
        _require_frame_id(source_frame, name="source_frame")
        stamp_ns = _require_stamp(stamp_ns, name="stamp_ns")

        try:
            transform = self._buffer.lookup_transform(
                target_frame, source_frame, Time(nanoseconds=stamp_ns), self._timeout
            )
        except tf2_ros.TransformException as exc:
            raise TfLookupError(
                target_frame=target_frame, source_frame=source_frame, stamp_ns=stamp_ns, cause=exc
            ) from exc

        result = camera_ground_input_from_transform(transform)
        if (result.ground_frame_id, result.camera_frame_id) != (target_frame, source_frame):
            raise ContractError(
                f"TF returned {result.ground_frame_id!r} <- {result.camera_frame_id!r}, "
                f"requested {target_frame!r} <- {source_frame!r}"
            )
        return result


def point_cloud_in_frame(
    lookup: TfCameraPoseLookup, cloud: PointCloudInput, *, target_frame: str
) -> PointCloudInput:
    """Look up cloud.frame_id in target_frame at cloud.stamp_ns and transform the cloud.

    The lookup time is always the cloud's own stamp, never "latest". The
    target frame has no default. Errors from TfCameraPoseLookup.lookup
    (TfLookupError, ContractError, ProjectionError) propagate unchanged: no
    fallback pose is ever used.
    """
    _require_type(lookup, TfCameraPoseLookup, name="lookup")
    _require_type(cloud, PointCloudInput, name="cloud")
    camera_ground = lookup.lookup(
        target_frame=target_frame, source_frame=cloud.frame_id, stamp_ns=cloud.stamp_ns
    )
    return transform_point_cloud(cloud, camera_ground)
