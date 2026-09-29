"""sensor_msgs/CameraInfo -> CameraIntrinsicsInput.

Only K, width, height and header.frame_id are used. Distortion (D,
distortion_model) is not read: the core is pure pinhole and whether the
mask is computed on a rectified image is PENDING. CameraIntrinsicsInput has
no stamp field, so header.stamp is not carried: a CameraInfo keeps its own
(calibration) stamp, which says nothing about the mask it is used with.

Placeholder K values are rejected like Dev 1 does ("reject missing/fake K",
dev.md Dev 1 task 1; ugv_perception ingest/decode.py): an all-zero K (the
ROS "uncalibrated" value) and the identity matrix.

Which CameraInfo topic feeds this adapter (`/segmentation/camera_info` or
the driver topic) is PENDING; the conversion does not depend on it.
"""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import CameraInfo

from costmap_core.contracts import CameraIntrinsicsInput, ContractError
from costmap_core.projection import CameraIntrinsics


def camera_intrinsics_from_camera_info(msg: CameraInfo) -> CameraIntrinsicsInput:
    """Convert CameraInfo into CameraIntrinsicsInput.

    K is row-major [fx, s, cx, 0, fy, cy, 0, 0, 1]. The core has no skew term,
    so a K that is not of that zero-skew form is rejected rather than
    silently truncated.

    Raises ContractError for a wrong message type, malformed or placeholder
    (all-zero or identity) K, and lets the
    core's own errors propagate (ProjectionError for fx/fy <= 0, ContractError
    for non-positive size or empty frame_id).
    """
    if not isinstance(msg, CameraInfo):
        raise ContractError(f"expected sensor_msgs/msg/CameraInfo, got {type(msg).__name__}")

    k = np.asarray(msg.k, dtype=np.float64)
    if k.shape != (9,):
        raise ContractError(f"CameraInfo K must have 9 elements, got shape {k.shape}")
    if not np.all(np.isfinite(k)):
        raise ContractError(f"CameraInfo K must be finite, got {k.tolist()}")
    if not np.any(k):
        raise ContractError("CameraInfo K is all zeros (uncalibrated camera)")
    if np.array_equal(k, np.eye(3).ravel()):
        raise ContractError("CameraInfo K is the identity matrix (placeholder, not a calibration)")
    fx, skew, cx, k10, fy, cy, k20, k21, k22 = k.tolist()
    if skew != 0.0 or k10 != 0.0 or (k20, k21, k22) != (0.0, 0.0, 1.0):
        raise ContractError(
            f"CameraInfo K is not a zero-skew pinhole matrix: {k.tolist()}"
        )

    return CameraIntrinsicsInput(
        intrinsics=CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy),
        image_width=int(msg.width),
        image_height=int(msg.height),
        frame_id=msg.header.frame_id,
    )
