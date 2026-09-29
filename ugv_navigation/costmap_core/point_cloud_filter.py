"""Obstacle-candidate selection for a PointCloudInput already in the ground frame.

Input: a cloud transformed by point_cloud_transform.transform_point_cloud
plus the CameraGroundInput used for that transform. The ground frame
follows CameraPose's model: z up, flat ground at z = 0, so a point's z is
its height above the ground and the camera origin is the pose translation.

A point is kept as obstacle evidence iff (Nav2 ObstacleLayer/VoxelLayer
comparison convention, the geometry layer named in architecture.md §6):

    min_height <= z <= max_height
    min_range  <= |p - camera origin| < max_range      (3D distance)

Everything else is dropped. A dropped point (ground, too high, too near,
too far) is NOT free space: it simply is not obstacle evidence. An empty
result means this observation identified no obstacle points, not that any
cell is free.

Not implemented, because the cloud alone cannot support it:
- slope: the cloud is unorganised (no pixel adjacency, no normals), so a
  local surface slope needs a neighbourhood definition that is not agreed;
- "traversable surface" rejection: the cloud carries no semantic class.

All thresholds are required (ObstacleFilterParams has no defaults); the
real values are PENDING.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from costmap_core.contracts import CameraGroundInput, ContractError, PointCloudInput, _require_type
from costmap_core.projection import CameraPose, camera_ground_geometry_to_pose


def _require_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ContractError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ContractError(f"{name} must be finite, got {value!r}")
    return value


@dataclass(frozen=True)
class ObstacleFilterParams:
    """Obstacle-candidate thresholds, metres. No defaults: real values are PENDING.

    min_height, max_height: inclusive band of ground-frame z kept as
        obstacle evidence. min_height < max_height. Heights below the band
        (the ground itself) and above it (e.g. overhangs) are dropped.
    min_range, max_range: 3D distance from the camera origin,
        min_range <= d < max_range, 0 <= min_range < max_range.
    """

    min_height: float
    max_height: float
    min_range: float
    max_range: float

    def __post_init__(self) -> None:
        for name in ("min_height", "max_height", "min_range", "max_range"):
            object.__setattr__(self, name, _require_finite_real(getattr(self, name), name=name))
        if not self.min_height < self.max_height:
            raise ContractError(
                f"min_height ({self.min_height}) must be < max_height ({self.max_height})"
            )
        if self.min_range < 0.0:
            raise ContractError(f"min_range must be >= 0, got {self.min_range}")
        if not self.min_range < self.max_range:
            raise ContractError(
                f"min_range ({self.min_range}) must be < max_range ({self.max_range})"
            )


def filter_obstacle_points(
    cloud: PointCloudInput,
    camera_ground: CameraGroundInput,
    params: ObstacleFilterParams,
) -> PointCloudInput:
    """Return the obstacle-candidate subset of cloud (same stamp and frame).

    cloud must be in camera_ground.ground_frame_id (i.e. already
    transformed); an untransformed optical-frame cloud is rejected, since
    its z is depth, not height. Kept points stay in their input order.
    Vectorised; the input is not modified and the result is a new
    read-only PointCloudInput.

    Raises ContractError for wrong types or a frame mismatch.
    """
    _require_type(cloud, PointCloudInput, name="cloud")
    _require_type(camera_ground, CameraGroundInput, name="camera_ground")
    _require_type(params, ObstacleFilterParams, name="params")
    if cloud.frame_id != camera_ground.ground_frame_id:
        raise ContractError(
            f"cloud.frame_id {cloud.frame_id!r} is not the ground frame "
            f"{camera_ground.ground_frame_id!r}; transform the cloud first"
        )

    geometry = camera_ground.geometry
    pose = geometry if isinstance(geometry, CameraPose) else camera_ground_geometry_to_pose(geometry)

    points = cloud.points
    z = points[:, 2]
    distance = np.linalg.norm(points - pose.translation, axis=1)
    keep = (
        (z >= params.min_height)
        & (z <= params.max_height)
        & (distance >= params.min_range)
        & (distance < params.max_range)
    )
    return PointCloudInput(points=points[keep], stamp_ns=cloud.stamp_ns, frame_id=cloud.frame_id)
