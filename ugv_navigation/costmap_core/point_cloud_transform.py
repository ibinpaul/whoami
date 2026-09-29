"""Rigid transform of a PointCloudInput from the camera optical frame into the ground frame.

Reuses the existing pose representation: CameraGroundInput (from
tf_lookup / tf_adapter, or built directly) carries a CameraPose, or a
CameraGroundGeometry converted by projection.camera_ground_geometry_to_pose.
No second transform type exists. CameraPose's rotation maps camera optical
axes (x-right, y-down, z-forward) into the ground frame, so optical
coordinates are used exactly as Dev 1 publishes them; nothing is re-axed here.

Frame rule: cloud.frame_id must equal camera_ground.camera_frame_id. The
result is in camera_ground.ground_frame_id.

Stamp rule: the result keeps cloud.stamp_ns (the observation time). The
pose's own stamp is not compared with it: the allowed cloud/TF stamp
mismatch is PENDING. costmap_ros.tf_lookup.point_cloud_in_frame looks the
pose up at the cloud stamp.

No filtering, thresholds or rasterisation happen here.
"""

from __future__ import annotations

import numpy as np

from costmap_core.contracts import CameraGroundInput, ContractError, PointCloudInput, _require_type
from costmap_core.projection import CameraPose, camera_ground_geometry_to_pose


def transform_point_cloud(
    cloud: PointCloudInput, camera_ground: CameraGroundInput
) -> PointCloudInput:
    """Return cloud's points expressed in camera_ground.ground_frame_id.

    p_ground = R @ p_camera + t for every point (vectorised), with R, t from
    the CameraPose. Point order and count are preserved; an empty cloud
    stays empty. The output points are a read-only float64 array (the pose
    is float64); the input cloud is not modified.

    Raises ContractError for wrong input types or a camera frame mismatch.
    Pose validity (finite, orthonormal, det +1) is already enforced by
    CameraPose when camera_ground is built.
    """
    _require_type(cloud, PointCloudInput, name="cloud")
    _require_type(camera_ground, CameraGroundInput, name="camera_ground")
    if cloud.frame_id != camera_ground.camera_frame_id:
        raise ContractError(
            f"cloud.frame_id {cloud.frame_id!r} does not match "
            f"camera_ground.camera_frame_id {camera_ground.camera_frame_id!r}"
        )

    geometry = camera_ground.geometry
    pose = geometry if isinstance(geometry, CameraPose) else camera_ground_geometry_to_pose(geometry)

    points = cloud.points.astype(np.float64) @ pose.rotation.T + pose.translation
    return PointCloudInput(
        points=points,
        stamp_ns=cloud.stamp_ns,
        frame_id=camera_ground.ground_frame_id,
    )
