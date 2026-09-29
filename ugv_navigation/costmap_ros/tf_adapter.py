"""geometry_msgs/TransformStamped -> CameraPose / CameraGroundInput.

Pure message conversion. This module does not look up TF, own a buffer,
subscribe or publish, and applies no timeout or stamp-matching policy: the
TF tree (Dev 2 / Dev 5), frame names and allowed stamp mismatch are all
PENDING. Whoever eventually performs the lookup passes the resulting
TransformStamped here.

ROS convention: a TransformStamped with header.frame_id = T and
child_frame_id = C maps points in C into T. For the costmap the lookup is
target = ground/costmap frame, source = camera optical frame, which is
exactly what CameraPose expects (camera optical frame in the ground frame).

Quaternion maths lives only in costmap_core.projection.camera_pose_from_quaternion.
"""

from __future__ import annotations

from geometry_msgs.msg import TransformStamped

from costmap_core.contracts import CameraGroundInput, ContractError
from costmap_core.projection import CameraPose, camera_pose_from_quaternion
from costmap_ros.mask_adapter import stamp_to_ns


def camera_pose_from_transform(msg: TransformStamped) -> CameraPose:
    """Convert transform.translation / transform.rotation into a CameraPose.

    Header and child frame are not inspected. Raises ContractError for a
    wrong message type and lets the core's ProjectionError propagate for
    non-finite values, a non-unit quaternion or translation z <= 0.
    """
    if not isinstance(msg, TransformStamped):
        raise ContractError(
            f"expected geometry_msgs/msg/TransformStamped, got {type(msg).__name__}"
        )
    t = msg.transform.translation
    q = msg.transform.rotation
    return camera_pose_from_quaternion(
        qx=q.x, qy=q.y, qz=q.z, qw=q.w, tx=t.x, ty=t.y, tz=t.z
    )


def camera_ground_input_from_transform(msg: TransformStamped) -> CameraGroundInput:
    """Convert a TransformStamped into CameraGroundInput, keeping its frames.

    ground_frame_id = header.frame_id, camera_frame_id = child_frame_id and
    stamp_ns = header.stamp, all taken unchanged. No frame name is expected
    or checked here; CostmapCoreInputs checks consistency with the mask and
    grid. Raises ContractError for an empty frame id or non-positive stamp.
    """
    return CameraGroundInput(
        geometry=camera_pose_from_transform(msg),
        camera_frame_id=msg.child_frame_id,
        ground_frame_id=msg.header.frame_id,
        stamp_ns=stamp_to_ns(msg.header.stamp),
    )
