"""sensor_msgs/PointCloud2 (Dev 1 `/perception/depth_cloud`) -> PointCloudInput.

Dev 1's cloud layout (node/cloud.py, subarch8 §5): unorganised (height 1),
exactly three fields x, y, z, each FLOAT32 with count 1 at offsets 0, 4, 8,
point_step 12, row_step 12 * width, metres in the camera optical frame.
Invalid/sky points are omitted, so width may be 0 (an empty cloud).

Only that layout is accepted. Any other layout (extra or missing fields,
other datatypes, offsets, padding, organised clouds) is rejected rather than
reinterpreted, so a producer-side change fails loudly here. Both byte orders
are decoded; the result is native-endian float32.

The cloud is carried as-is: no TF, filtering or rasterisation happens here.
"""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

from costmap_core.contracts import ContractError, PointCloudInput
from costmap_ros.mask_adapter import stamp_to_ns

XYZ_OFFSETS = {"x": 0, "y": 4, "z": 8}
XYZ_POINT_STEP = 12


def point_cloud_from_pointcloud2(msg: PointCloud2) -> PointCloudInput:
    """Convert one XYZ float32 PointCloud2 into a PointCloudInput.

    stamp_ns and frame_id are taken unchanged from msg.header.

    Raises ContractError on any deviation from the layout above, a data
    length that does not match the metadata, non-finite coordinates, or a
    bad stamp/frame_id. Nothing is coerced.
    """
    if not isinstance(msg, PointCloud2):
        raise ContractError(f"expected sensor_msgs/msg/PointCloud2, got {type(msg).__name__}")

    names = [field.name for field in msg.fields]
    if sorted(names) != sorted(XYZ_OFFSETS):
        raise ContractError(f"PointCloud2 fields must be exactly x, y, z, got {names}")
    for field in msg.fields:
        if field.datatype != PointField.FLOAT32:
            raise ContractError(
                f"field {field.name!r} datatype must be FLOAT32 ({PointField.FLOAT32}), "
                f"got {field.datatype}"
            )
        if field.count != 1:
            raise ContractError(f"field {field.name!r} count must be 1, got {field.count}")
        if field.offset != XYZ_OFFSETS[field.name]:
            raise ContractError(
                f"field {field.name!r} offset must be {XYZ_OFFSETS[field.name]}, "
                f"got {field.offset}"
            )

    height, width = int(msg.height), int(msg.width)
    point_step, row_step = int(msg.point_step), int(msg.row_step)
    if point_step != XYZ_POINT_STEP:
        raise ContractError(f"point_step must be {XYZ_POINT_STEP}, got {point_step}")
    if height != 1:
        raise ContractError(f"cloud must be unorganised (height 1), got height {height}")
    if row_step != point_step * width:
        raise ContractError(
            f"row_step must be point_step*width = {point_step * width}, got {row_step}"
        )
    n_bytes = len(msg.data)
    if n_bytes != row_step * height:
        raise ContractError(
            f"cloud data has {n_bytes} bytes, expected row_step*height = {row_step * height}"
        )

    dtype = np.dtype(">f4" if msg.is_bigendian else "<f4")
    points = np.frombuffer(msg.data, dtype=dtype).reshape(width, 3).astype(np.float32)

    return PointCloudInput(
        points=points,
        stamp_ns=stamp_to_ns(msg.header.stamp),
        frame_id=msg.header.frame_id,
    )
