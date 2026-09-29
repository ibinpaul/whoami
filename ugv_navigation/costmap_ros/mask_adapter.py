"""sensor_msgs/Image (Dev 1 `/segmentation/mask`) -> SemanticMaskInput.

Dev 1's canonical port: mono8, one byte per pixel, step == width, pixel
values already canonical {0 unknown, 1 traversable, 2 hazard}. The model's
native class ids never reach this adapter, so nothing is remapped here: the
bytes are passed through and SemanticMaskInput rejects any other value.
"""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image

from costmap_core.contracts import ContractError, SemanticMaskInput

MASK_ENCODING = "mono8"

_NS_PER_S = 1_000_000_000


def stamp_to_ns(stamp: object) -> int:
    """builtin_interfaces/Time -> integer nanoseconds."""
    return int(stamp.sec) * _NS_PER_S + int(stamp.nanosec)


def semantic_mask_from_image(msg: Image, *, valid: bool) -> SemanticMaskInput:
    """Convert one mask Image into a SemanticMaskInput.

    stamp_ns and frame_id are taken unchanged from msg.header (source image
    capture time and camera optical frame).

    valid: Dev 1's port validity flag, supplied by the caller. It lives in
        `/segmentation/port_meta`, whose final message format is PENDING, so
        this adapter does not parse port_meta and has no default.

    Raises ContractError on any deviation from the port contract (wrong type
    or encoding, row padding, size mismatch, non-canonical ids, bad stamp or
    frame_id). Nothing is coerced.
    """
    if not isinstance(msg, Image):
        raise ContractError(f"expected sensor_msgs/msg/Image, got {type(msg).__name__}")
    if msg.encoding != MASK_ENCODING:
        raise ContractError(
            f"mask encoding must be {MASK_ENCODING!r}, got {msg.encoding!r}"
        )

    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    if height <= 0 or width <= 0:
        raise ContractError(f"mask size must be positive, got {width}x{height}")
    if step != width:
        raise ContractError(f"mono8 mask step must equal width ({width}), got {step}")

    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size != height * step:
        raise ContractError(
            f"mask data has {data.size} bytes, expected height*step = {height * step}"
        )

    return SemanticMaskInput(
        classes=data.reshape(height, width),
        stamp_ns=stamp_to_ns(msg.header.stamp),
        frame_id=msg.header.frame_id,
        valid=valid,
    )
