"""std_msgs/Float64MultiArray (Dev 1 `/segmentation/port_meta`) -> PortMetaStopgap.

This decodes Dev 1's live stopgap, not the final interface. Dev 1's
`PortMeta.msg` (header, valid, age, scale) is only a spec file and is not
compiled; the node publishes a Float64MultiArray built in
`ugv_perception/node/wire.py`:

    layout.dim labels = ["valid", "age", "scale"], data_offset = 0
    data = [1.0 if valid else 0.0, age_s, scale]

The message has no header, so the decoded value has no stamp or frame_id.
Which mask it belongs to cannot be derived from it, and this module does
not try to pair it with one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from std_msgs.msg import Float64MultiArray

from costmap_core.contracts import ContractError

PORT_META_LABELS = ("valid", "age", "scale")

# Dev 1's v1 port only publishes scale 1.0, and the Dev 3 core assumes the
# mask is at source resolution. Any other scale is rejected, not rescaled.
SUPPORTED_SCALE = 1.0


@dataclass(frozen=True)
class PortMetaStopgap:
    """Decoded port_meta. Deliberately carries no stamp or frame_id.

    valid: Dev 1's validity of the mask this meta was published for.
    age_s: Dev 1's (now - image stamp) in seconds at compose time.
        Informational only; freshness must be judged from the mask's
        header.stamp against the consumer's own clock.
    scale: mask-to-source resolution scale.
    """

    valid: bool
    age_s: float
    scale: float


def _decode_valid(value: float) -> bool:
    if value == 1.0:
        return True
    if value == 0.0:
        return False
    raise ContractError(f"port_meta valid must be exactly 1.0 or 0.0, got {value!r}")


def port_meta_from_float64_multi_array(msg: Float64MultiArray) -> PortMetaStopgap:
    """Decode one port_meta message. Raises ContractError on any deviation."""
    if not isinstance(msg, Float64MultiArray):
        raise ContractError(
            f"expected std_msgs/msg/Float64MultiArray, got {type(msg).__name__}"
        )

    labels = tuple(dim.label for dim in msg.layout.dim)
    if labels != PORT_META_LABELS:
        raise ContractError(
            f"port_meta layout labels must be {list(PORT_META_LABELS)}, got {list(labels)}"
        )
    if msg.layout.data_offset != 0:
        raise ContractError(
            f"port_meta layout data_offset must be 0, got {msg.layout.data_offset}"
        )
    data = [float(x) for x in msg.data]
    if len(data) != len(PORT_META_LABELS):
        raise ContractError(f"port_meta data must have 3 values, got {len(data)}")
    raw_valid, age_s, scale = data

    valid = _decode_valid(raw_valid)
    if not math.isfinite(age_s) or age_s < 0.0:
        raise ContractError(f"port_meta age must be finite and >= 0 s, got {age_s!r}")
    if scale != SUPPORTED_SCALE:
        raise ContractError(
            f"port_meta scale must be {SUPPORTED_SCALE} (Dev 1 v1 port), got {scale!r}"
        )

    return PortMetaStopgap(valid=valid, age_s=age_s, scale=scale)
