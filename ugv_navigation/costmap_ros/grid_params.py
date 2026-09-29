"""Explicit grid parameters -> GridInput.

Resolution, extent, origin and frame name are not decided yet, so every
value is required and none has a default. The ROS parameter namespace and
the config/robots/ file that will supply them are not defined yet either;
a future node passes the values it reads as a plain mapping.
"""

from __future__ import annotations

from collections.abc import Mapping

from costmap_core.contracts import ContractError, GridInput
from costmap_core.grid import CostmapGridGeometry

GRID_PARAMETER_NAMES = ("resolution", "origin_x", "origin_y", "width", "height", "frame_id")


def _require_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a real number, got {value!r}")
    return float(value)


def grid_input_from_parameters(params: Mapping[str, object]) -> GridInput:
    """Build a GridInput from exactly the keys in GRID_PARAMETER_NAMES.

    Missing or unknown keys raise ContractError. Range checks are left to
    CostmapGridGeometry (GridError) and GridInput (ContractError).
    """
    missing = [name for name in GRID_PARAMETER_NAMES if name not in params]
    unknown = sorted(set(params) - set(GRID_PARAMETER_NAMES))
    if missing or unknown:
        raise ContractError(f"grid parameters: missing {missing}, unknown {unknown}")

    geometry = CostmapGridGeometry(
        resolution=_require_real(params["resolution"], name="resolution"),
        origin_x=_require_real(params["origin_x"], name="origin_x"),
        origin_y=_require_real(params["origin_y"], name="origin_y"),
        width=params["width"],
        height=params["height"],
    )
    return GridInput(geometry=geometry, frame_id=params["frame_id"])
