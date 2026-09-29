"""Semantic mask -> costmap projection.

Bridges the pieces that already exist as separate, independently-tested
stages into the one pipeline described in PROJECT_CONTEXT.md's Dev 3 flow:

    semantic mask (per-pixel class ids)
        -> pixel (u, v)
        -> camera ray                (projection.pixel_to_camera_ray)
        -> ground point               (projection.camera_ray_to_ground_point)
        -> costmap grid cell          (grid.world_to_grid_cell)
        -> semantic cost written into that cell (class_to_cost.class_to_cost)

This module is ROS-independent: it does not touch sensor_msgs, TF, real
camera calibration, Depth Anything, or inflation.

The whole mask is processed as numpy arrays in one pass. Each step performs
the same float64 operations, in the same order, as the single-pixel helpers
it mirrors (`projection.pixel_to_camera_ray`,
`projection.camera_ray_to_ground_point`, `grid.world_to_grid_cell`), so
results match them exactly, cell for cell. In particular every ray is
rotated with numpy matmul of a 3x3 matrix by a (3, 1) vector -- the same
kernel as the single-pixel `rotation @ ray` -- rather than a batched
`rays @ R.T`, whose different summation order can move a ground point by
one ulp across a cell boundary. tests/mask_projection_reference.py keeps
the original per-pixel loop for comparison.
"""

from __future__ import annotations

import numpy as np

from costmap_core.class_to_cost import (
    CostValues,
    DEFAULT_COST_VALUES,
    InvalidSemanticClassError,
    SemanticClass,
    class_to_cost,
)
from costmap_core.grid import CostmapGridGeometry
from costmap_core.projection import (
    _MIN_DOWNWARD_Z_COMPONENT,
    _OPTICAL_TO_LEVEL_WORLD,
    CameraGroundGeometry,
    CameraIntrinsics,
    CameraPose,
    _pitch_rotation,
)

# Precedence used when more than one mask pixel projects into the same cell:
# the class with the higher rank wins, i.e. HAZARD > UNKNOWN > TRAVERSABLE.
# This is decided on semantic classes, NOT on numeric costs: with the default
# costs UNKNOWN (255) is numerically above HAZARD (254), so a numeric max
# would let "no information" overwrite known hazard evidence.
SEMANTIC_CLASS_PRECEDENCE = {
    SemanticClass.TRAVERSABLE: 0,
    SemanticClass.UNKNOWN: 1,
    SemanticClass.HAZARD: 2,
}


def project_mask_to_costmap(
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    geometry: CameraGroundGeometry | CameraPose,
    grid_geometry: CostmapGridGeometry,
    cost_values: CostValues = DEFAULT_COST_VALUES,
) -> np.ndarray:
    """Project a 2D semantic class mask into a costmap grid.

    Args:
        mask: 2D array-like of canonical semantic class ids (0=unknown,
            1=traversable, 2=hazard), one entry per image pixel. Pixel
            (row, col) is treated as image coordinates (u=col, v=row) --
            the same convention `projection.py` expects.
        intrinsics: pinhole camera intrinsics (see `projection.py`).
        geometry: camera pose relative to the ground plane: either the
            height + pitch CameraGroundGeometry or a 6-DoF CameraPose in
            the grid's frame (see `projection.py`).
        grid_geometry: costmap grid geometry (see `grid.py`).
        cost_values: cost values for each canonical class; defaults to
            `class_to_cost.DEFAULT_COST_VALUES`.

    Returns:
        A 2D numpy array of shape (grid_geometry.height,
        grid_geometry.width) and dtype int64.

        Cells that no mask pixel projects into keep
        `cost_values.unknown_cost`. This mirrors the existing convention
        (class 0 = "never free") elsewhere in this package: "no
        perception data reached this cell" is treated the same as
        "perception says unknown", not as "free".

        When more than one mask pixel projects into the same cell, the
        winning semantic class is chosen by `SEMANTIC_CLASS_PRECEDENCE`
        (HAZARD > UNKNOWN > TRAVERSABLE) and only then converted to a
        cost. A hazard pixel is therefore never overwritten by an unknown
        pixel, regardless of the numeric cost values.

    Pixel handling (never raises for these; the pixel is skipped instead):
        - A pixel whose camera ray does not intersect the ground plane
          (`ProjectionError` from `projection.py`, e.g. a "sky" pixel
          under a downward-pitched camera) is skipped.
        - A pixel whose ground point falls outside `grid_geometry`
          (`GridError` from `grid.py`) is skipped -- it is never clamped
          into the nearest in-bounds cell.

    Raises:
        InvalidSemanticClassError: if `mask` is not 2D, or any pixel
            holds a class id outside {0, 1, 2}. This is checked for
            every pixel regardless of whether that pixel's ray would
            have been skipped, so a malformed mask always fails loudly.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise InvalidSemanticClassError(
            f"Expected a 2D semantic class mask, got array with shape "
            f"{mask.shape} (ndim={mask.ndim})."
        )

    shape = (grid_geometry.height, grid_geometry.width)
    costmap = np.full(shape, cost_values.unknown_cost, dtype=np.int64)
    if mask.size == 0:
        return costmap

    class_ids = _validated_class_ids(mask, geometry, cost_values)
    rays = _pixel_rays(mask.shape, intrinsics)
    hits, ground_x, ground_y = _rays_to_ground(rays, geometry)
    in_grid, flat_cells = _ground_to_flat_cells(ground_x, ground_y, grid_geometry)

    ranks = _CLASS_RANK[class_ids[hits][in_grid]]
    # Highest precedence rank per cell; -1 = no pixel reached the cell.
    winning_rank = np.full(costmap.size, -1, dtype=np.int64)
    np.maximum.at(winning_rank, flat_cells, ranks)

    flat_costmap = costmap.reshape(-1)
    for semantic_class, cost in cost_values.as_mapping().items():
        cells = winning_rank == SEMANTIC_CLASS_PRECEDENCE[semantic_class]
        if cells.any():
            flat_costmap[cells] = cost
    return costmap


_VALID_CLASS_IDS = tuple(int(c) for c in SemanticClass)
# Precedence rank indexed by class id.
_CLASS_RANK = np.array(
    [SEMANTIC_CLASS_PRECEDENCE[SemanticClass(i)] for i in range(len(SemanticClass))],
    dtype=np.int64,
)


def _validated_class_ids(
    mask: np.ndarray, geometry: object, cost_values: CostValues
) -> np.ndarray:
    """Flat int64 class ids of a non-empty 2D mask, in row-major order.

    Errors are raised in the same order as the original per-pixel loop,
    which checked pixel 0's class, then the geometry type (on its first
    projection), then every later pixel's class in row-major order:
    InvalidSemanticClassError (via class_to_cost) for the first invalid id,
    TypeError for a wrong geometry type. Class ids are int(value), as before.
    """
    class_to_cost(int(mask.flat[0]), cost_values)
    if not isinstance(geometry, (CameraGroundGeometry, CameraPose)):
        raise TypeError(
            "geometry must be a CameraGroundGeometry or CameraPose, "
            f"got {type(geometry).__name__}"
        )

    if mask.dtype.kind in "biu":
        flat = mask.reshape(-1)
        valid = (flat >= 0) & (flat <= _VALID_CLASS_IDS[-1])
        if not valid.all():
            class_to_cost(int(flat[np.argmin(valid)]), cost_values)
        return flat.astype(np.int64)

    # Other dtypes (float, object, ...): int() per pixel, exactly as before.
    ids = np.empty(mask.size, dtype=np.int64)
    for i, value in enumerate(mask.flat):
        class_id = int(value)
        if class_id not in _VALID_CLASS_IDS:
            class_to_cost(class_id, cost_values)
        ids[i] = class_id
    return ids


def _pixel_rays(mask_shape: tuple[int, int], intrinsics: CameraIntrinsics) -> np.ndarray:
    """(N, 3) optical-frame rays for every pixel, row-major; see pixel_to_camera_ray."""
    height, width = mask_shape
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    rays = np.empty((height, width, 3), dtype=np.float64)
    rays[:, :, 0] = ((u - intrinsics.cx) / intrinsics.fx)[np.newaxis, :]
    rays[:, :, 1] = ((v - intrinsics.cy) / intrinsics.fy)[:, np.newaxis]
    rays[:, :, 2] = 1.0
    return rays.reshape(-1, 3)


def _rotate(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """matrix @ v for each row v of (N, 3) `vectors`, using the same (3x3 @ 3x1)
    matmul kernel as the single-vector product so rounding is identical."""
    return np.matmul(matrix, vectors[:, :, np.newaxis])[:, :, 0]


def _rays_to_ground(
    rays: np.ndarray, geometry: CameraGroundGeometry | CameraPose
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised camera_ray_to_ground_point.

    Returns (hits, x, y): hits is a bool mask over the rays; x and y are the
    ground points of the hitting rays only. A ray misses when its world z
    component is not < -_MIN_DOWNWARD_Z_COMPONENT (the per-pixel
    ProjectionError case).
    """
    if isinstance(geometry, CameraPose):
        world = _rotate(geometry.rotation, rays)
    else:
        level = _rotate(_OPTICAL_TO_LEVEL_WORLD, rays)
        world = _rotate(_pitch_rotation(geometry.pitch_rad), level)

    hits = ~(world[:, 2] >= -_MIN_DOWNWARD_Z_COMPONENT)
    world = world[hits]
    if isinstance(geometry, CameraPose):
        origin = geometry.translation
        t = -origin[2] / world[:, 2]
        return hits, origin[0] + t * world[:, 0], origin[1] + t * world[:, 1]
    t = -geometry.camera_height / world[:, 2]
    return hits, t * world[:, 0], t * world[:, 1]


def _ground_to_flat_cells(
    x: np.ndarray, y: np.ndarray, grid_geometry: CostmapGridGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised world_to_grid_cell.

    Returns (in_grid, flat_cells): in_grid is a bool mask over the points;
    flat_cells are row * width + col for the in-grid points only. Points
    that are non-finite or outside the grid are dropped (the per-pixel
    GridError case), never clamped.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        col = np.floor((x - grid_geometry.origin_x) / grid_geometry.resolution)
        row = np.floor((y - grid_geometry.origin_y) / grid_geometry.resolution)
        in_grid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (col >= 0)
            & (col < grid_geometry.width)
            & (row >= 0)
            & (row < grid_geometry.height)
        )
    flat_cells = row[in_grid].astype(np.int64) * grid_geometry.width + col[in_grid].astype(np.int64)
    return in_grid, flat_cells
