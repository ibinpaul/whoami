"""ROS-independent pixel-to-ground projection core.

Implements the geometry-only part of:

    pixel (u, v)
        -> camera intrinsics
        -> 3D camera ray
        -> ground-plane intersection
        -> ground position

This module does not know about images, masks, or costmaps -- it only
answers "where on the ground does this pixel look?" for a single pixel at a
time. Combining this with a semantic mask (to project a whole mask onto a
costmap grid) is a separate, not-yet-implemented component.

No real camera calibration values are defined here (see PROJECT_CONTEXT.md
Section 5): `CameraIntrinsics` and `CameraGroundGeometry` have no default
values, so callers must supply real or test values explicitly.
"""

import math

import numpy as np
from dataclasses import dataclass


class ProjectionError(ValueError):
    """Raised when projection inputs or geometry make the result invalid.

    Covers: invalid intrinsics, invalid camera/ground geometry, non-finite
    pixel coordinates, and camera rays that do not intersect the ground
    plane (parallel to or diverging away from the ground).
    """


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics.

    fx, fy: focal lengths in pixels.
    cx, cy: principal point in pixels.

    No defaults are provided on purpose: these values must come from a real
    CameraInfo message or a configuration file, never be invented here.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.fx) and self.fx > 0):
            raise ProjectionError(f"Invalid fx: {self.fx!r}. fx must be a finite value > 0.")
        if not (math.isfinite(self.fy) and self.fy > 0):
            raise ProjectionError(f"Invalid fy: {self.fy!r}. fy must be a finite value > 0.")


@dataclass(frozen=True)
class CameraGroundGeometry:
    """Camera pose relative to a flat ground plane.

    The ground plane is the world XY plane at z=0, in a right-handed world
    frame with x-forward, y-left, z-up (matching the map/odom/base_link
    convention). The camera is mounted at `camera_height` above the ground,
    directly above the world-frame origin, and pitched downward from
    horizontal by `pitch_rad` (0 = looking along the horizon).

    Roll and yaw are assumed zero. This is a deliberate simplification for
    this first projection core -- a full 6-DoF extrinsic pose is not
    implemented here.
    """

    camera_height: float
    pitch_rad: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.camera_height) and self.camera_height > 0):
            raise ProjectionError(
                f"Invalid camera_height: {self.camera_height!r}. "
                "camera_height must be a finite value > 0 (camera must be above the ground)."
            )
        if not math.isfinite(self.pitch_rad):
            raise ProjectionError(f"Invalid pitch_rad: {self.pitch_rad!r}. Must be finite.")


# Numerical tolerance for pose validation: max |R^T R - I| entry, |det(R) - 1|
# and | |q| - 1 |. Absorbs float64 round-off from upstream quaternion or matrix
# arithmetic. It is not a real-world value.
POSE_TOLERANCE = 1e-6


def _finite_float_array(value: object, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    """Read-only float64 copy of `value`, or ProjectionError. Never coerces bools."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ProjectionError(f"{name} must be a real numeric array, got dtype {array.dtype}")
    if array.shape != shape:
        raise ProjectionError(f"{name} must have shape {shape}, got {array.shape}")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ProjectionError(f"{name} must be finite, got {array.tolist()}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class CameraPose:
    """Rigid 6-DoF pose of the camera optical frame in the ground/costmap frame.

    rotation: 3x3 matrix taking a direction in the camera optical frame
        (x-right, y-down, z-forward) to the same direction in the ground
        frame. Must be orthonormal with det = +1, within POSE_TOLERANCE.
    translation: camera optical centre in the ground frame, (x, y, z).

    This is the transform a TF lookup (target = ground frame, source = camera
    optical frame) returns. The ground is the z = 0 plane of the ground
    frame (flat-ground model), so translation z must be > 0: the camera must
    be above the ground, as with CameraGroundGeometry.camera_height > 0.

    Both arrays are stored as read-only float64 copies. No defaults.
    """

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = _finite_float_array(self.rotation, (3, 3), name="rotation")
        translation = _finite_float_array(self.translation, (3,), name="translation")

        orthonormal_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        if orthonormal_error > POSE_TOLERANCE:
            raise ProjectionError(
                f"rotation is not orthonormal: max |R^T R - I| = {orthonormal_error!r} "
                f"> {POSE_TOLERANCE}"
            )
        det = float(np.linalg.det(rotation))
        if abs(det - 1.0) > POSE_TOLERANCE:
            raise ProjectionError(
                f"rotation determinant must be +1 (a proper rotation), got {det!r}"
            )
        if not translation[2] > 0.0:
            raise ProjectionError(
                f"translation z must be > 0 (camera above the ground plane), "
                f"got {float(translation[2])!r}"
            )

        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)


@dataclass(frozen=True)
class GroundPoint:
    """A point on the ground plane (z=0 implicit), in the world frame."""

    x: float
    y: float


# Maps a camera-optical-frame vector (x-right, y-down, z-forward) to the
# equivalent direction in the world frame (x-forward, y-left, z-up) when the
# camera has zero pitch: camera-forward (z) becomes world-forward (x),
# camera-right (x) becomes world-right (-y), camera-down (y) becomes
# world-down (-z).
_OPTICAL_TO_LEVEL_WORLD = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)

# Ray must point at least this far below horizontal (in world-frame z) to be
# treated as intersecting the ground plane. Guards against dividing by a
# near-zero value for rays that are only numerically, not meaningfully,
# downward.
_MIN_DOWNWARD_Z_COMPONENT = 1e-9


def _pitch_rotation(pitch: float) -> np.ndarray:
    """Rotation about the world y axis by `pitch` (positive = nose down)."""
    return np.array(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ]
    )


def _require_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ProjectionError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ProjectionError(f"{name} must be finite, got {value!r}")
    return value


def camera_pose_from_quaternion(
    *, qx: float, qy: float, qz: float, qw: float, tx: float, ty: float, tz: float
) -> CameraPose:
    """Build a CameraPose from a rotation quaternion and a translation.

    The quaternion is (x, y, z, w), Hamilton convention, the same order and
    meaning as a ROS geometry_msgs/Quaternion (no ROS types are used here).
    Its norm must be 1 within POSE_TOLERANCE; it is then normalised so the
    rotation matrix is orthonormal to float64 precision. q and -q give the
    same rotation. Keyword-only, so x/y/z/w order cannot be mixed up.

    Raises ProjectionError for non-real, non-finite or non-unit quaternions
    and for an invalid translation (see CameraPose).
    """
    q = np.array(
        [
            _require_real(qx, name="qx"),
            _require_real(qy, name="qy"),
            _require_real(qz, name="qz"),
            _require_real(qw, name="qw"),
        ]
    )
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > POSE_TOLERANCE:
        raise ProjectionError(f"quaternion must have unit norm, got |q| = {norm!r}")
    x, y, z, w = q / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    translation = [
        _require_real(tx, name="tx"),
        _require_real(ty, name="ty"),
        _require_real(tz, name="tz"),
    ]
    return CameraPose(rotation=rotation, translation=translation)


def camera_ground_geometry_to_pose(geometry: CameraGroundGeometry) -> CameraPose:
    """The CameraPose equivalent to a height + pitch CameraGroundGeometry.

    rotation = R_y(pitch) @ optical-to-level-world axis change,
    translation = (0, 0, camera_height).
    """
    if not isinstance(geometry, CameraGroundGeometry):
        raise TypeError(
            f"geometry must be a CameraGroundGeometry, got {type(geometry).__name__}"
        )
    return CameraPose(
        rotation=_pitch_rotation(geometry.pitch_rad) @ _OPTICAL_TO_LEVEL_WORLD,
        translation=[0.0, 0.0, geometry.camera_height],
    )


def pixel_to_camera_ray(u: float, v: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Convert a pixel coordinate into a 3D ray direction in the camera's
    optical frame (x-right, y-down, z-forward). The returned vector is not
    normalized -- only its direction matters for the ground intersection.

    Raises ProjectionError if u or v is not finite.
    """
    if not (math.isfinite(u) and math.isfinite(v)):
        raise ProjectionError(f"Invalid pixel coordinates: u={u!r}, v={v!r}. Both must be finite.")

    x_cam = (u - intrinsics.cx) / intrinsics.fx
    y_cam = (v - intrinsics.cy) / intrinsics.fy
    return np.array([x_cam, y_cam, 1.0])


def _require_downward(world_dir: np.ndarray) -> None:
    if world_dir[2] >= -_MIN_DOWNWARD_Z_COMPONENT:
        raise ProjectionError(
            "Camera ray does not intersect the ground plane: world-frame "
            f"downward component is {world_dir[2]!r} (must be < 0). The ray "
            "is parallel to or diverging away from the ground."
        )


def camera_ray_to_ground_point(
    ray_camera: np.ndarray, geometry: CameraGroundGeometry | CameraPose
) -> GroundPoint:
    """Intersect a camera-frame ray with the ground plane (z = 0).

    CameraGroundGeometry: the ray is rotated by the pitch; the camera is at
    (0, 0, camera_height). This path is kept exactly as before.
    CameraPose: the ray is rotated by `rotation`; the camera is at
    `translation`, so the ground point includes the camera's x/y offset.

    Raises ProjectionError if the world-frame ray does not point downward
    (i.e. it is parallel to, or diverges away from, the ground plane), and
    TypeError for any other geometry type -- never ProjectionError, so a
    wrong type is not mistaken for a per-pixel miss.
    """
    if isinstance(geometry, CameraPose):
        world_dir = geometry.rotation @ ray_camera
        _require_downward(world_dir)
        origin = geometry.translation
        t = -origin[2] / world_dir[2]
        return GroundPoint(
            x=float(origin[0] + t * world_dir[0]),
            y=float(origin[1] + t * world_dir[1]),
        )
    if not isinstance(geometry, CameraGroundGeometry):
        raise TypeError(
            "geometry must be a CameraGroundGeometry or CameraPose, "
            f"got {type(geometry).__name__}"
        )

    world_dir = _pitch_rotation(geometry.pitch_rad) @ (_OPTICAL_TO_LEVEL_WORLD @ ray_camera)
    _require_downward(world_dir)

    t = -geometry.camera_height / world_dir[2]
    ground_x = t * world_dir[0]
    ground_y = t * world_dir[1]
    return GroundPoint(x=ground_x, y=ground_y)


def project_pixel_to_ground(
    u: float, v: float, intrinsics: CameraIntrinsics, geometry: CameraGroundGeometry | CameraPose
) -> GroundPoint:
    """Project a single pixel to a ground-plane point.

    Convenience wrapper chaining `pixel_to_camera_ray` and
    `camera_ray_to_ground_point`.
    """
    ray_camera = pixel_to_camera_ray(u, v, intrinsics)
    return camera_ray_to_ground_point(ray_camera, geometry)
