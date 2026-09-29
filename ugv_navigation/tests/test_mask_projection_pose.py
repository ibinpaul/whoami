"""CameraPose through mask projection, the input contract and the pipeline.

EVERY camera, pose, grid, frame name and stamp below is SYNTHETIC. The
setup mirrors test_pipeline.py: a downward camera, fx = fy = 10, a 20x20
image over a 20x20 grid of 0.1 m cells, so each pixel lands at a cell centre.
"""

import math

import numpy as np
import pytest

from costmap_core.class_to_cost import DEFAULT_COST_VALUES, SemanticClass
from costmap_core.contracts import (
    CameraGroundInput,
    CameraIntrinsicsInput,
    ContractError,
    CostmapCoreInputs,
    GridInput,
    OccupancyInput,
    SemanticMaskInput,
)
from costmap_core.geometry_costmap import DEFAULT_GEOMETRY_COST_VALUES
from costmap_core.grid import CostmapGridGeometry
from costmap_core.mask_projection import project_mask_to_costmap
from costmap_core.pipeline import run_costmap_pipeline
from costmap_core.projection import (
    CameraGroundGeometry,
    CameraIntrinsics,
    CameraPose,
    camera_ground_geometry_to_pose,
)

UNKNOWN_COST = DEFAULT_COST_VALUES.unknown_cost
FREE_COST = DEFAULT_COST_VALUES.traversable_cost
LETHAL_COST = DEFAULT_GEOMETRY_COST_VALUES.lethal_cost
UNKNOWN = SemanticClass.UNKNOWN.value
TRAVERSABLE = SemanticClass.TRAVERSABLE.value
HAZARD = SemanticClass.HAZARD.value

SIZE = 20
INTRINSICS = CameraIntrinsics(fx=10.0, fy=10.0, cx=9.5, cy=9.5)
STRAIGHT_DOWN = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
YAW_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
CAM = "test_camera_frame"
GROUND = "test_ground_frame"
STAMP = 1_000_000_000

# Straight-down camera 1 m up at (3, -2), yawed +90 deg. Pixel offset
# (du, dv) = (u - 9.5, v - 9.5) lands at (3 + 0.1 du, -2 - 0.1 dv), so on a
# grid with origin (2, -3): pixel (u, v) -> cell (row = 19 - v, col = u).
X0, Y0 = 3.0, -2.0
YAWED_POSE = CameraPose(rotation=YAW_90 @ STRAIGHT_DOWN, translation=[X0, Y0, 1.0])
YAWED_GRID = CostmapGridGeometry(resolution=0.1, origin_x=X0 - 1.0, origin_y=Y0 - 1.0, width=SIZE, height=SIZE)


def yawed_cell(u, v):
    return SIZE - 1 - v, u


def test_yawed_translated_pose_maps_pixels_to_hand_derived_cells():
    for u in range(0, SIZE, 3):
        for v in range(0, SIZE, 3):
            mask = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
            mask[v, u] = HAZARD
            costmap = project_mask_to_costmap(mask, INTRINSICS, YAWED_POSE, YAWED_GRID)
            rows, cols = np.nonzero(costmap == LETHAL_COST)
            assert list(zip(rows.tolist(), cols.tolist())) == [yawed_cell(u, v)]


def test_pose_projection_keeps_class_precedence_and_unknown_default():
    # 0.2 m cells: 2x2 pixels per cell. A 14x14 grid is larger than the
    # 2 m x 2 m view, so its border cells are never reached.
    grid = CostmapGridGeometry(resolution=0.2, origin_x=X0 - 1.4, origin_y=Y0 - 1.4, width=14, height=14)
    mask = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    mask[0, 0] = HAZARD   # shares a cell with three traversable pixels
    mask[0, 19] = UNKNOWN  # shares a cell with three traversable pixels
    costmap = project_mask_to_costmap(mask, INTRINSICS, YAWED_POSE, grid)

    def cell_of(u, v):
        return (int((Y0 - 0.1 * (v - 9.5) - grid.origin_y) // 0.2),
                int((X0 + 0.1 * (u - 9.5) - grid.origin_x) // 0.2))

    assert costmap[cell_of(0, 0)] == LETHAL_COST       # hazard beats traversable
    assert costmap[cell_of(19, 0)] == UNKNOWN_COST     # unknown beats traversable
    assert costmap[cell_of(10, 10)] == FREE_COST
    assert np.all(costmap[0, :] == UNKNOWN_COST) and np.all(costmap[:, -1] == UNKNOWN_COST)


def test_pose_projection_skips_out_of_grid_and_sky_pixels():
    small = CostmapGridGeometry(resolution=0.1, origin_x=X0 - 0.2, origin_y=Y0 - 0.2, width=4, height=4)
    mask = np.full((SIZE, SIZE), HAZARD, dtype=np.uint8)
    costmap = project_mask_to_costmap(mask, INTRINSICS, YAWED_POSE, small)
    assert np.all(costmap == LETHAL_COST)  # only in-grid pixels, never clamped

    level = CameraPose(
        rotation=np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        translation=[X0, Y0, 1.0],
    )
    costmap = project_mask_to_costmap(mask, INTRINSICS, level, YAWED_GRID)
    assert np.all(costmap == UNKNOWN_COST)  # upper half is sky, lower half beyond the grid


def test_converted_legacy_pose_gives_identical_mask_costmap():
    rng = np.random.default_rng(7)
    mask = rng.integers(0, 3, size=(16, SIZE)).astype(np.uint8)
    intr = CameraIntrinsics(fx=10.0, fy=10.0, cx=9.5, cy=7.5)
    grid = CostmapGridGeometry(resolution=0.25, origin_x=0.0, origin_y=-5.0, width=40, height=40)
    geometry = CameraGroundGeometry(camera_height=2.0, pitch_rad=math.radians(30.0))
    legacy = project_mask_to_costmap(mask, intr, geometry, grid)
    pose = project_mask_to_costmap(mask, intr, camera_ground_geometry_to_pose(geometry), grid)
    np.testing.assert_array_equal(pose, legacy)
    assert np.any(legacy != UNKNOWN_COST)


def test_mask_projection_rejects_wrong_geometry_type_loudly():
    mask = np.full((2, 2), TRAVERSABLE, dtype=np.uint8)
    with pytest.raises(TypeError):
        project_mask_to_costmap(mask, INTRINSICS, (1.0, 0.5), YAWED_GRID)


# --- Contract and pipeline ----------------------------------------------------------


def make_inputs(geometry, classes, occupied=None, grid=YAWED_GRID):
    return CostmapCoreInputs(
        mask=SemanticMaskInput(classes=classes, stamp_ns=STAMP, frame_id=CAM, valid=True),
        intrinsics=CameraIntrinsicsInput(
            intrinsics=INTRINSICS, image_width=SIZE, image_height=SIZE, frame_id=CAM
        ),
        camera_ground=CameraGroundInput(
            geometry=geometry, camera_frame_id=CAM, ground_frame_id=GROUND, stamp_ns=STAMP
        ),
        grid=GridInput(geometry=grid, frame_id=GROUND),
        occupancy=None if occupied is None else OccupancyInput(occupied=occupied, stamp_ns=STAMP, frame_id=GROUND),
    )


def test_camera_ground_input_accepts_pose():
    cg = CameraGroundInput(geometry=YAWED_POSE, camera_frame_id=CAM, ground_frame_id=GROUND, stamp_ns=STAMP)
    assert cg.geometry is YAWED_POSE


@pytest.mark.parametrize("bad", [None, (1.0, 0.5), np.eye(3)])
def test_camera_ground_input_still_rejects_other_types(bad):
    with pytest.raises(ContractError, match="CameraPose"):
        CameraGroundInput(geometry=bad, camera_frame_id=CAM, ground_frame_id=GROUND, stamp_ns=STAMP)


def test_pipeline_with_pose_keeps_geometry_lethal_precedence():
    classes = np.full((SIZE, SIZE), TRAVERSABLE, dtype=np.uint8)
    classes[4, 6] = UNKNOWN
    occupied = np.zeros((SIZE, SIZE), dtype=bool)
    occupied[yawed_cell(12, 5)] = True  # under semantic traversable
    result = run_costmap_pipeline(make_inputs(YAWED_POSE, classes, occupied), inflation_radius=0.3)

    assert result.semantic[yawed_cell(12, 5)] == FREE_COST
    assert result.fused[yawed_cell(12, 5)] == LETHAL_COST
    assert result.final[yawed_cell(12, 5)] == LETHAL_COST
    assert result.semantic[yawed_cell(6, 4)] == UNKNOWN_COST
    assert result.final[yawed_cell(6, 4)] == UNKNOWN_COST


def test_pipeline_with_converted_pose_matches_legacy_geometry():
    # Legacy camera sits above the origin, so use test_pipeline.py's grid.
    legacy_geometry = CameraGroundGeometry(camera_height=1.0, pitch_rad=math.pi / 2)
    grid = CostmapGridGeometry(resolution=0.1, origin_x=-1.0, origin_y=-1.0, width=SIZE, height=SIZE)
    rng = np.random.default_rng(3)
    classes = rng.integers(0, 3, size=(SIZE, SIZE)).astype(np.uint8)
    occupied = np.zeros((SIZE, SIZE), dtype=bool)
    occupied[5, 5] = True
    legacy = run_costmap_pipeline(
        make_inputs(legacy_geometry, classes, occupied, grid=grid), inflation_radius=0.3
    )
    pose = run_costmap_pipeline(
        make_inputs(camera_ground_geometry_to_pose(legacy_geometry), classes, occupied, grid=grid),
        inflation_radius=0.3,
    )
    assert np.any(legacy.semantic == FREE_COST) and np.any(legacy.semantic == LETHAL_COST)
    for stage in ("semantic", "fused", "final"):
        np.testing.assert_array_equal(getattr(pose, stage), getattr(legacy, stage), err_msg=stage)
