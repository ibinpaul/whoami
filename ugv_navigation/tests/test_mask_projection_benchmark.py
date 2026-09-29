"""Timing record for project_mask_to_costmap at representative resolutions.

No test here fails on runtime: timings are machine-dependent, so they are
only recorded (as junit testsuite properties, and printed; run `pytest -s`
to see them). Each case also checks the result is a sane costmap so the
benchmark cannot silently time a broken path. All scene values are synthetic.
"""

import time

import numpy as np
import pytest

from costmap_core.grid import CostmapGridGeometry
from costmap_core.mask_projection import project_mask_to_costmap
from costmap_core.projection import CameraGroundGeometry, CameraIntrinsics, camera_ground_geometry_to_pose
from mask_projection_reference import reference_project_mask_to_costmap

GRID = CostmapGridGeometry(resolution=0.1, origin_x=0.0, origin_y=-5.0, width=100, height=100)
POSE = camera_ground_geometry_to_pose(CameraGroundGeometry(camera_height=1.0, pitch_rad=0.35))
REPEATS = 3


def _scene(width, height):
    mask = np.random.default_rng(0).integers(0, 3, (height, width)).astype(np.uint8)
    intrinsics = CameraIntrinsics(fx=0.8 * width, fy=0.8 * width, cx=(width - 1) / 2, cy=(height - 1) / 2)
    return mask, intrinsics


def _best_time_s(fn, repeats):
    best, result = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


@pytest.mark.parametrize("width, height", [(320, 240), (640, 480)])
def test_benchmark_vectorised(width, height, record_testsuite_property):
    mask, intrinsics = _scene(width, height)
    seconds, costmap = _best_time_s(
        lambda: project_mask_to_costmap(mask, intrinsics, POSE, GRID), REPEATS
    )
    record_testsuite_property(f"project_mask_to_costmap_{width}x{height}_ms", round(seconds * 1000, 2))
    print(f"\nproject_mask_to_costmap {width}x{height}: {seconds * 1000:.1f} ms (best of {REPEATS})")
    assert costmap.shape == (GRID.height, GRID.width)
    assert len(np.unique(costmap)) == 3  # the scene reaches free, hazard and unknown cells


def test_benchmark_speedup_vs_reference_320x240(record_testsuite_property):
    # The per-pixel reference takes ~1 s here, so it is timed once and only at 320x240.
    mask, intrinsics = _scene(320, 240)
    ref_s, expected = _best_time_s(
        lambda: reference_project_mask_to_costmap(mask, intrinsics, POSE, GRID), 1
    )
    new_s, actual = _best_time_s(
        lambda: project_mask_to_costmap(mask, intrinsics, POSE, GRID), REPEATS
    )
    record_testsuite_property("reference_320x240_ms", round(ref_s * 1000, 2))
    record_testsuite_property("vectorised_320x240_ms", round(new_s * 1000, 2))
    print(f"\n320x240 reference {ref_s * 1000:.1f} ms, vectorised {new_s * 1000:.1f} ms, "
          f"speedup x{ref_s / new_s:.0f}")
    np.testing.assert_array_equal(actual, expected)
