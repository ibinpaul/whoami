"""Timing record for inflate_costmap at representative grid sizes.

No test here fails on runtime: timings are machine-dependent, so they are
only recorded (as junit testsuite properties, and printed; run `pytest -s`
to see them). Each case also checks the output is sane so the benchmark
cannot silently time a broken path. All grid values are synthetic.
"""

import time

import numpy as np
import pytest

from costmap_core.inflation import inflate_costmap
from inflation_reference import reference_inflate_costmap

LETHAL = 254
REPEATS = 3
# (cells per side, resolution m, inflation radius m); 5 % lethal, 20 % unknown.
SCENES = [(100, 0.1, 0.5), (200, 0.05, 0.5), (400, 0.05, 1.0)]


def _grid(cells):
    rng = np.random.default_rng(0)
    return rng.choice(np.array([0, LETHAL, 255]), size=(cells, cells), p=[0.75, 0.05, 0.20])


def _best_time_s(fn, repeats):
    best, result = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


@pytest.mark.parametrize("cells, resolution, radius", SCENES)
def test_benchmark_inflation(cells, resolution, radius, record_testsuite_property):
    grid = _grid(cells)
    seconds, result = _best_time_s(lambda: inflate_costmap(grid, resolution, radius), REPEATS)
    key = f"inflate_costmap_{cells}x{cells}_res{resolution}_r{radius}_ms"
    record_testsuite_property(key, round(seconds * 1000, 2))
    print(f"\ninflate_costmap {cells}x{cells} @{resolution} m, radius {radius} m: "
          f"{seconds * 1000:.1f} ms (best of {REPEATS})")
    assert (result[grid == LETHAL] == LETHAL).all()
    assert ((result > 0) & (result < LETHAL)).any()  # some cells were inflated


def test_benchmark_inflation_speedup_vs_reference_100x100(record_testsuite_property):
    # The per-cell reference takes ~0.1 s here (seconds on larger grids), so it
    # is timed once and only on the smallest scene.
    cells, resolution, radius = SCENES[0]
    grid = _grid(cells)
    ref_s, expected = _best_time_s(lambda: reference_inflate_costmap(grid, resolution, radius), 1)
    new_s, actual = _best_time_s(lambda: inflate_costmap(grid, resolution, radius), REPEATS)
    record_testsuite_property("inflate_reference_100x100_ms", round(ref_s * 1000, 2))
    record_testsuite_property("inflate_vectorised_100x100_ms", round(new_s * 1000, 2))
    print(f"\ninflation 100x100 reference {ref_s * 1000:.1f} ms, vectorised {new_s * 1000:.1f} ms, "
          f"speedup x{ref_s / new_s:.0f}")
    np.testing.assert_array_equal(actual, expected)
