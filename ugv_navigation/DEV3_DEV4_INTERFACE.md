# Dev 3 → downstream (Dev 4 / Nav2) output contract

What the Dev 3 costmap node (`costmap_ros/costmap_node.py`) publishes today,
stated precisely enough for a consumer to depend on. Every statement below is
implemented and covered by tests in `ugv_navigation/tests/` (1673 passing at
the time of writing). Values marked **PENDING** are not defined by any project
document and have **no default** in code: the node refuses to start without
them.

This is the output of a standalone ROS 2 node. **Nav2 integration is not
done**; how this output enters a Nav2 costmap (StaticLayer, a custom layer, or
Nav2 layers replacing parts of Dev 3) is a team decision (see §10).

Sources: `architecture.md` (**ARCH**), `dev.md` (**DEV**), Dev 3 code.

---

## 1. Message, topic, QoS

| Item | Contract | Where |
|---|---|---|
| Type | `nav_msgs/msg/OccupancyGrid` | `occupancy_grid.occupancy_grid_from_costmap` |
| Topic | ROS parameter `costmap_topic` (required, no default). DEV §3 names `/global_costmap/costmap` and `/local_costmap/costmap`; which one a given node instance publishes is a deployment choice. One node instance = one grid = one topic. | `costmap_node.PARAMETERS` |
| Publisher QoS | depth 10, RELIABLE, VOLATILE (rclpy default). Not latched: a late subscriber receives the next publication, not the previous one. A TRANSIENT_LOCAL subscription cannot match it. | `CostmapNode.__init__` |

## 2. Frame and grid geometry

| Field | Contract |
|---|---|
| `header.frame_id` | ROS parameter `target_frame` (required). It is also the TF target for projection and the grid frame. Frame name **PENDING**. The frame must have z = 0 on the ground plane (flat-ground model; `CameraPose` rejects a camera at z ≤ 0). |
| `info.resolution` | `grid.resolution` (m/cell, finite > 0) |
| `info.width`, `info.height` | `grid.width` (columns, x), `grid.height` (rows, y), positive ints |
| `info.origin` | position (`grid.origin_x`, `grid.origin_y`, 0), orientation identity (w = 1). It is the **min-x, min-y corner of cell (0, 0)**. |
| `data` | row-major: index = `row * width + col`; row increases with +y, col with +x. Cell (row, col) covers x ∈ [origin_x + col·res, +res), y ∈ [origin_y + row·res, +res). |
| Grid motion | **Fixed origin** in `target_frame`: the grid does not move with the robot (no rolling window). |

All grid values are **PENDING** (Dev 3 `config/robots/`, DEV §1). None has a default.

## 3. Cost semantics and encoding

**The published costmap is NOT inflated.** It is
`CostmapPipelineResult.fused`: semantic costs with geometry-lethal precedence
applied. Obstacle inflation belongs to Nav2 downstream. Dev 3's own
inflation (`inflation.py`, `CostmapPipelineResult.final`) still exists and is
still computed, but it is not published.

Internal Dev 3 costs (Nav2 `costmap_2d` conventions) are translated with
Nav2's own `Costmap2DPublisher` table (`occupancy_grid.COST_TO_OCCUPANCY`).
Costs outside 0..255 are rejected, never clipped.

| Internal cost | Meaning in Dev 3 | `OccupancyGrid.data` | In the published output |
|---|---|---|---|
| 0 | free: semantic traversable, no geometry obstacle | 0 | yes |
| 254 | lethal: semantic hazard, geometric obstacle, or fail-safe ROI | 100 | yes |
| 255 | unknown: semantic unknown, or a cell no mask pixel reached | −1 | yes |
| 1..252 | Dev 3 inflation | 1..98 (1 + 97·(cost − 1) // 251) | **no** |
| 253 | Dev 3 inflation (see note) | 99 | **no** |

So the published `data` contains only **0, 100 and −1** (tested).

Rules the consumer can rely on (enforced in code and tests):

- **Geometry lethal always wins** (ARCH §9): a geometrically occupied cell is
  lethal whatever the semantic class; semantic traversable never clears it.
- Several mask pixels in one cell: hazard > unknown > traversable.
- **Unknown is never emitted as free** (ARCH §8.1, §8.6).
- No Dev 3 inflation is applied to the published output; `inflation_radius`
  does not change it (tested).
- Missing, empty, stale or future-dated geometry never removes a lethal cell
  and never turns a cell free.

**Note on 1..253 (not published).** These values only appear in Dev 3's
unpublished `final` array. Dev 3's placeholder inflation rounds
`253 · (1 − d / inflation_radius)`, so it can produce 253 (Nav2 INSCRIBED)
without any inscribed-radius concept. That is one reason inflation is left
to Nav2.

## 4. Timestamp

| Publish path | `header.stamp` (= `info.map_load_time`) |
|---|---|
| Normal output for a mask | `output_stamp_source == "mask"`: that mask's header stamp (image capture time). `"now"`: the node clock when the costmap was built. |
| Re-fused republish (a depth cloud arrived after its mask; §5) | `"mask"`: the **same** mask's stamp, so two messages can share a stamp with different content (the second has geometry fused). `"now"`: the node clock at the re-fusion. |
| Fail-safe grid | **Always the node clock**, for either policy: no current mask exists. |

Consequence (tested): with `"mask"`, stamps are **not monotonic** across a
fail-safe → recovery transition. The recovery grid carries its mask's capture
time, which is older than the preceding fail-safe grid's node-clock stamp.
**Consumers must treat the most recently received message as current, not
the message with the largest stamp.** `output_stamp_source` is **PENDING**;
the node clock is ROS time (sim time under `use_sim_time`).

## 5. When the node publishes

| Trigger | Output |
|---|---|
| A mask arrives, passes `is_fresh(mask stamp, now, mask_max_age_s)`, and the pipeline succeeds | normal costmap |
| A depth cloud arrives that pairs with the last processed mask better than before (same camera frame, `0 ≤ mask stamp − cloud stamp ≤ geometry.max_age_s`) and that mask is still fresh | the same mask's costmap re-fused with this geometry |
| Fail-safe timer tick (`fail_safe.check_period_s`) while **no** mask that produced a costmap is fresh (including at startup) | fail-safe grid (§6) |
| Anything else (stale mask arrives, adapter/TF/pipeline error, invalid cloud, stale or unpaired cloud) | nothing; logged |

While the node runs with a valid clock, output never stops for longer than
about `mask_max_age_s + fail_safe.check_period_s` plus processing time:
either fresh masks produce normal output, or the fail-safe timer publishes.
**Prolonged silence therefore means the node itself is not running (or the
ROS clock is 0, e.g. sim time before the first `/clock`),** not that the
environment is clear. A downstream timeout on message arrival detects that;
the watchdog for it is Dev 5's (DEV Dev 5 task 2), not Dev 3's.

## 6. Failure behaviour

| Condition | Dev 3 output |
|---|---|
| Semantic mask stale or missing (including Dev 1 adapter failure, which stops Dev 1 publishing masks) | After the last good costmap ages past `mask_max_age_s`: fail-safe grid on every timer tick. Cells in `fail_safe.roi_*` are lethal, every other cell is unknown, **no cell is free**. (The fail-safe builder still applies `inflation_radius`, but with the default costs every non-ROI cell is unknown (255), which inflation never lowers, so no inflated value appears. This is tested in `tests/test_fail_safe.py`.) |
| Fresh mask but processing fails (no/invalid CameraInfo, TF failure, contract violation) | No output for that mask; the fail-safe takes over once the last good costmap is stale. |
| Geometry disabled (`geometry.enabled: false`) | Semantic-only costmap. |
| Geometry missing, empty, stale (`mask stamp − cloud stamp > geometry.max_age_s`), future-dated, from another camera frame, or failed | Semantic-only costmap for that mask. It is identical to the geometry-disabled output. |
| Geometry while in fail-safe | Not used: geometry is only fused into a fresh mask. The fail-safe grid is unchanged. |
| Recovery (a fresh mask produces a costmap) | Normal output resumes at once; fail-safe ticks stop. |

**Distinguishing a fail-safe grid.** The node publishes no separate flag. A
fail-safe grid is recognisable only by content (ROI lethal, everything else
unknown). The project's degraded-perception signal is Dev 1's
`/ugv/perception_degraded`, consumed by Dev 5 (DEV §3). Dev 4 does not need
to detect the fail-safe: its costs are conservative by construction.

## 7. What Dev 4 can rely on

- A standard `nav_msgs/OccupancyGrid` with the encoding in §3. No Dev 4 or
  planner-specific field is used, and Dev 3 needs no knowledge of the
  planner.
- The frame, geometry and layout rules in §2.
- Free (0) only where the semantic mask saw traversable ground and no
  geometric obstacle exists. Unknown (−1) is never free. Lethal (100) covers
  hazards, geometric obstacles and the fail-safe ROI.
- Only the values 0, 100 and −1: no inflation. The consumer (Nav2
  InflationLayer) must inflate.
- The stamp rules in §4, including "latest arrival is current".
- Continuous output while the node is alive (§5).

## 8. What Dev 4 must not assume

- Final topic names, frame names, resolution, extent, origin or update rate
  (all **PENDING**; configurable, no defaults).
- That the grid is robot-centred or rolling (it is fixed-origin).
- Monotonic stamps under `output_stamp_source: mask` (§4).
- Any inflation or footprint padding in the published costmap. There is
  none; Nav2 owns it.
- A footprint applied to cells. The robot footprint (Dev 5) is validated and
  carried in the pipeline result but **not rasterised**; no project document
  defines footprint rasterisation.
- That a consumer's own unknown handling preserves "unknown ≠ free". For
  example, Nav2 StaticLayer's default `track_unknown_space: false` turns −1
  into free. The consumer must configure this.

## 9. Synthetic testing for Dev 4

Dev 4 can test against synthetic grids in the §3 encoding (0 / 100 / −1,
non-inflated), or generate them with `costmap_core.pipeline.run_costmap_pipeline` (use `result.fused`) plus
`costmap_ros.occupancy_grid.occupancy_grid_from_costmap`. Useful scenarios:
free map; wall with an opening; corridor; unknown block (tests Dev 4's own
unknown policy); an obstacle for Nav2's inflation to act on; a sequence where lethal cells appear or
move; and a fail-safe-shaped grid (lethal rectangle, rest unknown). All grid
sizes and values in such tests are test values, not project values.

## 10. Open items (team decisions)

| Item | Owner per documents |
|---|---|
| Nav2 integration form (StaticLayer / custom layer / Nav2 VoxelLayer for geometry) | not defined: team |
| `output_stamp_source`, and whether stamps must be monotonic | not defined: team |
| Topic names per costmap, frame names, grid geometry, global vs local extents, rolling vs fixed | Dev 3 `config/robots/` (frames with Dev 2) |
| Nav2 InflationLayer parameters (radius, cost scaling) and the footprint they use. Dev 3 no longer inflates the published output. | Dev 3 config (DEV Dev 3 task 4) / Dev 5 footprint |
| Whether the now-unused Dev 3 `inflation_radius` parameter (still required; it only feeds the unpublished `final` and the fail-safe builder) should be retired | team |
| Footprint values and footprint → inflation rule | Dev 5 (values); team (rule) |
| Fail-safe ROI extent, check period, "max-inflate" meaning, owner | not defined: team |
| Publisher durability (volatile vs latched) | not defined: team |
