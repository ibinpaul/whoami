# Dev 3 handoff: costmap subsystem (`ugv_navigation`)

State of the Dev 3 work as it is in the repository. The detailed output
contract for consumers is `DEV3_DEV4_INTERFACE.md`; this file does not repeat
it. Nothing here sets a real robot, camera, TF or grid value: every such
value is **PENDING** and has no default in code.

## 1. What Dev 3 implemented

- **`costmap_core/`** is ROS-independent; a test (`test_core_ros_independence.py`)
  forbids ROS and `ugv_perception` imports. It contains:
  - input contracts (`contracts.py`) and grid geometry (`grid.py`);
  - class → cost mapping (`class_to_cost.py`, the only place cost values live);
  - pinhole ground projection of the mask (`projection.py`, `mask_projection.py`)
    and the semantic costmap (`semantic_costmap.py`);
  - depth point cloud transform, obstacle filter and rasterisation
    (`point_cloud_*.py`), and the geometry costmap (`geometry_costmap.py`);
  - fusion with geometry-lethal precedence (`costmap_fusion.py`), inflation
    (`inflation.py`), footprint validation and padding (`footprint.py`) and the
    stale-mask fail-safe grid (`fail_safe.py`);
  - `pipeline.py`, which only connects the stages.
- **`costmap_ros/`** is the ROS 2 layer:
  - adapters that turn messages into the core dataclasses: mask, CameraInfo,
    PointCloud2, TF, grid parameters, and PortMeta (the PortMeta adapter
    exists but is not used by the node; see §13);
  - OccupancyGrid output;
  - `costmap_node.py`: parameters, subscriptions, the publisher and the
    fail-safe timer.
- **Packaging:** `launch/costmap.launch.py` and
  `config/robots/costmap_node.template.yaml`.

## 2. ROS interfaces (`costmap_node`)

| Direction | Topic (parameter) | Type | QoS |
|---|---|---|---|
| Subscribes | `mask_topic` (Dev 1: `/segmentation/mask`) | `sensor_msgs/Image`, `mono8` | depth 1 (only the newest mask matters) |
| Subscribes | `camera_info_topic` | `sensor_msgs/CameraInfo` | RELIABLE, KEEP_LAST 1, durability = `camera_info_durability` (`volatile` or `transient_local`, required) |
| Subscribes, only if `geometry.enabled` | `geometry.depth_cloud_topic` (Dev 1: `/perception/depth_cloud`) | `sensor_msgs/PointCloud2` | KEEP_LAST 1, RELIABLE, VOLATILE (compatible with Dev 1's publisher; only the newest cloud is kept) |
| Subscribes | TF (`/tf`, `/tf_static`) via a `tf2_ros` listener | — | tf2 defaults |
| Publishes | `costmap_topic` | `nav_msgs/OccupancyGrid` | depth 10, RELIABLE, VOLATILE (not latched) |

**Not subscribed:**
- `/segmentation/port_meta`: its format and how to pair it with a mask are
  PENDING with Dev 1. For now, receiving a mask is treated as the validity
  signal.
- `/segmentation/confidence` and `/ugv/perception_degraded`.

**Mask:**
- `header.stamp` (the image capture time) and `header.frame_id` (the camera
  optical frame) are used unchanged. The frame name is never hard-coded.
- Freshness rule: `now − stamp` must be between 0 and `mask_max_age_s`.
  Stale masks and masks stamped in the future are rejected.

**CameraInfo:**
- The latest valid message is used for each mask. Its stamp is **not**
  compared with the mask's stamp.
- Its content is checked for every mask: `K` must be finite, not a
  placeholder, with zero skew and fx, fy > 0; its `frame_id` and its
  width/height must equal the mask's.
- Distortion (`D`) is **not** used: the core is pure pinhole.

## 3. Semantic class contract

| Mask value | Class | Internal cost | OccupancyGrid |
|---|---|---|---|
| 0 | UNKNOWN | 255 | −1 |
| 1 | TRAVERSABLE | 0 | 0 |
| 2 | HAZARD | 254 | 100 |

- Any other value is rejected (`InvalidSemanticClassError`). It is never
  converted to a safe value.
- Unknown is never free. Cells that no mask pixel reaches stay unknown.
- When several pixels fall in one cell, the order of precedence is
  hazard > unknown > traversable.
- A mostly-unknown mask is still a valid, current sample.
- Dev 3 does not use RUGD or any model's native class IDs.

## 4. Geometry / depth integration

- Geometry is optional (`geometry.enabled`). Each PointCloud2 is processed
  in this order:
  1. TF lookup: `target_frame` ← cloud frame, at the cloud's own stamp;
  2. height/range filter (`geometry.min/max_height`, `min/max_range`);
  3. rasterisation: occupied cells become lethal.
- **Pairing rule:** a cloud is fused with a mask only if it comes from the
  same camera frame and `0 ≤ mask stamp − cloud stamp ≤ geometry.max_age_s`.
  A cloud stamped after the mask is kept for a later mask.
- Dev 1 publishes the cloud **after** the mask of the same image. When the
  matching cloud arrives, the last mask is fused again and the costmap is
  republished, as long as that mask is still fresh.
- Geometry lethal always wins over semantics. Geometry that is missing,
  empty, stale or has failed never clears anything; the output is then the
  semantic-only costmap.

## 5. TF requirements

- The node needs a TF transform from `target_frame` to the mask's
  `header.frame_id` at the mask's stamp. With geometry enabled, it also needs
  `target_frame` ← cloud frame at the cloud's stamp.
- Each lookup waits at most `tf_timeout_s`. There is no retry and no
  fallback pose. If the lookup fails, that mask is skipped.
- `target_frame` must have **z = 0 on the ground plane** (flat-ground model),
  and the camera must be above it (z > 0).
- Blocked on Dev 2 / Dev 5:
  - the TF tree (`map → odom → base_link` and the camera extrinsic);
  - the frame names;
  - the allowed mismatch between mask, TF and occupancy stamps.

## 6. Costmap / OccupancyGrid output

- The published grid is `CostmapPipelineResult.fused`: semantic costs plus
  geometry lethal, **not inflated**. `data` contains only 0, 100 and −1.
- `header.frame_id` = `target_frame`. The grid has a **fixed origin** (not
  rolling). `info.origin` is the min-x, min-y corner of cell (0, 0), and
  `data` is row-major.
- **Stamp:**
  - normal output: `output_stamp_source` = `mask` (the mask's stamp) or
    `now` (the node clock);
  - fail-safe output: always the node clock.
  - With `mask`, stamps are not monotonic, so consumers should treat the
    latest arrival as current.
- A mask that cannot be processed is logged and skipped; nothing is
  published for it.
- Full details are in `DEV3_DEV4_INTERFACE.md` §§1–5.

## 7. Fail-safe (architecture §8.6)

- A timer runs every `fail_safe.check_period_s`. If no mask that produced a
  costmap is fresh, the node publishes the fail-safe grid on every tick. This
  includes startup.
- **Fail-safe grid:** cells in the `fail_safe.roi_*` rectangle (metres in
  `target_frame`, which must lie inside the grid) are lethal, every other
  cell is unknown, and no cell is free.
- The first fresh mask that produces a costmap ends the fail-safe.
- Geometry is not fused while the node is in fail-safe.
- The node publishes no separate fail-safe flag. `/ugv/perception_degraded`
  (Dev 1) and the safety hold (Dev 5) are not Dev 3's.

## 8. Inflation (current decision)

- **Inflation is Nav2's job**: the published costmap is not inflated.
- `inflation.py` still exists, and `result.final` is still computed but not
  published.
- `inflation_radius` is still a required parameter. It only feeds that
  unpublished result and the fail-safe builder, where it has no visible
  effect because every cell outside the ROI is unknown.
- Whether to retire `inflation_radius` is a team decision.
- The Nav2 InflationLayer parameters and the footprint they use are still
  open.

## 9. Configuration files

**`launch/costmap.launch.py`**
- Declares one launch argument, `params_file`, which is **required and has
  no default**.
- Starts `ugv_navigation/costmap_node` (named `costmap_node`) with that
  params file.
- Usage: `ros2 launch ugv_navigation costmap.launch.py params_file:=<file>.yaml`

**`config/robots/costmap_node.template.yaml`**
- Lists every node parameter, including the conditional `geometry.*` and
  `footprint.*` groups.
- Only three values are filled in: `/segmentation/mask`,
  `/segmentation/camera_info` and `/perception/depth_cloud`.
- Every other value is `[TODO]`, a string array, which is the wrong type for
  every parameter, so the node refuses to start until each one is replaced.
- To use it, copy it for each robot or costmap and fill in the real values.
  **Do not fill in the template itself.**

## 10. Package / build / install status

- `ament_python` package. `setup.py` installs:
  - the Python packages `costmap_core` and `costmap_ros`;
  - the console script `costmap_node` into `lib/ugv_navigation` (set in
    `setup.cfg`);
  - `share/ugv_navigation/launch/*.launch.py`;
  - `share/ugv_navigation/config/robots/*.yaml`;
  - `package.xml` and the ament resource marker.
- `package.xml` exec dependencies: numpy, `geometry_msgs`, `launch`,
  `launch_ros`, `nav_msgs`, `rclpy`, `sensor_msgs`, `std_msgs`, `tf2_ros_py`.
- **Verified on ROS 2 lyrical:**
  - A clean `colcon build --packages-select ugv_navigation` succeeds.
  - The executable, launch file and template are all installed.
  - `ros2 pkg executables` lists `costmap_node`.
- **Verified launch behaviour:**
  - The launch fails without `params_file`.
  - The node exits with `ParameterUninitializedException` when it has no
    parameters, and with `InvalidParameterTypeException` when given the
    unfilled template.
  - With a scratch params file of synthetic values (kept outside the repo),
    the node started, the parameter dump matched the file, the fail-safe grid
    was published, and the node shut down cleanly on SIGINT.
- `maintainer` and `license` in `setup.py` and `package.xml` are still `TODO`.

## 11. Tests

- `python3 -m pytest ugv_navigation/tests` with ROS lyrical sourced:
  **1681 passed, 0 failed, 0 skipped**, from 35 `test_*.py` files.
- ROS-dependent tests call `pytest.importorskip` and are skipped when ROS is
  not installed.
- `DEV3_DEV4_INTERFACE.md` still quotes an older count (1673).

## 12. Remaining dependencies

| Owner | Needed by Dev 3 |
|---|---|
| **Dev 1** | Final `PortMeta` format (header or not) and how to pair it with a mask. Whether the mask is computed on a rectified image (Dev 3 is pure pinhole; Dev 1 defaults to `/camera/image_raw`). The RUGD confidence thresholds are not recalibrated (expect more class-0 pixels). |
| **Dev 2** | The TF tree (`map → odom → base_link`), the costmap frame (`target_frame`) and the camera frame names; the TF lookup policy and the allowed stamp mismatch. |
| **Dev 4** | How the costmap enters Nav2 (StaticLayer, a custom layer, or Nav2 layers); the unknown handling on the consumer side (e.g. StaticLayer's `track_unknown_space` must keep −1 as not free); which costmap topic is the global one and which is the local one. |
| **Dev 5** | Camera driver and real intrinsics / calibration; the camera extrinsic; the robot footprint (`footprint.*`); the `config/robots/` grid resolution, extent and origin, and the inflation values; the safety hold and the watchdog on costmap arrival. |
| **Team** | `camera_info_topic` (`/segmentation/camera_info` or the driver topic) and the matching `camera_info_durability`; `mask_max_age_s` (Dev 1's 0.50 s is not automatically Dev 3's); `geometry.max_age_s` and the obstacle filter limits; `output_stamp_source`; the fail-safe ROI extent, check period and owner; `tf_timeout_s`; `geometry.enabled` and `footprint.enabled`; publisher durability. |

## 13. Known ambiguities for the next developer

- **CameraInfo topic and durability are coupled.** Dev 1's
  `/segmentation/camera_info` is VOLATILE and published only after a mask, so
  it needs `volatile`. A latched driver topic needs `transient_local`. The
  template fills in the topic, but CLAUDE.md still lists that choice as
  PENDING.
- **PortMeta is ignored.** Every received mask is treated as valid. Revisit
  this once a timestamped PortMeta exists.
- **Distortion is ignored.** If masks come from unrectified images,
  projection accuracy degrades. There is no check for this.
- **Fixed-origin grid.** The grid does not move with the robot. A local
  costmap may need a rolling window; that is not implemented.
- **Footprint is carried, not applied.** It is validated and returned in the
  pipeline result, but never drawn into cells.
- **Stamps are not monotonic** under `output_stamp_source: mask` (for
  example, on a fail-safe → recovery transition).
- **`inflation_radius` has no visible effect** on the published output (see §8).
- The mask projection assumes flat ground (z = 0 in `target_frame`).
