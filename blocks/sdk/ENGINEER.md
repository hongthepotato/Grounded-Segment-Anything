# Block SDK — Engineer Guide

This guide is for engineers packaging ROS2 nodes as platform blocks. After reading it,
you should be able to create a new block, install it, and have it appear in the canvas
palette without any platform code changes.

---

## 1. Block anatomy

A block is a directory under `blocks/` containing:

```
blocks/my-block/
├── block.json     # Block definition (required)
├── Dockerfile     # Container build spec (required)
└── entrypoint.sh  # ROS2 node launch script (required)
```

The platform auto-discovers any `blocks/{dir}/block.json` on startup.

### block.json — all fields explained

```json
{
  "id": "my-block",
  "version": "1.0.0",
  "display_name": "My Block Name",
  "category": "sensor",
  "hardware_family": "camera",
  "icon": "camera",
  "dockerfile": "./Dockerfile",
  "hardware_detection_hint": "realsense",
  "requires_gpu": false,
  "ports": {
    "input": [
      {
        "id": "image_in",
        "alias": "image_frame",
        "ros2_type": "sensor_msgs/Image",
        "label": "Camera Input",
        "optional": false
      }
    ],
    "output": [
      {
        "id": "image_out",
        "alias": "image_frame",
        "ros2_type": "sensor_msgs/Image",
        "label": "Color Camera"
      }
    ]
  },
  "supervisor_params": [
    {
      "id": "fps",
      "type": "int",
      "label": "Frame rate",
      "default": 30,
      "min": 1,
      "max": 120,
      "units": "fps"
    }
  ],
  "engineer_params": [
    {
      "id": "exposure",
      "type": "int",
      "label": "Exposure (-1 = auto)",
      "default": -1
    }
  ]
}
```

**Field rules:**

| Field | Required | What happens if wrong |
|-------|----------|-----------------------|
| `id` | Yes | Platform ignores block (validation error at startup) |
| `version` | Yes | Graphs cannot pin the block version |
| `display_name` | Yes | Block shows "undefined" in palette |
| `category` | Yes | Block appears in wrong palette group |
| `ports` | Yes | No connectors shown on canvas |
| `hardware_family` | No | Block not grouped with related hardware |
| `requires_gpu` | No | GPU not allocated in compose (block fails at runtime) |

---

## 2. Port alias reference

Wires connect ports with matching aliases. The canvas enforces this at draw time.

| Alias | Compatible ROS2 types | Notes |
|-------|----------------------|-------|
| `image_frame` | `sensor_msgs/Image`, `sensor_msgs/CompressedImage` | Camera output or model input |
| `detections` | `vision_msgs/Detection2DArray` | Raw model detections |
| `filtered_detections` | `vision_msgs/Detection2DArray` | Post-filter — distinct to force intentional filter step |
| `flag_event` | `std_msgs/Bool`, `std_msgs/String` | Trigger / alert signal |
| `annotated_image` | `sensor_msgs/Image` | Visualization only — NOT wirable to image_frame inputs |
| `depth_frame` | `sensor_msgs/PointCloud2`, `sensor_msgs/Image` (16UC1) | Depth / 3D output |
| `joint_state` | `sensor_msgs/JointState` | Arm joint positions (read-only from arm-state) |
| `ee_pose` | `geometry_msgs/PoseStamped` | End-effector pose |
| `force_reading` | `geometry_msgs/WrenchStamped` | Force/torque at end-effector |
| `motion_id` | `std_msgs/String` | Named motion preset trigger (arm-motion input) |
| `motion_complete` | `std_msgs/Bool` | Motion finished signal |
| `gripper_cmd` | `std_msgs/String` | "open" or "close" (arm-gripper input) |
| `grasp_state` | `std_msgs/Bool` | Gripper contact detected |

**Rule:** your block's output port MUST publish one of the ROS2 types listed for its alias.
Publishing a different type will cause runtime topic incompatibility — the canvas won't
catch this (it validates aliases, not ROS2 types).

---

## 3. Hardware family guide

`hardware_family` groups blocks in the palette by shared interface. Current families:

| Family | Port interface | Palette group |
|--------|---------------|---------------|
| `camera` | Outputs: `image_frame` (required), `depth_frame` (optional) | "Camera" |
| `arm` | Split blocks: state outputs joint_state/ee_pose/force_reading; motion inputs motion_id; gripper inputs gripper_cmd | "Robot Arm" |
| `plc` | Inputs: `flag_event`. No standard output — PLC-specific | "PLC / Conveyor" |
| `conveyor` | Inputs: `flag_event`. No standard output | "PLC / Conveyor" |

**Adding a new hardware_family:** Requires a platform PR to register the new alias set
in `platform/registry.py`. A new block within an existing family needs no platform changes.

---

## 4. Dockerfile requirements

Every block Dockerfile MUST:

1. Start with the shared base image:
   ```dockerfile
   FROM host-gateway:5000/ros2-block-base:humble
   # or for GPU blocks:
   FROM host-gateway:5000/ros2-block-base:humble-cuda
   ```

2. Copy and set your `entrypoint.sh`:
   ```dockerfile
   COPY entrypoint.sh /entrypoint.sh
   RUN chmod +x /entrypoint.sh
   ENTRYPOINT ["/entrypoint.sh"]
   ```

3. The entrypoint MUST accept and forward `ROS2_REMAP_ARGS` (see entrypoint.sh template).

4. The block container is launched by the platform — do NOT use `CMD` to launch the node.
   The `ENTRYPOINT` is the only launcher.

**ROS2_REMAP_ARGS contract:**
The launch engine injects this env var containing topic remapping arguments:
```
--ros-args --remap image_in:=/pipeline/abc123/camera/image_out --remap detections_out:=/pipeline/abc123/yolo/detections_in
```
Pass it UNQUOTED in your entrypoint so multiple args expand correctly:
```bash
exec ros2 run my_pkg my_node $ROS2_REMAP_ARGS "$@"
```

**PARAM_* environment variables:**
Every param in `supervisor_params` and `engineer_params` is injected as an env var:
`PARAM_{ID}` (uppercase). Read these in your node:
```python
import os
fps = int(os.environ.get('PARAM_FPS', '30'))
```

---

## 5. Parameter tier guide

| Tier | Who sees it | Rule of thumb |
|------|-------------|---------------|
| `supervisor_params` | Supervisors and engineers | If a supervisor can set this without breaking anything, it goes here. Examples: device index, resolution, confidence threshold, output path. |
| `engineer_params` | Engineers only (behind "Advanced" toggle) | If getting it wrong could crash the node, damage hardware, or violate calibration. Examples: exposure, calibration file path, DDS transport params, PLC addresses. |

When in doubt, start in `engineer_params` and promote to `supervisor_params` after you've
seen supervisors need to change it regularly.

---

## 6. Motion preset registration (arm blocks only)

For `arm-motion` and `arm-gripper` blocks, the platform queries your container's REST API
to get the available motion presets:

```
GET http://localhost:{block_port}/presets
Returns: [{"id": "home", "display_name": "Home position"}, ...]
```

Your block container must serve this endpoint. The platform starts the container, waits
up to 10 seconds, and fetches `/presets`. If the fetch fails, the supervisor sees a
"Could not load presets — retry" error and cannot run the pipeline.

The endpoint should read presets from your arm driver's source of truth — not a static
list in block.json. This ensures the dropdown always reflects what's actually programmed.

Set `"presets_endpoint": true` in your block.json to tell the platform to fetch this.

---

## 7. Testing your block locally

Before installing in the platform:

```bash
# Build the image (assumes local registry is running)
cd blocks/my-block
docker build -t my-block:test .

# Test the entrypoint contract
docker run --rm \
  -e ROS_DOMAIN_ID=99 \
  -e ROS2_REMAP_ARGS="" \
  -e PARAM_FPS=30 \
  my-block:test

# Verify the node starts and subscribes/publishes the expected topics
# In a second terminal:
docker exec -it <container_id> bash -c \
  "source /opt/ros/humble/setup.bash && ros2 topic list"
```

Common failures:
- `ROS2_REMAP_ARGS` quoted → args not split → node gets one malformed arg
- Wrong base image → missing vision_msgs or cv_bridge at import time
- Missing `ENTRYPOINT` → docker run exits immediately

---

## 8. Installing and uninstalling

**Install:**
```bash
cp -r my-block/ /path/to/blocks/my-block/
```
Then click **"Refresh Block Registry"** in the platform engineer settings panel.
The platform validates `block.json`, builds the Docker image (background task with
SSE progress stream), and pushes to the local registry. The block appears in the palette
when the build completes. Build time: 2–15 minutes (shared base layers already cached).

**Uninstall:**
Delete the block directory, then click "Refresh Block Registry".
Graphs using the deleted block show a "Block not found" error on next load.
Existing running pipelines using the block continue until stopped.

---

## 9. Reference blocks

Study these before building your own:

| Block | Category | What it shows |
|-------|----------|---------------|
| `blocks/usb-camera/` | sensor, camera | Simplest camera block — USB UVC driver |
| `blocks/yolo-inference/` | detection | GPU block, multi-input, annotated_image output |
| `blocks/defect-flagging/` | logic | Pure Python ROS2 node, no camera hardware |
| `blocks/conveyor-stop/` | action | Output-only action block with engineer_params |
| `blocks/live-view/` | output | HTTP server inside a ROS2 node |

All 9 default blocks are living reference implementations.

---

## Port alias expansion (adding new aliases)

If your block needs a port type that doesn't exist in the alias table, you need a
platform PR:

1. Add the alias to `ALIAS_COMPAT_MAP` in `platform/registry.py`
2. Add it to `block.schema.json` enum list
3. Add it to the alias table in this document
4. Open a PR with: the new alias name, the ROS2 types it maps to, and at least one
   reference block using it

New aliases expand the platform's capability surface — they require a review because
they affect every future block that uses them.
