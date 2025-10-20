# Isaac Lab vs ManiSkill: Implementation Comparison

## Summary of Key Differences

This document explains how OpenReal2Sim is implemented differently in Isaac Lab vs ManiSkill, based on code analysis.

---

## 1. Initial Pose Strategy

### Isaac Lab Approach

**In `sim_base.py::reset()` (lines 624-640):**

```python
# Objects are placed at environment origin
env_origins = self.scene.env_origins[env_ids]  # (M,3)
object_pose = torch.zeros((M, 7))
object_pose[:, :3] = env_origins  # Position at [0,0,0] relative to env
object_pose[:, 3] = 1.0  # Identity rotation [1,0,0,0]
self.object_prim.write_root_pose_to_sim(object_pose, env_ids)
```

**Key Points:**

- Objects spawn at **environment origin** `[0,0,0]` (relative to each env)
- With **identity rotation** `[1,0,0,0]`
- Physics simulation naturally settles them onto the background
- The reconstruction provides **mesh geometry**, not spawn poses

### ManiSkill Approach (Our Implementation)

**In `openr2s_ms_env.py::_initialize_episode()`:**

```python
# Objects placed at their reconstructed centers
for obj_id, actor in self.object_actors.items():
    obj_config = self.scene_config.objects[obj_id]
    pos = torch.tensor(obj_config.center)  # Use reconstruction center
    quat = torch.tensor(self.OBJECT_INIT_QUAT)
    actor.set_pose(Pose.create_from_pq(p=pos, q=quat))
```

**Key Points:**

- Objects spawn at **reconstruction centers** from `scene.json`
- With configurable rotation (you're using `[0,1,0,0]`)
- More explicit control over initial placement

**Why Different:** ManiSkill doesn't have the same multi-env offset system, so we explicitly set poses from reconstruction data.

---

## 2. Camera Configuration

### Isaac Lab Approach

**In `sim_env_factory.py::create_camera()` (lines 99-124):**

```python
def create_camera():
    C = _SCENE_CTX.cam_dict
    cam_orientation = tuple(C["cam_orientation"])  # Quaternion from scene.json
    cam_pos = tuple(C["scene_info"]["move_to"])     # Position from scene.json
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(
            pos=cam_pos,           # ABSOLUTE world position
            rot=cam_orientation,   # ABSOLUTE world orientation
            convention="ros"
        ),
        ...
    )
```

**Key Points:**

- Camera uses **absolute world coordinates**
- Position and orientation directly from `scene.json`
- **NOT robot-relative** (it's a fixed world camera)
- Uses ROS convention (same as SAPIEN)

### ManiSkill Approach (Our Implementation)

**Current implementation:**

```python
@property
def _default_sensor_configs(self):
    cam_config = self.scene_config.camera
    extrinsic_matrix = np.array(cam_config.extrinsic_matrix)
    camera_pose_world = opencv_to_sapien_pose(extrinsic_matrix)

    # Using world-frame pose (like Isaac Lab)
    return [
        CameraConfig(
            uid="base_camera",
            pose=camera_pose_world,  # World frame
            ...
        )
    ]
```

**Key Points:**

- Also using **world-frame** camera pose
- Transforms from OpenCV to SAPIEN coordinates
- Fixed camera position (not mounted on robot)

**Why Same:** Both frameworks treat this as a **fixed external camera** observing the scene, not a robot-mounted sensor.

---

## 3. Coordinate Transformation

### The Critical Transform

**From OpenCV to SAPIEN/ROS:**

```
X_sapien = +Z_opencv  (forward)
Y_sapien = -X_opencv  (left)
Z_sapien = -Y_opencv  (up)
```

**Rotation Matrix:**

```python
R = [
    [0,  0,  1],   # X_sapien = +Z_opencv
    [-1, 0,  0],   # Y_sapien = -X_opencv
    [0, -1,  0],   # Z_sapien = -Y_opencv
]
```

### In Isaac Lab

Isaac Lab stores camera pose **already in ROS convention** in `scene.json`:

- `camera_heading_wxyz`: Already in ROS frame
- `camera_position`: Already in ROS frame
- **No transformation needed** during scene loading

### In ManiSkill

We transform during loading because:

- `camera_opencv_to_world`: In OpenCV convention
- Must convert to SAPIEN/ROS convention
- Applied in `opencv_to_sapien_pose()`

---

## 4. Robot Initialization

### Isaac Lab

**In `sim_env_factory.py::create_robot()` (lines 127-133):**

```python
def create_robot():
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = tuple(_SCENE_CTX.robot_pos)  # From config
    robot.init_state.rot = tuple(_SCENE_CTX.robot_rot)  # From config
    return robot
```

**Robot pose comes from:**

1. Configuration file or
2. `robot_placement_candidates_v2()` (auto-placement based on scene bounds)

### ManiSkill

**In `openr2s_ms_env.py::_load_agent()`:**

```python
def _load_agent(self, options: dict):
    super()._load_agent(options, sapien.Pose(p=self.ROBOT_BASE_POS))
```

**Robot pose:**

- Hard-coded constant: `[-0.6, 1.0, 0.0]`
- Can be made configurable later
- Fixed for all episodes (deterministic)

---

## 5. Background & Objects

### Isaac Lab

**Background:**

- Loaded as USD file from `scene.json`
- **Kinematic** (static, disable_gravity=True)
- Has collision

**Objects:**

- Loaded as USD files
- **Dynamic** (affected by physics)
- Spawn at env origin, settle onto background

### ManiSkill

**Background:**

- Loaded as GLB file from `scene.json`
- **Static** (using `build_static()`)
- Has collision
- Positioned at `[0, 0, 0.2]` with 180° Y-rotation

**Objects:**

- Loaded as GLB files
- **Dynamic** (using `build()`)
- Positioned at reconstruction centers
- With 180° Y-rotation

**Why Y-rotation?** You added `OBJECT_INIT_QUAT = [0, 1, 0, 0]` which is a 180° rotation around Y-axis. This might be to match the orientation from reconstruction.

---

## 6. Physics Properties

### Isaac Lab

**Background:**

```python
mass: 100.0
disable_gravity: True
kinematic_enabled: True
collision_enabled: True
```

**Objects:**

```python
mass: 0.5
disable_gravity: False
kinematic_enabled: False
collision_enabled: True
```

### ManiSkill

**Background:**

- Built with `build_static()` (similar to kinematic)
- Collision enabled
- Not affected by gravity

**Objects:**

- Built with `build()` (dynamic actor)
- Full physics enabled
- Affected by gravity

**Similar behavior**, just different API calls.

---

## 7. Key Architectural Differences

| Aspect           | Isaac Lab                  | ManiSkill                           |
| ---------------- | -------------------------- | ----------------------------------- |
| **Framework**    | Isaac Sim (Omniverse)      | SAPIEN                              |
| **Assets**       | USD files                  | GLB files                           |
| **Multi-Env**    | `env_origins` offset       | Single scene (num_envs=1 typically) |
| **Camera**       | World-frame, absolute      | World-frame, absolute (same!)       |
| **Object Init**  | At env origin              | At reconstruction centers           |
| **Coord System** | ROS (pre-converted)        | ROS (convert from OpenCV)           |
| **Reset Logic**  | `write_root_pose_to_sim()` | `actor.set_pose()`                  |

---

## 8. Your Current Configuration

Looking at your constants in `openr2s_ms_env.py`:

```python
ROBOT_BASE_POS = [-0.6, 1.0, 0.0]  # Robot at x=-0.6, y=1.0
ROBOT_BASE_QUAT = [1, 0, 0, 0]      # No rotation
OBJECT_INIT_QUAT = [0, 1, 0, 0]     # 180° around Y
OBJECT_INIT_HEIGHT = 0.6            # Objects at z=0.6
BACKGROUND_INIT_POS = [0, 0, 0.2]   # Background at z=0.2
BACKGROUND_INIT_QUAT = [0, 1, 0, 0] # 180° around Y
```

**Questions to Consider:**

1. **Why 180° Y-rotation for objects/background?**

   - Is this to match the reconstruction orientation?
   - Or to flip the mesh coordinate system?

2. **Why robot at y=1.0?**

   - Isaac Lab uses y=0.0
   - Is this to position robot differently relative to scene?

3. **Why background at z=0.2?**
   - Isaac Lab uses z from ground plane
   - This lifts the background up

---

## 9. Recommendations

### For Object Initialization

**Option A: Match Isaac Lab (Physics-based)**

```python
# Place objects at origin, let physics settle
pos = torch.zeros(3, device=self.device)
quat = torch.tensor([1, 0, 0, 0], device=self.device)
```

**Option B: Use Reconstruction (Deterministic)**

```python
# Place objects at their reconstructed positions
pos = torch.tensor(obj_config.center, device=self.device)
quat = torch.tensor([1, 0, 0, 0], device=self.device)  # Or from reconstruction
```

**Your Current:** Option B with custom rotation and height override

### For Camera

**Current approach is correct:**

- World-frame pose (matches Isaac Lab)
- OpenCV → SAPIEN transformation applied
- No robot-relative transform needed

### For Background

**Consider:**

- Why the 180° rotation? Is this from the reconstruction pipeline?
- The z=0.2 offset - does this align with your scene geometry?

---

## 10. Debugging Camera View

To verify camera is correct, compare:

**Isaac Lab camera position** (from `scene.json`):

```json
"camera_position": [-0.023, -0.014, 0.614]
```

**Your ManiSkill camera** should be at the **same position** after transformation.

You can verify by checking the `camera_pose_world` output.

---

## Conclusion

**Main Differences:**

1. **Object poses**: Isaac Lab uses env origins, you use reconstruction centers
2. **180° rotations**: You added Y-axis rotations - verify if needed
3. **Background z-offset**: You use 0.2, Isaac Lab uses ground plane height
4. **Camera**: Both use world-frame (correct!)

**Action Items:**

1. Verify the 180° rotations are intentional
2. Check if `OBJECT_INIT_HEIGHT = 0.6` matches your scene
3. Confirm `ROBOT_BASE_POS = [-0.6, 1.0, 0.0]` positions robot correctly
4. Test camera view matches expected perspective

Would you like me to help debug any of these aspects?
