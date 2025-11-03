# Physical Scene Reconstruction

This directory provides tools for building a physically interactable scene from images or videos.


## Code Structure

```
openreal2sim/
├── reconstruction
│   ├── tools
│   │   ├── segmentation_annotator.py
│   ├── modules
│   │   ├── background_pixel_inpainting.py
│   │   ├── background_point_inpainting.py
│   │   ├── background_mesh_generation.py
│   │   ├── object_mesh_generation.py
│   │   ├── scenario_construction.py
│   │   ├── scenario_fdpose_optimization.py
│   │   ├── scenario_collision_optimization.py
│   ├── recon_agent.py
│   └── README.md                    
```

## Workflow

### object segmentation

We first need to segment the objects and the ground (for gravity alignment) in the scene.

With the object and ground masks, we can reconstruct the 3D meshes of each object and the background in the following steps.

We use the tool `tools/segmentation_annotator.py` to segment and annotate the video.

### background pixel inpainting

We conduct background pixel inpainting to fill in missing regions in the background after object segmentation.

The inpainted background image will serve as the texture for the background mesh. (See `modules/background_pixel_inpainting.py`)

### background point inpainting

We inpaint the background point cloud to fill in missing regions (by depth estimation or ground plane assumption) and then generate a complete 3D background mesh. (See `modules/background_point_inpainting.py`)

### background mesh generation

We generate the 3D mesh of the background using the inpainted point cloud and texture. This step produces a complete background mesh that can be used in the final scene. (See `modules/background_mesh_generation.py`)

### object mesh generation

We generate the 3D mesh of each object using the segmented masks and input images. This step produces complete object meshes that can be used in the final scene. (See `modules/object_mesh_generation.py`)

### scenario construction

We assemble the object and background meshes together to form the complete scene.

Specifically, we resize and register the object mesh to the correct position in the scene using metric image point cloud. Then, we rotate the scene (background+objects) mesh so the ground plane is aligned with the gravity Z direction. (See `modules/scenario_construction.py`)

### scenario fdpose optimization

We use Foundation Pose to futher optimize the object placement and track the object trajectory in a video. (See `modules/scenario_fdpose_optimization.py`)

### scenario collision optimization

We optimize the object placement to avoid collisions between objects and the background. (See `modules/scenario_collision_optimization.py`)

## How to Use

Make sure you have `scene.pkl` (from preprocess steps) in the `outputs/{key_name}/scene` folder.

We first need to segment objects that needs to be reconstructed. We provide a GUI for this purpose:
```
python openreal2sim/reconstruction/tools/segmentation_annotator.py
```

If everything is ready, run the all-in-one reconstruction script:
```
python openreal2sim/reconstruction/recon_agent.py
```

You can also start from a certain stage using the `--stage` argument:
```
python openreal2sim/reconstruction/recon_agent.py --stage scenario_construction 
```
The `stage` argument can be set to one of the following:
[object_segmentation, background_pixel_inpainting, background_point_inpainting, background_mesh_generation, object_mesh_generation, scenario_construction, scenario_fdpose_optimization]

You can also run individual stages separately, for example:
```
python modules/background_pixel_inpainting.py
```


## Outputs

We will obtain a portable scene assets folder at `outputs/{key_name}/simulation` and meta information at `outputs/{key_name}/simulation/scene.json`, which contains all necessary scene information:

```
pkl file structure:
{
    "images": # [N, H, W, 3], uint8 np array (input images or videos)
    "depths": # [N, H, W], float array (predicted metric depths)
    "intrinsics": # [3, 3] float array (camera fx, fy, cx, cy)
    "extrinsics": # [N, 4, 4] float array (camera to world transform)
    "n_frames": # number of frames N
    "height": # frame height H
    "width": # frame width W
    "recon": {
        "background": # background inpainted image [H, W, 3] uint8 np array
        "foreground": # foreground (first frame) image [H, W, 3] uint8 np array
        "ground_mask": # ground mask [H, W] bool array
        "object_mask": # object mask [H, W] bool array
        "bg_depth": # background inpainted depth [H, W] float array
        "fg_depth": # foreground original depth [H, W] float array
        "bg_pts": # background xyzrgb map [H, W, 6] float array, rgb in [0, 1]
        "fg_pts": # foreground xyzrgb map [H, W, 6] float array, rgb in [0, 1]
        "normal": # ground normal direction [3]

    }
    "objects": {
            "oid": {
                "oid":   # object id,
                "name": # object name,
                "glb": # object glb path,
                "mask": # object mask [H, W] boolean array,
            },
            ...
        }
    "info" : {
        "camera": {
            "camera_heading_wxyz": # camera heading as a quaternion [w,x,y,z],
            "camera_position":     # camera position in world frame [x,y,z],
            "camera_opencv_to_world": # camera extrinsics (opencv camera convention to world) as a flattened 4x4 matrix,
            "width":  # image width,
            "height": # image height,
            "fx":     # focal length x,
            "fy":     # focal length y,
            "cx":     # principal point x,
            "cy":     # principal point y,
        },
        "objects": {  # a list of objects in the scene
            "oid":   {
                    "oid":   # object id,
                    "name": # object name,
                    "object_center": # object center [x,y,z],
                    "object_min":    # object aabb min [x,y,z],
                    "object_max":    # object aabb max [x,y,z],
                    "original":      # original object mesh path,
                    "registered":    # registered object mesh path,
                    "fdpose":      # object placement at using foundation pose estimation [glb],
                    "fdpose_trajs": # object relative trajs [N,4,4],
                    "optimized":      # object placement after collision optimization [glb],
                },
            ...
        },
        "background": {
            "original":   # original background mesh path,
            "registered": # registered background mesh path,
        },
        "aabb": {
            "scene_min": # scene aabb min [x,y,z],
            "scene_max": # scene aabb max [x,y,z],
        },
        "scene_mesh": {
            "registered": # registered scene mesh path,
            "fdpose": # scene with fdpose placed object meshes [glb],
            "optimized": # entire scene with collision optimized object meshes [glb],
        },
        "groundplane_in_cam": {
            "point":  # a point on the ground plane [x,y,z],
            "normal": # the normal of the ground plane [x,y,z],
        },
        "groundplane_in_sim": {
            "point":  # a point on the ground plane [x,y,z],
            "normal": # the normal of the ground plane [x,y,z],
        }
    }
}
```