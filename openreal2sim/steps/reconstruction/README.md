# Physical Scene Reconstruction

This directory provides tools for building a physically interactable scene from images or videos.


## Code Structure

```
openreal2sim/steps/reconstruction
├── reconstruction.py                # the stage runner
├── steps
│   ├── segmentation_auto.py
│   ├── segmentation_mask_propagation.py
│   ├── background_pixel_inpainting.py
│   ├── background_point_inpainting.py
│   ├── background_mesh_generation.py
│   ├── sam_object_mesh_inference.py
│   ├── sam_object_mesh_generation.py
│   ├── pose_candidate_search.py
│   ├── object_image_fit.py
│   ├── scenario_construction.py
│   ├── scenario_fdpose_optimization.py
│   ├── scenario_collision_optimization.py
│   └── object_photo_texture.py
├── tools
│   └── scene_check.py
├── utils
│   └── viz_object_trajectory.py
└── README.md
```

The segmentation GUI lives in `openreal2sim/gui/segmentation_annotator.py`.

## Workflow

### object segmentation

We first need to segment the objects and the ground (for gravity alignment) in the scene.

With the object and ground masks, we can reconstruct the 3D meshes of each object and the background in the following steps.

We use the GUI `openreal2sim/gui/segmentation_annotator.py` to segment and annotate the video, and `steps/segmentation_mask_propagation.py` to propagate the masks over the frames.

Alternatively, `steps/segmentation_auto.py` does the same step without a human: Grounding DINO with the prompt `all objects.` boxes everything it scores above 0.4, SAM 2.1 masks each box (three candidates, the one filling the box best wins) and the objects are written to `scene.pkl` as `object_1..N`, then propagated. A second Grounding DINO pass with the prompt `table. desk. floor.` writes the `ground` mask that the later steps need for gravity alignment, so no hand annotation is required. It replaces the masks already in `scene.pkl`, together with the `objects` and `recon` entries built from them.

### background pixel inpainting

We conduct background pixel inpainting to fill in missing regions in the background after object segmentation.

The inpainted background image will serve as the texture for the background mesh. (See `steps/background_pixel_inpainting.py`)

### background point inpainting (opt-in)

We inpaint the background point cloud to fill in the region the objects occupied. The background mesh completes that region with the ground plane on its own, so this step is off by default (`bg_point_inpainting: false`). Turn it on to complete the geometry with a monocular depth prediction instead (`bg_completion_mode: moge`), which is what a background that is not a plane needs. (See `steps/background_point_inpainting.py`)

### background mesh generation

We generate the 3D mesh of the background using the inpainted background image as its texture. A plane is fitted on the ground mask, the mask outline is projected onto it and extruded into a flat textured slab (`bg_table_mode: slab`); `bg_table_mode: depth` meshes the whole background per pixel instead. This step produces a complete background mesh that can be used in the final scene. (See `steps/background_mesh_generation.py`)

### object mesh generation

We generate the 3D mesh of each object using the segmented masks and input images. This step produces complete object meshes that can be used in the final scene. (See `steps/sam_object_mesh_generation.py`)

### scenario construction

We assemble the object and background meshes together to form the complete scene.

Specifically, we resize and register the object mesh to the correct position in the scene using metric image point cloud. Then, we rotate the scene (background+objects) mesh so the ground plane is aligned with the gravity Z direction. (See `steps/scenario_construction.py`)

The uniform scale that step applies keeps the aspect ratio of the generated mesh, which is where the objects come out too thick for their footprint. `steps/object_image_fit.py` then fits each object to the frame-0 image: a yaw about the ground normal, the uniform scale, the placement in the ground plane and a thickness factor along the normal, with the lowest vertex held on the plane, minimising one minus the mask overlap plus `image_fit_depth_weight` times the median depth error over the mask. The thickness axis is floored at the height the object's own masked depth cloud reaches (`image_fit_height_floor_percentile`), and skipped altogether when that depth map saturates the miss charge or does not improve when the object is thinned; a fit that ends with less silhouette than the placement it started from is dropped. It is baked into the vertices before foundation pose, which takes the registered mesh as its model, and it is turned off with `image_fit: false`.

### scenario fdpose optimization

We use Foundation Pose to futher optimize the object placement and track the object trajectory in a video. (See `steps/scenario_fdpose_optimization.py`)

### scenario collision optimization

We optimize the object placement to avoid collisions between objects and the background. (See `steps/scenario_collision_optimization.py`)

### object photo texture

The mesh generator paints the whole object, the hidden faces included, so the parts that are in the photo come out blurred and any printing on them is unreadable. Since the placed mesh is registered to the frame-0 camera, every texel can be projected into that photo: where it lands in front of the rest of the scene, inside the object mask and on a face that is not seen edge on, the texel takes the photo pixel it came from. Everywhere else the generated texture is kept, brightness matched to the photo per colour cluster, and the UV gutters are filled from the nearest covered texel. The per-cluster gain is one number applied to all three channels, so a texel keeps its hue; it is capped so it cannot push the cluster into the clipping point, and the step warns and records in `scene.pkl` whenever a cap bites. Only the texture image changes: the glb is written back to the same path with the same geometry, vertex order, UVs and material, and the mesh as the generator textured it is kept next to it as `{oid}_{name}_optimized_sam.glb` and carried into `scene.json` as `optimized_sam`. Set `photo_texture: false` to skip the step. (See `steps/object_photo_texture.py`)

The copied texels are photographed pixels, so they carry the lighting of the room the photo was taken in: shadows, specular highlights and the camera's white balance are baked into the base colour map. Relighting such an object in a simulator double counts that lighting, and the brightness match spreads the photo's exposure onto the generated faces as well. Nothing here de-lights the texture. A simulator that needs an unlit base colour should load `optimized_sam` instead.

### scene check

The last step measures the scene it just exported, per object: the silhouette IoU and centroid error of the optimized mesh rendered from the recorded camera against the frame-0 annotation mask, the thickness ratio (height over convex-hull diameter) of the mesh against the same measure on the depth back-projection of that mask, and a 3 s PyBullet drop over the ground plane and over the background mesh, reported as drift and rotation. It writes `outputs/{key}/simulation/check.json` and a `check.png` figure (photo, render, mask-versus-render overlay) and prints one summary line per object, so a pipeline change can be compared against the baseline in `tasks/agent_vs_pipeline/baseline_check.md`. The step only reads the exported assets: a failure in it is logged and does not fail the key, and `scene_check: false` in the reconstruction config turns it off. (See `tools/scene_check.py`, which also runs on its own with `--key` or `--dir`)

## How to Use

Make sure you have `scene.pkl` (from preprocess steps) in the `outputs/{key}/scene` folder.

We first need to segment objects that needs to be reconstructed. We provide a GUI for this purpose:
```
python openreal2sim/gui/segmentation_annotator.py
```

If everything is ready, run the all-in-one reconstruction script, which loops over every key of the config:
```
python openreal2sim/openreal2sim.py reconstruction
```

You can also run a single key with `--key`, and start from a certain step using the `--stage` argument:
```
python openreal2sim/openreal2sim.py reconstruction --key <key> --stage scenario_construction
```
To segment automatically instead of using the GUI, run the whole pipeline starting from the automatic step (it is not part of the default run):
```
python openreal2sim/openreal2sim.py reconstruction --key <key> --stage auto_segmentation
```
or run it on its own, propagation included:
```
python openreal2sim/steps/reconstruction/steps/segmentation_auto.py --key <key>
```
The `stage` argument can be set to one of the following:
[auto_segmentation, mask_propagation, background_pixel_inpainting, background_point_inpainting, background_mesh_generation, object_mesh_generation, scenario_construction, scenario_fdpose_optimization, scenario_collision_optimization, object_photo_texture, save_assets, scene_check]

The default run is the list without `auto_segmentation` and without `background_point_inpainting`: the first is selected by name only, the second also runs when the config turns `bg_point_inpainting` on.

You can also run individual stages separately, for example:
```
python openreal2sim/steps/reconstruction/steps/background_pixel_inpainting.py
```


## Outputs

We will obtain a portable scene assets folder at `outputs/{key}/simulation` and meta information at `outputs/{key}/simulation/scene.json`, which contains all necessary scene information:

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
        "fg_depth": # foreground original depth [H, W] float array
        # the three maps below come from the opt-in point inpainting step only; the
        # background mesh and the scenario construction rebuild them when they are absent
        "bg_depth": # background inpainted depth [H, W] float array
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
                    "optimized_sam":  # the same mesh with the generated texture, before the photo
                                      # was projected onto it [glb], null if that step did not run,
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