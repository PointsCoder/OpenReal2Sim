# Physical Scene Reconstruction

This directory provides tools for building a physically interactable scene from images or videos.


## Code Structure

```
openreal2sim/
├── reconstruction
│   ├── tools
│   │   ├── segmentation_annotator.py        
│   └── README.md                    
```

## Workflow

### object segmentation

We first need to segment the objects and the ground (for gravity alignment) in the scene.

With the object and ground masks, we can reconstruct the 3D meshes of each object and the background in the following steps.

We provide an interactive object segmentation tool using the `tools/segmentation_annotator.py` script.

## Outputs

We will obtain `outputs/{key_name}/scene/scene.pkl` that contains all necessary scene information:

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
}
```

## How to Use

Make sure you have `scene.pkl` (from preprocess steps) in the `outputs/{key_name}/scene` folder.

If everything is ready, run the all-in-one script:
```
bash scripts/running/reconstruction.sh
```

Or run per-step following the instructions below.

**Step-1: object segmentation**

Launch the interactive segmentation tool:
```
# Interactive object segmentation
python openreal2sim/reconstruction/tools/segmentation_annotator.py
```

This will open a local Gradio web UI. Follow the instructions on the webpage to annotate objects and save the results.

After finishing the annotation, you will get updated mask dict in `outputs/{key_name}/scene/scene.pkl`:
```
pkl file structure:
{
    ... # previous keys
    "mask": {
      frame_idx: { # frame index starts from 0
        object_id: { # object id starts from 1
          "name": str, # object name
          "bbox": [x0, y0, x1, y1], # bounding box
          "mask": [H, W] binary np array # object mask
        }
      }
    }
}
```

For debugging purpose, we also save the annotated images in `outputs/{key_name}/annotated_images`.

**Step-7: object tracking and pose estimation**

We use another docker image for this step:
```
docker compose -p "$USER" -f docker/compose.yml run foundationpose
```

Inside the container, run:
```
cd third_party/FoundationPose && bash build_all.sh
```

Then, run the tracking and pose estimation script:
```
python openreal2sim/reconstruction/modules/scenario_fdpose_optimization.py
```