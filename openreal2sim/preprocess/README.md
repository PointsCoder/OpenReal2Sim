# Data Preprocessing

This directory contains scripts and instructions for preprocessing raw images or videos into depth maps, camera intrinsics/extrinsics, and dynamic point clouds.

These information recovers the underlying geometry and motion of the scene, which can be further used in the [reconstruction](../reconstruction/README.md) step to construct a physically interactable scene.

## Code Structure

```
openreal2sim/
├── preprocess
│   ├── utils
│   │   ├── depth_to_color.py        
│   │   ├── viz_dynamic_pcd.py       
│   ├── image_extraction.py          
│   ├── depth_prediction.py          
│   ├── depth_calibration.py         
│   └── README.md                    
```

## Workflow

We current support inputs in several modes:

- Single image: we run metric depth estimation (MoGe) for depth and camera estimation
- Single monocular video: we run 4D reconstruction (mega-sam) for video-based depth and camera estimation   
- With ground truth depth image: we align the metric scale of predicted depth to the ground truth depth image

### image_extraction

We extract frames from images or videos, and resize them if needed.

### depth_prediction

Based on the input type (image or video), we obtain depth, camera infos, and dynamic point clouds through different pipelines (mono-depth or 4D reconstruction) respectively

### depth calibration

If we have GT info, e.g., depth image, known intrinsics, calibrate the predicted depth and camera info to GT. This step is optional and is needed only when ground truth image is provided. 


## Outputs

We will obtain `outputs/{key_name}/scene/scene.pkl` that contains all necessary geometry and motion information:

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

We also save intermediate results in `outputs/{key_name}/geometry/` for debugging purpose, e.g., dynamic point clouds.

## How to Use

Make sure you have put input image or video in the `data/` folder like this:
```
data/
├── key_name.png/jpg   # single image mode
├── key_name.mp4       # monocular video mode
├── key_name_depth.png # with GT depth image
```

Then, make sure you have correct configurations for each `key_name` in `config/config.yaml`:
```
keys:
  - key_name

global:
    ...

local:
  key_name:
    preprocess:
      ...
```


If everything is ready, run the all-in-one script:
```
bash scripts/running/preprocess.sh
```

Or run per-step:
```
# Extract frames from images or videos
python openreal2sim/preprocess/image_extraction.py

# Obtain depth, camera infos, and dynamic point clouds from images or videos
python openreal2sim/preprocess/depth_prediction.py

# (Optional) if we have GT info, e.g., depth image, known intrinsics, 
# calibrate the predicted depth and camera info to GT
python openreal2sim/preprocess/depth_calibration.py
```