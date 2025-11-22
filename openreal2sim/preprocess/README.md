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

### Record3D Usage

Record3D is an iOS app that can record rgb, depth and camera info. If you want to use Record3D to record the data, please make sure that the image width is larger than image height, and export the data as r3d format in the library section of the app. Then, put the .r3d file under the data folder.

In our practice, we find that the depth and camera info may be too noisy to use for reconstruction. Therefore, we only use the rgb images and the first frame depth. The first frame depth is fed into PromptDA for resolution alignment.


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
python openreal2sim/preprocess/preprocess_manager.py
```

The pipeline can be executed with a custom config file:
```
python openreal2sim/preprocess/preprocess_manager.py --config <path_to_config_file>
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
