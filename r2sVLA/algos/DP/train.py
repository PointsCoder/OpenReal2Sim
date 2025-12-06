"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra, pdb
from omegaconf import OmegaConf
import pathlib, yaml
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def get_camera_config_from_data(zarr_path):
    """
    Dynamically get camera config from zarr dataset.
    Returns dict with 'h', 'w', and 'fovy' keys.
    """
    import zarr
    try:
        root = zarr.open(zarr_path, mode='r')
        config = {}
        
        # Method 1: Try to get from meta attributes (stored during processing)
        if 'meta' in root and hasattr(root['meta'], 'attrs'):
            attrs = root['meta'].attrs
            if 'head_camera_h' in attrs and 'head_camera_w' in attrs:
                h = int(attrs['head_camera_h'])
                w = int(attrs['head_camera_w'])
                config['h'] = h
                config['w'] = w
                
                # Get fovy if available
                if 'head_camera_fovy' in attrs:
                    config['fovy'] = float(attrs['head_camera_fovy'])
                    print(f"[Camera Config] Detected from meta: {w}x{h}, fovy={config['fovy']:.2f}°")
                else:
                    print(f"[Camera Config] Detected from meta: {w}x{h} (fovy not available)")
                
                if config:
                    return config
        
        # Method 2: Try to read from actual image data
        if 'data' in root and 'head_camera' in root['data']:
            head_cam = root['data']['head_camera']
            if len(head_cam) > 0:
                first_img = head_cam[0]
                # Handle CHW format (3, H, W)
                if len(first_img.shape) == 3 and first_img.shape[0] == 3:
                    h, w = first_img.shape[1], first_img.shape[2]
                    config['h'] = int(h)
                    config['w'] = int(w)
                    print(f"[Camera Config] Detected from image data: {w}x{h}")
                    return config
    except Exception as e:
        print(f"[Camera Config] Warning: Could not read from zarr: {e}")
    
    return None


def get_camera_config(camera_type, zarr_path=None):
    """
    Get camera config, either from data (preferred) or from config file (fallback).
    
    Args:
        camera_type: Camera type name (for fallback)
        zarr_path: Path to zarr dataset (optional, for dynamic detection)
    """
    # Try to get from data first
    if zarr_path and os.path.exists(zarr_path):
        config = get_camera_config_from_data(zarr_path)
        if config is not None:
            return config
    
    # Fallback to config file
    camera_config_path = os.path.join(parent_directory, "../../task_config/_camera_config.yml")
    
    if not os.path.isfile(camera_config_path):
        raise FileNotFoundError(f"Camera config file not found: {camera_config_path}")
    
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    if camera_type not in args:
        raise KeyError(f"Camera type '{camera_type}' not found in config. Available: {list(args.keys())}")
    
    print(f"[Camera Config] Using from config file: {camera_type}")
    return args[camera_type]


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    head_camera_type = cfg.head_camera_type
    zarr_path = cfg.task.dataset.zarr_path if hasattr(cfg.task, 'dataset') and hasattr(cfg.task.dataset, 'zarr_path') else None
    
    # Get camera config dynamically from data, fallback to config file
    head_camera_cfg = get_camera_config(head_camera_type, zarr_path=zarr_path)
    
    # Ensure we have h and w
    if "h" not in head_camera_cfg or "w" not in head_camera_cfg:
        raise ValueError(f"Camera config missing h or w: {head_camera_cfg}")
    
    cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
    cfg.task.shape_meta.obs.head_cam.shape = [
        3,
        head_camera_cfg["h"],
        head_camera_cfg["w"],
    ]
    OmegaConf.resolve(cfg)
    cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
    cfg.task.shape_meta.obs.head_cam.shape = [
        3,
        head_camera_cfg["h"],
        head_camera_cfg["w"],
    ]
    
    # Log fovy if available
    if "fovy" in head_camera_cfg:
        print(f"[Camera Config] Using fovy: {head_camera_cfg['fovy']:.2f}°")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    print(cfg.task.dataset.zarr_path, cfg.task_name)
    workspace.run()


if __name__ == "__main__":
    main()
