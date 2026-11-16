"""Compose the per key configuration of a stage from the global config file."""

default_configs = {
    "preprocess": {
        "fps": 25,
        "quality": 1,
        "resize": 1,
        "depth_refinement": True,
        "gpu": "0",
        "fx": None,
        "fy": None,
        "cx": None,
        "cy": None,
        "depth_scale": None,
        "depth_min": None,
        "depth_max": None,
    },
    "reconstruction": {
        "gpu": "0",
        "obj_dilate_pixels": 8,
        "ground_dilate_pixels": 8,
        "bg_completion_mode": "plane",  # "plane" or "moge"
        "bg_mesh_simplify_step": 1,  # 1 means no background mesh simplification
        "bg_mesh_target_faces": 0,  # 0 means no background mesh simplification
        "bg_mesh_thickness": 0.1,  # in meters, added thickness of the background mesh
        "fast_scene_construction": False,  # whether to use fast (but inaccurate) point cloud registration
        "fdpose_tracking_mode": "normal",  # foundation pose tracking mode: "normal" or "perframe"
        "fdpose_est_refine_iter": 5,  # foundation pose: number of iterations for pose estimation refinement
        "fdpose_track_refine_iter": 1,  # foundation pose: number of iterations for pose tracking refinement
        "collision_optimization_mode": "plane",  # collision checking mode: "sdf" or "plane"
        "collision_clearance": 0.002,  # in meters, minimum clearance distance for collision optimization
        "collision_sdf_resolution": 192,  # resolution for SDF grid if using "sdf" mode
    },
    "motion": {
        "gpu": "0",
        "n_points": 5000,
        "keep": None,
        "overwrite": True,
        "vis_pts_per_gripper": 400,
        "affordance_erosion_pixel": 5,
        "traj_key": None,
        "manipulated_oid": None,
        "num_frames": 7,
        "filter_collisions": True,
    },
}


def compose_configs(key: str, config: dict, stage: str) -> dict:
    default_config = default_configs[stage]
    ret_key_config = {}
    local_config = config["local"].get(key, {})
    local_config = local_config.get(stage, {})
    global_config = config.get("global", {})
    global_config = global_config.get(stage, {})
    for param in default_config.keys():
        value = local_config.get(param, global_config.get(param, default_config[param]))
        ret_key_config[param] = value
    print(f"[Info] Config for {key}: {ret_key_config}")
    return ret_key_config
