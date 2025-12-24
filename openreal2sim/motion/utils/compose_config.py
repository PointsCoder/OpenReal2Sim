
default_config = {
    "gpu": "0",
    "n_points": 5000,
    "keep": None,
    "overwrite": False,
    "vis_pts_per_gripper": 400,
    "affordance_erosion_pixel": 5,
    "traj_key": None,
    "manipulated_oid": None,
    "overwrite": True,
    "num_frames": 7,
    "filter_collisions": False,
}

def compose_configs(key_name: str, config: dict) -> dict:
    ret_key_config = {} 
    local_config = config["local"].get(key_name, {})
    local_config = local_config.get("motion", {})
    global_config = config.get("global", {})
    global_config = global_config.get("motion", {})
    for param in default_config.keys():
        value = local_config.get(param, global_config.get(param, default_config[param]))
        ret_key_config[param] = value
    print(f"[Info] Config for {key_name}: {ret_key_config}")
    return ret_key_config