
default_config = {
    "fps": 25,
    "quality": 1,
    "resize": 1,
    "depth_refinement": True
}

def compose_configs(key_name: str, config: dict) -> dict:
    ret_key_config = {}
    local_config = config["local"].get(key_name, {})
    local_config = local_config.get("preprocess", {})
    global_config = config.get("global", {})
    global_config = global_config.get("preprocess", {})
    for param in default_config.keys():
        value = local_config.get(param, global_config.get(param, default_config[param]))
        ret_key_config[param] = value
    print(f"[Info] Config for {key_name}: {ret_key_config}")
    return ret_key_config