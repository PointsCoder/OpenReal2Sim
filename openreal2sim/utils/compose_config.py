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
        "bg_mesh_discontinuity_rel": None,  # drop background faces whose vertex depths differ by
                                            # more than this fraction of the depth (0 means no
                                            # filtering). None picks the default for the mode: on
                                            # for the slab's depth layer, off in "depth" mode so
                                            # that mode reproduces the pre-slab background mesh
        "bg_table_mode": "slab",  # "slab": the table is a flat textured slab fitted on the ground
                                  # mask, "depth": the old per-pixel mesh for the whole background
        "bg_table_thickness": 0.03,  # in meters, thickness of the table slab
        "bg_table_margin": 0.0,  # in meters, how far the slab extends past the view frustum where
                                 # the ground mask touches the image border. 0 stops the slab at
                                 # the border; a margin keeps objects from falling off, but it also
                                 # grows the scene AABB that the robot base placement uses
        "bg_table_footprint_mode": "hull",  # how the projected ground mask outline is turned into
                                            # the slab footprint: "hull", "rect" or "outline"
        "bg_table_max_edge": 0.05,  # in meters, max triangle edge of the slab top face; the top is
                                    # flat either way, this only bounds the texture projection error
        "bg_table_simplify_px": 4.0,  # in pixels, outline simplification tolerance (approxPolyDP)
        "bg_table_border_px": 2,  # in pixels, how close to the image border counts as touching it
        "bg_table_keep_depth_layer": False,  # keep whatever the slab does not cover as a per-pixel
                                             # mesh in a second node of the same glb. Off by
                                             # default: that node is collision geometry downstream
        "fast_scene_construction": False,  # whether to use fast (but inaccurate) point cloud registration
        "object_placement_mode": "sam3d",  # "sam3d": place the object with the rotation,
                                           # translation and scale SAM-3D predicted, converted
                                           # into our camera; "pointcloud_fit": the old path that
                                           # fits the scale from the bounding box diagonal, centres
                                           # the object on the point cloud and refines with an ICP
        "table_snap": True,  # translate the object along the ground plane normal so it rests on it
        "table_snap_max_height_fraction": 0.2,  # only snap when the gap to the plane is under this
                                                # fraction of the object height, so a stacked or
                                                # hanging object is not pulled down
        "fdpose_tracking_mode": "normal",  # foundation pose tracking mode: "normal" or "perframe"
        "fdpose_seed_from_registration": True,  # seed the frame-0 foundation pose with the
                                                # registered placement instead of searching a
                                                # global rotation grid, which can flip the object
        "fdpose_placement_from_registration": True,  # keep the registered placement for the
                                                     # exported mesh and use foundation pose only
                                                     # to track the motion across the frames
        "fdpose_est_refine_iter": 5,  # foundation pose: number of iterations for pose estimation refinement
        "fdpose_track_refine_iter": 1,  # foundation pose: number of iterations for pose tracking refinement
        "collision_optimization_mode": "plane",  # collision checking mode: "sdf" or "plane"
        "collision_clearance": 0.002,  # in meters, minimum clearance distance for collision optimization
        "collision_sdf_resolution": 192,  # resolution for SDF grid if using "sdf" mode
    },
    "simulation": {
        "physics": "default",
        "extrinsics": None,
        "goal_offset": 0.005,
        "grasp_idx": -1,
        "grasp_pre": None,
        "grasp_delta": 0.1,
        "traj_key": "fdpose_trajs",  # "fdpose_trajs", "simple_trajs", "hybrid_trajs"
        "manip_object_id": "2",
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
