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
        "table_rest": True,  # before the snap, turn the support face of the object (the face of
                             # its oriented bounding box nearest the plane) parallel to the plane,
                             # so a tilted object ends up resting instead of touching at one point.
                             # Off restores the plain snap along the normal
        "table_rest_max_rotation_deg": 45.0,  # largest rotation the resting step may ask for
        "table_rest_lift_below_plane": True,  # an object whose lowest vertex is under the plane is
                                              # lifted however deep it had sunk: the height
                                              # fraction above guards against dragging a stacked
                                              # object down, and nothing rests inside a table
        "pose_candidates": True,  # choose the orientation by render-and-compare over the seed and
                                  # its symmetric twins instead of trusting the seed. Off restores
                                  # the placement as the layout predicted it
        "pose_candidate_square_ratio": 0.8,  # footprints at least this square also get the 90
                                             # degree yaws, which doubles the candidate set
        "pose_candidate_max": 8,  # hard cap on the number of candidates scored
        "pose_score_frames": 5,  # how many frames of the clip each candidate is scored on
        "pose_score_mask_iou_min": 0.8,  # frames count only while the propagated mask still
                                         # overlaps the frame-0 mask this much: past that a hand
                                         # has moved the object and the frame says nothing about
                                         # the reconstructed pose
        "pose_score_weight_silhouette": 1.0,  # weight of one minus the mask intersection over union
        "pose_score_weight_colour": 1.0,  # weight of the normalised colour correlation term
        "pose_score_weight_depth": 1.0,  # weight of the depth error against the mega-sam depth
        "pose_score_depth_scale_fraction": 0.25,  # the depth error is read against this fraction of
                                                  # the object diagonal, so that a few centimetres
                                                  # of error cost about as much as a wrong colour
        "pose_refine_mode": "fdpose",  # local refinement of the winning candidate: "fdpose" is the
                                       # FoundationPose refiner seeded with it, "photometric" is a
                                       # pattern search on the same cost, "off" skips it
        "pose_refine_max_rotation_deg": 45.0,  # cap on how far the refinement may turn the object
                                               # away from the candidate it was seeded with
        "pose_refine_max_translation_fraction": 0.5,  # same cap on the shift, as a fraction of the
                                                      # object bounding box diagonal. A refinement
                                                      # is only kept when it also lowers the cost
        "pose_refine_levels": 3,  # pattern search: how many times the step size is halved
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
        "photo_texture": True,  # project the frame-0 photo onto the texture of every object
                                # where its surface is visible from that camera, and keep the
                                # generated texture, gain corrected, everywhere else. Off leaves
                                # the generated texture alone
        "photo_texture_mask_erode_pixels": 2,  # in pixels, how far the object mask is eroded
                                               # before it is sampled, so the part background
                                               # pixels of the silhouette stay out of the texture
        "photo_texture_depth_tolerance": 0.004,  # in meters, how far behind the rendered scene
                                                 # depth a texel may still count as visible
        "photo_texture_max_grazing_deg": 75.0,  # faces seen at a larger angle than this between
                                                # their normal and the view ray keep the generated
                                                # texture: one photo pixel there covers too much
                                                # surface to copy
        "photo_texture_blend_sigma": 1.5,  # in texels, feather of the seam between the projected
                                           # and the generated texture
        "photo_texture_gain_clusters": 4,  # how many colour clusters the generated texture is gain
                                           # corrected to the photo in, so a two tone object, or an
                                           # object the segmentation fused with its neighbours, is
                                           # not corrected with one gain. Each cluster gets one
                                           # luminance gain applied to all three channels, capped so
                                           # it cannot clip. 1 matches the mean brightness over the
                                           # whole visible region instead
        "photo_texture_super_resolution": False,  # upscale the photo 4x before projecting it. Off
                                                  # by default: no checkpoint ships in the image,
                                                  # so this needs a download at run time
        "photo_texture_super_resolution_model": "caidas/swin2SR-classical-sr-x4-64",
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
