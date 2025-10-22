### This part is used to generate the affordance of the object.
### Include:
### 1. Grasp point generation.
### 2. Affordance map generation. Mainly for articulated object. This might be further combined with a PartSLIP network.
### TODO: Human annotation.
### The logic we use for grasp point generation here is: we select the frist frame of object-hand contact, compute the contact point, overlay the object and extract the 3D coordinate of the point.
### Be careful that we transfer this point to the first frame using pose matrices and extrinsics.
### TODO: This might be helpful to object selection in the first step.





def find_nearest_point(point, object_mask):
    """
    Given a point and an object binary mask (H,W), return the nearest point in the mask.
    """
    H, W = object_mask.shape
    ys, xs = np.where(object_mask)
    if len(xs) == 0:
        return None
    dists = np.sqrt((xs - point[0]) ** 2 + (ys - point[1]) ** 2)
    min_idx = np.argmin(dists)
    return np.array([xs[min_idx], ys[min_idx]])

def compute_contact_point(kpts_2d, object_bbox_mask, object_mask, return_nearest=False):
    """
    Given 2D keypoints (N,2) and an object binary bbox mask (H,W), return the mean keypoint location [x, y]
    of all keypoints that fall inside the mask.
    """
    kpts_2d = np.asarray(kpts_2d)
    mask = object_bbox_mask
    inside = []
    for kp in kpts_2d:
        x, y = int(round(kp[0])), int(round(kp[1]))
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[y, x]:
                inside.append(kp)
    if len(inside) > 0:
        point = np.mean(np.stack(inside, axis=0), axis=0)
        if object_mask[point[1], point[0]]
            return point
        else:
            if return_nearest:
                return find_nearest_point(point, object_mask)
            else:
                return None
    else:
        if return_nearest:
            return find_nearest_point(np.mean(kpts_2d, axis=0), object_mask)
        else:
            return None


def grasp_point_generation(scene_dict, key_cfg):
    start_frame_idx = scene_dict["recon"]["start_frame_idx"]
    point_2d = None
    i = start_frame_idx
    while point_2d is None and i <= start_frame_idx + key_cfg["simulation"]["max_frame_gap"]:
        kpts_2d = scene_dict["simulation"]["hand_kpts"][i]
        object_bbox_mask = scene_dict['recon']['bbox_mask'][i]
        object_mask = scene_dict['recon']['object_mask'][i]
        point_2d = compute_contact_point(kpts_2d, object_bbox_mask, object_mask)
        i += 1
    if point_2d is None:
        point_2d = compute_contact_point(scene_dict["simulation"]["hand_kpts"][i], scene_dict['recon']['bbox_mask'][i], scene_dict['recon']['object_mask'][i], return_nearest=True)
    intrinsic = scene_dict['intrinsic']
    model_path = key_cfg["simulation"]["model_path"]
    model_pose = scene_dict['recon']['model_pose'][i]
    return point_2d_to_3d(point_2d, model_path, model_pose, intrinsic)
  


def point_2d_to_3d(, model_path, model_pose, intrinsic):








def affordance_generation(keys, key_scene_dicts, key_cfgs):
    base_dir = Path.cwd()
    for key in keys:
        scene_dict = key_scene_dicts[key]
        key_cfg = key_cfgs[key]
        grasp_point = grasp_point_generation(scene_dict, key_cfg)
        scene_dict["simulation"]["grasp_point"] = affordance_map
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)
        return key_scene_dicts
