### This part is used to generate the affordance of the object.
### Include:
### 1. Grasp point generation.
### 2. Affordance map generation. Mainly for articulated object. This might be further combined with a PartSLIP network.
### TODO: Human annotation.
### The logic we use for grasp point generation here is: we select the frist frame of object-hand contact, compute the contact point, overlay the object and extract the 3D coordinate of the point.
### Be careful that we transfer this point to the first frame using pose matrices and extrinsics.
### TODO: This might be helpful to object selection in the first step.
import pickle
import pathlib
from pathlib import Path
import logging
import yaml
import os
import sys
import torch
import numpy as np
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    BlendParams,
    DepthRenderer,
)
import trimesh


def render_pytorch3d_rgbd(mesh, K, img_size, device="cuda:0"):
    """
    Given a trimesh mesh (already in camera coordinate system), render using pytorch3d.
    Args:
        mesh: trimesh.Trimesh, should be in camera coordinate system.
        K: (3,3) camera intrinsic matrix, fx, fy, cx, cy.
        img_size: (H, W), output image (and depth) size.
        device: device string
    Returns:
        mask_img: (H,W) uint8 binary mask (1 for foreground, 0 for background)
        depth_img: (H,W) float32 (Z buffer, 0 for background)
    """
    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
            mesh = list(mesh.geometry.values())[0]
        else:
            raise ValueError('mesh is not a valid trimesh.Trimesh!')
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int64, device=device)
    verts_rgb = torch.ones_like(verts)[None] * torch.tensor([[0.7, 0.7, 0.7]], dtype=torch.float32, device=device)
    from pytorch3d.renderer import TexturesVertex
    textures = TexturesVertex(verts_features=verts_rgb)

    mesh_p3d = Meshes(verts=[verts], faces=[faces], textures=textures)

    # Get camera intrinsics
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if isinstance(img_size, int):
        image_size = [[img_size, img_size]]
        H, W = img_size, img_size
    else:
        H, W = img_size
        image_size = [[H, W]]

    # Use pytorch3d's PerspectiveCameras for custom intrinsics
    cameras = PerspectiveCameras(
        device=device,
        focal_length=torch.tensor([[fx, fy]], device=device),
        principal_point=torch.tensor([[cx, cy]], device=device),
        image_size=torch.tensor(image_size, dtype=torch.float32, device=device),
        in_ndc=False,
        R=torch.eye(3, device=device).unsqueeze(0),
        T=torch.zeros(1, 3, device=device),
    )
    
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, 5.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params
        ),
    )
    # Render RGB (used only for mask)
    images = renderer(mesh_p3d)
    # Depth renderer
    class DepthShader(torch.nn.Module):
        def __init__(self, device="cpu"):
            super().__init__()
            self.device = device
        def forward(self, fragments, meshes, **kwargs):
            return fragments.zbuf
    depth_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=DepthShader(device=device)
    )
    depth = depth_renderer(mesh_p3d)[0, ..., 0].detach().cpu().numpy()  # (H, W)
    mask = (depth > 0).astype(np.uint8)

    return mask, depth


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

def get_bbox_mask_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)
    return x_min, x_max, y_min, y_max

def compute_contact_point(kpts_2d, object_mask):
    """
    Given 2D keypoints (N,2) and an object binary bbox mask (H,W), return the mean keypoint location [x, y]
    of all keypoints that fall inside the mask.
    """
    kpts_2d = np.asarray(kpts_2d)
    bbox_mask = get_bbox_mask_from_mask(object_mask)
    inside = []
    for kp in kpts_2d:
        x, y = int(round(kp[0])), int(round(kp[1]))
        if 0 <= y < bbox_mask.shape[0] and 0 <= x < bbox_mask.shape[1]:
            if mask[y, x]:
                inside.append(kp)
    if len(inside) > 0:
        point = np.mean(np.stack(inside, axis=0), axis=0)
        if object_mask[point[1], point[0]]
            return point, True, True
        else:
            return point, True, False
    else:
        return np.mean(kpts_2d, axis=0), False, False

def bbox_to_mask(bbox):
    x, y, w, h = bbox
    mask = np.zeros((h, w))
    mask[y:y+h, x:x+w] = True
    return mask

    
def grasp_point_generation(scene_dict, key_cfg):
    gpu_id = key_cfg["simulation"]["gpu"]
    device = f"cuda:{gpu_id}"
    start_frame_idx = scene_dict["recon"]["start_frame_idx"]
    points_2d = None
    i = start_frame_idx
    manipulated_oid = scene_dict["recon"]["manipulated_oid"]
    if_close_list = []
    is_inside_list = []
    backup_list = []
    first_extrinsic = scene_dict["extrinsics"][0]
    T_c2w = np.array(scene_dict["info"]["camera"]["camera_opencv_to_world"]).astype(np.float64).reshape(4,4) 
    T_w2c = np.linalg.inv(T_c2w)
    model_path = scene_dict["info"]["objects"][manipulated_oid]["optimized"]
    model = trimesh.load(model_path)
    model_first = model.apply_transform(T_w2c)
    K = scene_dict["info"]["camera"]["intrinsics"]
    img_size = scene_dict["height"], scene_dict["width"]
    while i <= start_frame_idx + key_cfg["simulation"]["max_frame_gap"]:
        current_extrinsic = scene_dict["extrinsics"][i]
        transform_from_extrinsic = np.linalg.inv(current_extrinsic) @ first_extrinsic
        model_i = model_first.apply_transform(transform_from_extrinsic)
        mask_img, depth = render_pytorch3d_rgbd(model_i, K, img_size, device=device)
        kpts_2d = scene_dict["simulation"]["hand_kpts"][i]
        point_2d, if_close, is_inside = compute_contact_point(kpts_2d, mask_img)
        if if_close:
            if_close_list.append((point_2d, i))
        elif is_inside:
            is_inside_list.append((point_2d, i))
        else:
            backup_list.append((point_2d, i))
    final_list = if_close_list + is_inside_list + backup_list
    winner_i = final_list[0][1]
    winner_point = final_list[0][0]

    x, y = int(round(winner_point[0])), int(round(winner_point[1]))
    depth = render_pytorch3d_rgbd(model, K, img_size, device=device)[1]
    z = float(depth[y, x])
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    winner_point_3d = np.array([X, Y, z])
    winner_point_3d = T_c2w @first_extrinsic @ np.linalg.inv(scene_dict["extrinsics"][winner_i]) @ winner_point_3d
    return winner_point_3d



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
