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
import cv2
import imageio
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    BlendParams,
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
    # Note: R and T are camera-to-world transforms (identity for camera coordinate system)
    cameras = PerspectiveCameras(
        device=device,
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        image_size=torch.tensor(image_size, dtype=torch.float32, device=device),
        in_ndc=False,
        R=torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0),
        T=torch.zeros(1, 3, dtype=torch.float32, device=device),
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
    img = renderer(mesh_p3d).detach().cpu().numpy()[0][:,:,:3].clip(0, 1) * 255
    img = img.astype(np.uint8)
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
    return img, mask, depth


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
    Given 2D keypoints (N,2) and an object binary mask (H,W), return the mean keypoint location [x, y]
    of all keypoints that fall inside the mask.
    """
    kpts_2d = np.asarray(kpts_2d)
    H, W = object_mask.shape
    inside = []
    for kp in kpts_2d:
        x, y = int(round(kp[0])), int(round(kp[1]))
        if 0 <= y < H and 0 <= x < W:
            if object_mask[y, x]:
                inside.append(kp)
    if len(inside) > 0:
        point = np.mean(np.stack(inside, axis=0), axis=0)
        px, py = int(round(point[0])), int(round(point[1]))
        if 0 <= py < H and 0 <= px < W and object_mask[py, px]:
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


def visualize_grasp_points(image, kpts_2d, contact_point, output_path):
    """
    Simple visualization of fingertip keypoints and contact point on image.
    
    Args:
        image: Input image (H, W, 3) uint8 or float
        kpts_2d: Fingertip keypoints array (N, 2)
        contact_point: Contact point (2,)
        output_path: Path to save visualization
    """
    # Prepare image
    vis_image = image.copy()
    if vis_image.dtype != np.uint8:
        if vis_image.max() <= 1.0:
            vis_image = (vis_image * 255).astype(np.uint8)
        else:
            vis_image = vis_image.astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    if vis_image.shape[-1] == 3:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    
    H, W = vis_image.shape[:2]
    
    # Draw fingertip keypoints (blue)
    kpts_2d = np.asarray(kpts_2d)
    for kp in kpts_2d:
        kp_x, kp_y = int(round(kp[0])), int(round(kp[1]))
        if 0 <= kp_y < H and 0 <= kp_x < W:
            cv2.circle(vis_image, (kp_x, kp_y), 5, (255, 0, 0), -1)  # Blue filled circle
    
    # Draw contact point (red)
    cp_x, cp_y = int(round(contact_point[0])), int(round(contact_point[1]))
    if 0 <= cp_y < H and 0 <= cp_x < W:
        cv2.circle(vis_image, (cp_x, cp_y), 8, (0, 0, 255), -1)  # Red filled circle
    
    # Convert back to RGB and save
    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    imageio.imwrite(str(output_path), vis_image_rgb)
    logging.info(f"Saved grasp point visualization to: {output_path}")

    
def grasp_point_generation(keys, key_scene_dicts, key_cfgs):
    base_dir = Path.cwd()
    for key in keys:
        scene_dict = key_scene_dicts[key]
        scene_dict["key"] = key  # Store key in scene_dict for visualization
        key_cfg = key_cfgs[key]
        single_grasp_point_generation(scene_dict, key_cfg, key, base_dir)
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)

def single_grasp_point_generation(scene_dict, key_cfg, key, base_dir):
    gpu_id = key_cfg["gpu"]
    device = f"cuda:{gpu_id}"
    start_frame_idx = scene_dict["info"]["start_frame_idx"]
    points_2d = None    
    i = start_frame_idx
    manipulated_oid = scene_dict["info"]["manipulated_oid"]
    if_close_list = []
    is_inside_list = []
    backup_list = []
    extrinsics = np.array(scene_dict["extrinsics"]).astype(np.float64).reshape(-1, 4, 4)
    T_c2w = np.array(scene_dict["info"]["camera"]["camera_opencv_to_world"]).astype(np.float64).reshape(4,4) 
    T_w2c = np.linalg.inv(T_c2w)
    model_path = scene_dict["info"]["objects"][manipulated_oid]["optimized"]
    model = trimesh.load(model_path)
    
    K = np.array(scene_dict["intrinsics"]).astype(np.float32).reshape(3, 3)
    img_size = scene_dict["height"], scene_dict["width"]

    traj_key = scene_dict["info"]["traj_key"]
    traj_key = traj_key.replace("_recomputed", "")
    traj_path = scene_dict["info"]["objects"][manipulated_oid][traj_key]
    traj = np.load(traj_path)
    traj = traj.reshape(-1, 4, 4)
    start_pose = traj[start_frame_idx]
    model = model.apply_transform(start_pose)
    model = model.apply_transform(T_w2c)

    
    images, mask_img, depth = render_pytorch3d_rgbd(model, K, img_size, device=device)
    kpts_2d = scene_dict["simulation"]["hand_kpts"][i][[4,8,12,16,20]]
    point_2d, is_close, is_inside = compute_contact_point(kpts_2d, mask_img)

    # Use PyTorch3D's unproject_points for accurate 2D-to-3D conversion
    x, y = int(round(point_2d[0])), int(round(point_2d[1]))
    image = scene_dict["images"][i]
    
    # Visualize keypoints and contact point
    vis_dir = base_dir / f"outputs/{key}/simulation/debug"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_path = vis_dir / f"grasp_point_visualization_frame_{i:06d}.png"
    visualize_grasp_points(image, kpts_2d, point_2d, vis_path)
    mask_save_path = vis_dir / f"mask_frame_{i:06d}.png"
    cv2.imwrite(mask_save_path, mask_img * 255)
    image_save_path = vis_dir / f"image_frame_{i:06d}.png"
    imageio.imwrite(image_save_path, images)
                       
    
    
    # Create camera object for unprojection (same as in render function)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    H, W = img_size
    
    camera = PerspectiveCameras(
        device=device,
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        image_size=torch.tensor([[H, W]], dtype=torch.float32, device=device),
        in_ndc=False,
        R=torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0),
        T=torch.zeros(1, 3, dtype=torch.float32, device=device),
    )
    
    # Unproject 2D point to 3D using PyTorch3D
    xy_coords = torch.tensor([[x, y]], dtype=torch.float32, device=device)  # (1, 2)
    depth_values = torch.tensor([[z]], dtype=torch.float32, device=device)  # (1, 1)
    
    # Unproject to camera space (since mesh is already in camera coordinates)
    points_camera = camera.unproject_points(
        xy_depth=torch.cat([xy_coords, depth_values], dim=1),  # (1, 3) [x, y, depth]
        world_coordinates=False,  # Output in camera coordinates
        scaled_depth_input=False,  # Depth is in world units
    )
    
    winner_point_3d = points_camera[0].cpu().numpy()  # (3,) in camera coordinates

    winner_point_to_obj = winner_point_3d @ np.linalg.inv(start_pose)
    return winner_point_to_obj



if __name__ == "__main__":
    base_dir = Path.cwd()
    cfg_path = base_dir / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.open("r"))
    keys = cfg["keys"]
    sys.path.append(str(base_dir / "openreal2sim" / "simulation"))
    from utils.compose_config import compose_configs
    key_cfgs = {key: compose_configs(key, cfg) for key in keys} 
    print(f"Key cfgs: {key_cfgs}")
    key_scene_dicts = {}
    for key in keys:
        scene_pkl = base_dir / f'outputs/{key}/scene/scene.pkl'
        with open(scene_pkl, 'rb') as f:
            scene_dict = pickle.load(f)
        key_scene_dicts[key] = scene_dict
    grasp_point_generation(keys, key_scene_dicts, key_cfgs)

