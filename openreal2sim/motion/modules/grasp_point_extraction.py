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
import yaml
import os
import sys
import torch
import numpy as np
import cv2
import imageio
import json
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
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
import trimesh


def create_pytorch3d_camera(K, T_c2w, img_size, device="cuda:0"):
    """
    Create a PyTorch3D camera from OpenCV-style camera parameters.
    
    Args:
        K: (3, 3) camera intrinsic matrix
        T_c2w: (4, 4) camera-to-world transformation matrix
        img_size: (H, W) or int, image size
        device: device string
    
    Returns:
        PerspectiveCameras object
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Parse image size
    if isinstance(img_size, int):
        H, W = img_size, img_size
    else:
        H, W = img_size
    image_size = torch.tensor([[H, W]], dtype=torch.float32, device=device)
    
    # Convert camera-to-world to world-to-camera for OpenCV convention
    # OpenCV projection: x_screen = K @ (R_w2c @ x_world + tvec_w2c)
    T_w2c = np.linalg.inv(T_c2w)
    R_w2c = T_w2c[:3, :3]  # world-to-camera rotation
    tvec_w2c = T_w2c[:3, 3]  # world-to-camera translation
    
    # Convert to torch tensors with batch dimension
    R_w2c_torch = torch.tensor(R_w2c, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3, 3)
    tvec_w2c_torch = torch.tensor(tvec_w2c, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3)
    K_torch = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3, 3)
    
    # Create PyTorch3D camera using OpenCV convention
    camera = cameras_from_opencv_projection(
        R=R_w2c_torch,
        tvec=tvec_w2c_torch,
        camera_matrix=K_torch,
        image_size=image_size,
    ) 
    R = camera.R.cpu().numpy()
    R = R.reshape(3, 3)
    T = camera.T.cpu().numpy()
    T = T.reshape(3,)
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    height, width = image_size[0].cpu().numpy()
    camera = PerspectiveCameras(
        device=device,
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        R=[R],
        T=[T],
        image_size=torch.tensor([[height, width]], dtype=torch.float32, device=device),
        in_ndc=False,
    )
    return camera


def render_pytorch3d_rgbd(mesh, K, T_c2w, img_size, device="cuda:0"):
    """
    Given a trimesh mesh (in world coordinate system), render using pytorch3d.
    Args:
        mesh: trimesh.Trimesh, should be in world coordinate system.
        K: (3,3) camera intrinsic matrix, fx, fy, cx, cy.
        T_c2w: (4,4) camera-to-world transformation matrix.
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

    # Create camera using unified helper function
    cameras = create_pytorch3d_camera(K, T_c2w, img_size, device=device)
    
    # Get image dimensions for rasterization
    if isinstance(img_size, int):
        H, W = img_size, img_size
    else:
        H, W = img_size
    
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=8,
        bin_size=1024,
        max_faces_per_bin=1000000,
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


def find_nearest_point(point, object_mask, erosion_size=8):
    """
    Given a point and an object binary mask (H,W), return the nearest point in the mask,
    preferring points deeper inside the mask (away from edges).
    
    Args:
        point: (x, y) point coordinates
        object_mask: (H, W) binary mask
        erosion_size: size of erosion kernel to remove edges (default: 5)
    
    Returns:
        np.array([x, y]) or None if no valid point found
    """
    H, W = object_mask.shape
    # Convert mask to uint8 for cv2 operations
    mask_uint8 = (object_mask * 255).astype(np.uint8)
    
    # Erode the mask to remove edges and get inner region
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
    eroded_mask = (eroded_mask > 0).astype(bool)
    
    # Try to find point in eroded mask first (deeper inside)
    ys, xs = np.where(eroded_mask)
    if len(xs) > 0:
        dists = np.sqrt((xs - point[0]) ** 2 + (ys - point[1]) ** 2)
        min_idx = np.argmin(dists)
        return np.array([xs[min_idx], ys[min_idx]])
    
    # If eroded mask is empty, fall back to original mask
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
            cv2.circle(vis_image, (kp_x, kp_y), 1, (255, 0, 0), -1)  # Blue filled circle
    
    # Draw contact point (red)
    cp_x, cp_y = int(round(contact_point[0])), int(round(contact_point[1]))
    if 0 <= cp_y < H and 0 <= cp_x < W:
        cv2.circle(vis_image, (cp_x, cp_y), 1, (0, 0, 255), -1)  # Red filled circle
    
    # Convert back to RGB and save
    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    imageio.imwrite(str(output_path), vis_image_rgb)
    print(f"[Info] Saved grasp point visualization to: {output_path}")

    
def grasp_point_extraction(keys, key_scene_dicts, key_cfgs):
    base_dir = Path.cwd()
    for key in keys:
        scene_dict = key_scene_dicts[key]
        scene_dict["key"] = key  # Store key in scene_dict for visualization
        key_cfg = key_cfgs[key]
        single_grasp_point_generation(scene_dict, key_cfg, key, base_dir)
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)
    return key_scene_dicts

def single_grasp_point_generation(scene_dict, key_cfg, key, base_dir):
    """Generate grasp point for a single scene."""
    # Setup
    gpu_id = key_cfg["gpu"]
    device = f"cuda:{gpu_id}"
    start_frame_idx = scene_dict["info"]["start_frame_idx"]
    i = start_frame_idx
    manipulated_oid = int(scene_dict["info"]["manipulated_oid"])
    
    # Load camera parameters
    T_c2w = np.array(scene_dict["info"]["camera"]["camera_opencv_to_world"]).astype(np.float64).reshape(4, 4)
    K = np.array(scene_dict["intrinsics"]).astype(np.float32).reshape(3, 3)
    img_size = scene_dict["height"], scene_dict["width"]
    
    # Load object model and trajectory
    model_path = scene_dict["info"]["objects"][manipulated_oid]["optimized"]
    model = trimesh.load(model_path)
    
    traj_key = scene_dict["info"]["traj_key"].replace("_recomputed", "")
    print(f"[Info] Trajectory key: {traj_key}")
    traj_path = scene_dict["info"]["objects"][manipulated_oid][traj_key]
    traj = np.load(traj_path).reshape(-1, 4, 4)
    start_pose = traj[start_frame_idx]
    
    # Transform model to world coordinate system
    model = model.apply_transform(start_pose)
    
    # Render object to get mask and depth
    rendered_image, mask_img, depth = render_pytorch3d_rgbd(model, K, T_c2w, img_size, device=device)
    
    # Find contact point from hand keypoints
    kpts_2d = scene_dict["motion"]["hand_kpts"][i][[4, 8, 12, 16, 20]]
    direction = scene_dict["motion"]["hand_global_orient"][i]
    direction = direction.reshape(3, 3)
    direction_world = T_c2w[:3, :3] @ direction
    point_2d, is_close, is_inside = compute_contact_point(kpts_2d, mask_img)
    if not is_close:
        point_2d = find_nearest_point(point_2d, mask_img, erosion_size=key_cfg["affordance_erosion_pixel"])
        if point_2d is None:
            print(f"[Error] Failed to find valid contact point in mask")
            return None
    
    x, y = int(round(point_2d[0])), int(round(point_2d[1]))
    
    # Visualize keypoints and contact point
    vis_dir = base_dir / f"outputs/{key}/motion/debug"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_path = vis_dir / f"grasp_point_visualization_frame_{i:06d}.png"
    visualize_grasp_points(rendered_image, kpts_2d, point_2d, vis_path)
   
    # Get depth value at the contact point
    z = depth[y, x] if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1] and depth[y, x] > 0 else 0.0
    if z <= 0:
        print(f"[Warning] Invalid depth at contact point ({x}, {y}), using nearest valid depth")
        valid_mask = depth > 0
        if np.any(valid_mask):
            y_coords, x_coords = np.where(valid_mask)
            dists = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
            nearest_idx = np.argmin(dists)
            z = depth[y_coords[nearest_idx], x_coords[nearest_idx]]
        else:
            print(f"[Error] No valid depth found, cannot unproject point")
            return None
    
    # Create camera object for unprojection (reuse the same helper function)
    camera = create_pytorch3d_camera(K, T_c2w, img_size, device=device)
    
    # Unproject 2D point to 3D using PyTorch3D. The rasterized depth is an NDC z-buffer
    # in [0, 1]. Convert pixel coordinates to NDC [-1, 1] before calling unproject.
    if isinstance(img_size, int):
        H = W = img_size
    else:
        H, W = img_size
    xy_coords = torch.tensor([[x, y]], dtype=torch.float32, device=device)
    depth_values = torch.tensor([[z]], dtype=torch.float32, device=device)
    points_camera = camera.unproject_points(
        xy_depth=torch.cat([xy_coords, depth_values], dim=1),
        world_coordinates=False,
        scaled_depth_input=False,
        from_ndc=False,
    )
    winner_point_3d = points_camera[0].cpu().numpy()  # (3,) in camera coordinates
    print(f"[Info] Winner point 3D: {winner_point_3d}")
    # Convert from camera coordinates to object coordinates
    # winner_point_3d is in camera coordinate system
    # start_pose is object pose in world coordinate system
    # To get point in object coordinate system: 
    # 1. camera -> world: T_c2w @ point_cam
    # 2. world -> object: inv(start_pose) @ point_world
    point_cam_homo = np.hstack([winner_point_3d, 1.0])  # (4,)
    point_world = T_c2w @ point_cam_homo 
    point_world = point_world / point_world[3]
    winner_point_to_obj = np.linalg.inv(start_pose) @ point_world 
    winner_point_to_obj = winner_point_to_obj[:3] / winner_point_to_obj[3]
    scene_dict["info"]["objects"][manipulated_oid]["grasp_point"] = winner_point_to_obj.tolist()
    scene_dict["info"]["objects"][manipulated_oid]["grasp_direction"] = direction_world.tolist()

    scene_json_path = base_dir / f"outputs/{key}/simulation/scene.json"
    with open(scene_json_path, "r") as f:
        scene_json = json.load(f)
    scene_json["objects"][str(manipulated_oid)]["grasp_point"] = winner_point_to_obj.tolist()
    scene_json["objects"][str(manipulated_oid)]["grasp_direction"] = direction_world.tolist()
    with open(scene_json_path, "w") as f:
        json.dump(scene_json, f, indent=2)
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
    grasp_point_extraction(keys, key_scene_dicts, key_cfgs)

