#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running FoundationPose to estimate initial object pose (for better object placement) + object trajectory (for motion tracking).
Inputs:
    - outputs/{key_name}/scene/scene.pkl (must contain a "objects" key)
Outputs:
    - outputs/{key_name}/scene/scene.pkl (updated)
Note:
    - Updated keys in "info":{
        "objects": { 
            "oid": {
                ...
                "fdpose_trajs": # object relative trajs [N,4,4],
                "simple_trajs":  # trajectory from mask+depth only [N,4,4],
                "hybrid_trajs":  # rotation from fdpose, translation from mask+depth [N,4,4],
                "fdpose":      # object placement at using foundation pose estimation [glb],
            },
            ...
        },
        "scene_mesh": {
            ...
            "fdpose": # entire scene with fdpose placed object meshes [glb],
        }
    }
"""

import os, logging, shutil, sys, pickle
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import cv2, imageio
import trimesh
import yaml

BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / "outputs"
REPO_DIR = str(BASE_DIR / Path('third_party/FoundationPose'))
sys.path.append(REPO_DIR)

# FoundationPose 依赖
from estimater import *  # FoundationPose, ScorePredictor, PoseRefinePredictor, draw_posed_3d_box, draw_xyz_axis, set_seed
import nvdiffrast.torch as dr

# ------------------------- utils -------------------------
def set_logging():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)

def build_mask_array(
    oid: int,
    scene_dict: Dict[str, Any]
) -> np.ndarray:

    H, W, N = scene_dict["height"], scene_dict["width"], scene_dict["n_frames"]
    out = np.zeros((N, H, W), dtype=np.uint8)
    for i in range(N):
        m = scene_dict["mask"][i][oid]["mask"]
        out[i] = m.astype(np.uint8)
    return out

def masked_center_cam(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    method: str = "mean"
) -> Optional[np.ndarray]:
    """
    Compute object center in the camera frame from masked depth.
    Returns p_cam (3,) or None if insufficient valid pixels.

    - Unprojects all valid (u,v,z) to 3D using intrinsics K.
    - Aggregates by mean/median for robustness.
    """
    # Valid pixels: depth > 0 within the binary mask
    valid = (mask.astype(bool)) & (depth > 0)
    if not np.any(valid):
        return None

    v_idx, u_idx = np.where(valid)  # row (y), col (x)
    z_vals = depth[v_idx, u_idx]

    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])

    # Back-project to camera frame
    x = (u_idx.astype(np.float64) - cx) / fx * z_vals
    y = (v_idx.astype(np.float64) - cy) / fy * z_vals
    pts = np.stack([x, y, z_vals.astype(np.float64)], axis=1)  # [M,3]

    if method == "mean":
        center = pts.mean(axis=0)
    else:
        center = np.median(pts, axis=0)
    return center.astype(np.float64)

def nearest_valid_fill_inplace(seq: List[Optional[np.ndarray]]) -> None:
    """
    In-place nearest valid value fill. First forward, then backward pass.
    Each element is either None or a numpy array of identical shape.
    """
    last = None
    for i in range(len(seq)):
        if seq[i] is not None:
            last = seq[i]
        else:
            seq[i] = last
    last = None
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] is not None:
            last = seq[i]
        else:
            seq[i] = last

def convert_glb_to_obj_temp(glb_path: str) -> Tuple[str, Optional[str]]:
    import tempfile
    from PIL import Image
    from trimesh.visual.material import SimpleMaterial
    p = Path(glb_path)
    if p.suffix.lower() not in ['.glb', '.gltf']:
        return glb_path, None
    tmpdir = tempfile.mkdtemp(prefix='fp_glb2obj_')
    obj_path = Path(tmpdir) / 'textured_simple.obj'
    tex_path = Path(tmpdir) / 'texture_map.png'
    m = trimesh.load(str(p), force='mesh')
    mat = getattr(m.visual, 'material', None)
    tex_img = None
    if mat is not None:
        if hasattr(mat, 'image') and mat.image is not None:
            if isinstance(mat.image, Image.Image):
                tex_img = mat.image
            else:
                arr = np.asarray(mat.image)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                tex_img = Image.fromarray(arr)
        elif hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
            bct = mat.baseColorTexture
            if isinstance(bct, Image.Image):
                tex_img = bct
            else:
                arr = np.asarray(bct)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                tex_img = Image.fromarray(arr)
    if tex_img is None:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"[convert_glb_to_obj_temp] GLB lack texture: {glb_path}")
    tex_img.save(str(tex_path))
    m.visual.material = SimpleMaterial(image=str(tex_path))
    m.export(str(obj_path))
    logging.info(f"[convert_glb_to_obj_temp] export to OBJ: {obj_path}")
    return str(obj_path), tmpdir


def export_debug_track_as_video(
    track_vis_dir: Path,
    output_mp4_path: Path,
    fps: int = 20
) -> None:
    """
    Stitch per-frame debug images under `track_vis_dir` into an MP4 video using ffmpeg.
    Assumes frames are zero-padded and in numeric order (e.g., 000000.png/.jpg).
    No backups or fallbacks; simplest pipeline only.

    Args:
        track_vis_dir (Path): Directory containing per-frame images.
        output_mp4_path (Path): Target MP4 path to write.
        fps (int): Frames per second for the output video.
    """
    # Choose extension: prefer .jpg, otherwise .png
    jpg_list = sorted(track_vis_dir.glob("*.jpg"))
    png_list = sorted(track_vis_dir.glob("*.png"))
    ext = "jpg" if len(jpg_list) > 0 else ("png" if len(png_list) > 0 else None)
    if ext is None:
        logging.info(f"[export_debug_track_as_video] No images found in: {track_vis_dir}")
        return

    # Ensure output directory exists
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg to encode H.264 + yuv420p for maximum compatibility (VS Code / browsers)
    # Pattern uses glob; assumes zero-padded names ensure correct lexicographic order.
    import subprocess
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", str(track_vis_dir / f"*.{ext}"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_mp4_path),
    ]
    subprocess.run(cmd, check=True)
    logging.info(f"[export_debug_track_as_video] Wrote video: {output_mp4_path}")

# ------------------------- Reader -------------------------
def resize_rgb(rgb: np.ndarray, W: int, H: int) -> np.ndarray:
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

def resize_depth(depth: np.ndarray, W: int, H: int) -> np.ndarray:
    return cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

def resize_mask(msk: np.ndarray, W: int, H: int) -> np.ndarray:
    return cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

class FoundationPoseReader:
    def __init__(self,
                 imgs: np.ndarray, # (N, H, W, 3) uint8
                 depths: np.ndarray, # (N, H, W) float32
                 Ks: np.ndarray, # (N, 3, 3) float32
                 mask_array: np.ndarray, # (N, H, W) uint8
                 shorter_side: Optional[int] = None,
                 zfar: float = float("inf")):
        assert imgs.ndim == 4 and imgs.shape[-1] == 3
        assert depths.ndim == 3
        assert Ks.ndim == 3 and Ks.shape[1:] == (3,3)
        assert mask_array is not None and mask_array.ndim == 3 and mask_array.shape[0] == imgs.shape[0]
        self.N, self.H0, self.W0 = imgs.shape[0], imgs.shape[1], imgs.shape[2]
        self.downscale = 1.0
        if shorter_side is not None:
            self.downscale = float(shorter_side) / float(min(self.H0, self.W0))
        self.H = int(round(self.H0 * self.downscale))
        self.W = int(round(self.W0 * self.downscale))
        self.colors = np.stack([resize_rgb(imgs[i], self.W, self.H) for i in range(self.N)], axis=0)
        self.depths = np.stack([resize_depth(depths[i], self.W, self.H) for i in range(self.N)], axis=0)
        self.Ks = Ks.copy().astype(np.float32)
        self.Ks[:, :2, :] *= self.downscale
        for i in range(self.N):
            d = self.depths[i]
            d[(d < 1e-3) | (d >= zfar)] = 0.0
            self.depths[i] = d
        self._masks: List[np.ndarray] = []
        for i in range(self.N):
            m = (mask_array[i] > 0).astype(np.uint8)
            m = resize_mask(m, self.W, self.H)
            self._masks.append(m)
        self.id_strs = [f"{i:06d}" for i in range(self.N)]

    def __len__(self) -> int: return self.N
    @property
    def K(self) -> np.ndarray: return self.Ks[0]
    def get_color(self, i: int) -> np.ndarray: return self.colors[i]
    def get_depth(self, i: int) -> np.ndarray: return self.depths[i]
    def get_mask(self, i: int) -> np.ndarray:  return self._masks[i]
    def get_K(self, i: int) -> np.ndarray: return self.Ks[i]


# ------------------------- main function -------------------------
def scenario_fdpose_optimization(keys, key_scene_dicts, key_cfgs):
    base_dir = Path.cwd()
    # foundation pose modules
    scorer  = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx   = dr.RasterizeCudaContext()
    debug_level = 1 # hard code here


    for key in keys:
        scene_dict = key_scene_dicts[key]
        key_cfg = key_cfgs[key]
        pose_tracking_mode = key_cfg["fdpose_tracking_mode"]
        est_refine_iter = key_cfg["fdpose_est_refine_iter"]
        track_refine_iter = key_cfg["fdpose_track_refine_iter"]
        objects = scene_dict["info"]["objects"]
        debug_dir = base_dir / Path(f"outputs/{key}/reconstruction/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        placed_meshes: List[Tuple[str, trimesh.Trimesh]] = []

        for obj_id, obj_ent in objects.items():
            oid   = obj_ent["oid"]
            oname = obj_ent["name"]
            mesh_in_path = obj_ent["registered"]
            mask_array = build_mask_array(oid, scene_dict)

            logging.info(f"[object] oid={oid}, name={oname}")
            logging.info(f"  mesh={mesh_in_path}")
            logging.info(f"  mask(pkl-by-oid)={mask_array.shape}")

            droot = debug_dir / f"{oid}_{oname}_fdpose"
            if debug_level >= 1:
                (droot / "track_vis").mkdir(parents=True, exist_ok=True)
                (droot / "ob_in_cam").mkdir(parents=True, exist_ok=True)

            imgs = scene_dict["images"]
            depths = scene_dict["depths"]
            Ks = scene_dict["intrinsics"].astype(np.float32)
            N = imgs.shape[0]
            Ks = np.repeat(Ks[None, ...], N, axis=0)
            # imageio.imwrite(droot / "track_vis" / f"{0:06d}_mask.png", mask_array[0]*255)
            reader = FoundationPoseReader(
                imgs=imgs, depths=depths, Ks=Ks,
                mask_array=mask_array,
                shorter_side=None,
                zfar=float("inf")
            )

            mesh_path_tmp, tmpdir = convert_glb_to_obj_temp(mesh_in_path)

            mesh = trimesh.load(mesh_path_tmp)
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

            est = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals if mesh.vertex_normals is not None else None,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir=str(droot),
                debug=debug_level,
                glctx=glctx
            )

            all_poses: List[np.ndarray] = []
            pose_prev: Optional[np.ndarray] = None
            for i in range(len(reader)):
                K_i   = reader.Ks[i].astype(np.float64, copy=False)
                color = reader.get_color(i)
                depth = reader.get_depth(i)

                if pose_tracking_mode == "perframe":
                    ob_mask_i = reader.get_mask(i).astype(bool)
                    pose = est.register(K=K_i, rgb=color, depth=depth, ob_mask=ob_mask_i,
                                            iteration=max(1, est_refine_iter))
                else:
                    if i == 0:
                        ob_mask_0 = reader.get_mask(0).astype(bool)
                        pose = est.register(K=K_i, rgb=color, depth=depth,
                                            ob_mask=ob_mask_0, iteration=est_refine_iter)
                    else:
                        pose = est.track_one(rgb=color, depth=depth, K=K_i,
                                             iteration=track_refine_iter)

                all_poses.append(pose.reshape(4,4))
                pose_prev = pose.reshape(4,4)

                if debug_level >= 1:
                    np.savetxt(droot / "ob_in_cam" / f"{i:06d}.txt", pose.reshape(4,4))

                if debug_level >= 1:
                    center_pose = pose @ np.linalg.inv(to_origin)
                    vis = draw_posed_3d_box(K_i, img=color, ob_in_cam=center_pose, bbox=bbox)
                    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K_i,
                                        thickness=3, transparency=0, is_input_rgb=True)
                    imageio.imwrite(droot / "track_vis" / f"{i:06d}.png", vis)
                 
                    if debug_level >= 2:
                        cv2.imshow('foundationpose', vis[..., ::-1])
                        cv2.waitKey(1)

            track_vis_dir = droot / "track_vis"
            genvideo_out = base_dir / f"outputs/{key}/reconstruction/objects/{oid}_{oname}_fdpose.mp4"
            export_debug_track_as_video(track_vis_dir=track_vis_dir, output_mp4_path=genvideo_out, fps=20)

            # record the relative object pose trajectory (w.r.t. frame-0)
            all_poses_np = np.stack(all_poses, axis=0)
            T_c2w = np.array(scene_dict["info"]["camera"]["camera_opencv_to_world"]).astype(np.float64).reshape(4,4) 
            T_obj_w_abs = np.einsum('ij,njk->nik', T_c2w, all_poses_np)   # [N,4,4]
            T0_w_inv    = np.linalg.inv(T_obj_w_abs[0])
            rel_w       = T_obj_w_abs @ T0_w_inv                           # [N,4,4]

            pose_save_path = base_dir / f"outputs/{key}/reconstruction/motions" / f"{oid}_{oname}_trajs.npy"
            pose_save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(pose_save_path, rel_w.astype(np.float32))
            scene_dict["info"]["objects"][obj_id]["fdpose_trajs"] = str(pose_save_path)
            logging.info(f"[object {oid}] relative trajs(rel_world) -> {pose_save_path}")

            # transform the object mesh with the fdpose estimated pose at frame 0
            T_obj_w_0 = (T_c2w @ all_poses_np[0]).astype(np.float64)
            m_base = trimesh.load(mesh_in_path, force='mesh')
            m_fd = m_base.copy(); m_fd.apply_transform(T_obj_w_0)
            fdpose_path = base_dir / f"outputs/{key}/reconstruction/objects" / f"{oid}_{oname}_fdpose.glb"
            m_fd.export(str(fdpose_path))
            scene_dict["info"]["objects"][obj_id]["fdpose"] = str(fdpose_path)
            logging.info(f"[object {oid}] fdpose -> {fdpose_path}")

            placed_meshes.append((f"{oid}_{oname}", m_fd))

            ####  add simple trajs (from mask+depth only) ####
            logging.info(f"[object] oid={oid}, name={oname} - simple trajs from mask+depth only")
            centers_cam: List[Optional[np.ndarray]] = []
            for i in range(N):
                m_i = reader.get_mask(i)
                c_i = masked_center_cam(
                    depth=reader.get_depth(i),
                    mask=m_i,
                    K=reader.get_K(i),
                )
                centers_cam.append(c_i)            

            # Fill in gaps using nearest valid values
            nearest_valid_fill_inplace(centers_cam)
            # camera->world: p_w(t) = R_cw @ p_c(t) + t_w
            R_cw = T_c2w[:3, :3]; t_w = T_c2w[:3, 3]
            pc_stack = np.stack(centers_cam, axis=0).astype(np.float64)  # [N,3]
            pw_stack = (R_cw @ pc_stack.T).T + t_w[None, :]              # [N,3]
            p0_w = pw_stack[0].copy()
            # assemble simple Δ(t): rotation = I, translation = p_w(t) - p_w(0)
            simple_rel = np.repeat(np.eye(4, dtype=np.float32)[None, ...], N, axis=0)  # [N,4,4]
            simple_rel[:, :3, 3] = (pw_stack - p0_w[None, :]).astype(np.float32)

            simple_save_path = base_dir / f"outputs/{key}/reconstruction/motions" / f"{oid}_{oname}_simple_trajs.npy"
            np.save(simple_save_path, simple_rel.astype(np.float32))
            scene_dict["info"]["objects"][obj_id]["simple_trajs"] = str(simple_save_path)
            logging.info(f"[object {oid}] simple trajs(mask+depth only) -> {simple_save_path}")

            ####  add hybrid trajs (using object center + foundation pose object headings) ####
            logging.info(f"[object] oid={oid}, name={oname} - hybrid trajs (fd-rot + mask/depth-trans)")
            R_fd = rel_w[:, :3, :3].astype(np.float64)        # [N,3,3]
            t_hyb = np.stack([pw_stack[i] - (R_fd[i] @ p0_w) for i in range(N)], axis=0).astype(np.float32)  # [N,3]

            hybrid_rel = rel_w.copy().astype(np.float32)
            hybrid_rel[:, :3, 3] = t_hyb
            hybrid_rel[0] = np.eye(4, dtype=np.float32)        # exact identity at t=0

            hybrid_save_path = base_dir / f"outputs/{key}/reconstruction/motions" / f"{oid}_{oname}_hybrid_trajs.npy"
            np.save(hybrid_save_path, hybrid_rel.astype(np.float32))
            scene_dict["info"]["objects"][obj_id]["hybrid_trajs"] = str(hybrid_save_path)
            logging.info(f"[object {oid}] hybrid trajs(fd-rot + mask/depth-trans) -> {hybrid_save_path}")


            if tmpdir is not None and Path(tmpdir).exists():
                shutil.rmtree(tmpdir, ignore_errors=True)

        # export the entire scene with fdpose object meshes
        bg_path = Path(scene_dict["info"]["background"]["registered"])
        bg_mesh = trimesh.load(str(bg_path), force='mesh')
        sc = trimesh.Scene()
        sc.add_geometry(bg_mesh, node_name="background")
        for name, m in placed_meshes:
            sc.add_geometry(m, node_name=name)

        scene_fdpose_path = base_dir / f"outputs/{key}/reconstruction/scenario/scene_fdpose.glb"
        scene_fdpose_path.parent.mkdir(parents=True, exist_ok=True)
        sc.export(str(scene_fdpose_path), file_type="glb")
        logging.info(f"[scene] scene_fdpose -> {scene_fdpose_path}")
        scene_dict["info"]["scene_mesh"]["fdpose"] = str(scene_fdpose_path)

        if debug_level >= 1:
            shutil.rmtree(Path(debug_dir), ignore_errors=True)
            logging.info(f"Clear up the debug directory: {debug_dir}")
        
        # udpate scene_dict
        key_scene_dicts[key] = scene_dict
        with open(base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
            pickle.dump(scene_dict, f)

    return key_scene_dicts

if __name__ == "__main__":
    set_logging()
    set_seed(0)

    base_dir = Path.cwd()
    cfg_path = base_dir / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.open("r"))
    keys = cfg["keys"]
    from utils.compose_config import compose_configs
    key_cfgs = {key: compose_configs(key, cfg) for key in keys}
    key_scene_dicts = {}
    for key in keys:
        scene_pkl = base_dir / f'outputs/{key}/scene/scene.pkl'
        with open(scene_pkl, 'rb') as f:
            scene_dict = pickle.load(f)
        key_scene_dicts[key] = scene_dict
    scenario_fdpose_optimization(keys, key_scene_dicts, key_cfgs)