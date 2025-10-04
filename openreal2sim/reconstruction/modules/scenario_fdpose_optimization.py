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
                "fdpose":      # object placement at using foundation pose estimation [glb],
            },
            ...
        },
        "scene_mesh": {
            ...
            "scene_fdpose": # entire scene with fdpose placed object meshes [glb],
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

            droot = debug_dir / f"fdpose_obj_{oid}_{oname}"
            if debug_level >= 1:
                (droot / "track_vis").mkdir(parents=True, exist_ok=True)
                (droot / "ob_in_cam").mkdir(parents=True, exist_ok=True)

            imgs = scene_dict["images"]
            depths = scene_dict["depths"]
            Ks = scene_dict["intrinsics"].astype(np.float32)
            N = imgs.shape[0]
            Ks = np.repeat(Ks[None, ...], N, axis=0)
            imageio.imwrite(droot / "track_vis" / f"{0:06d}_mask.png", mask_array[0]*255)
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
        scene_dict["info"]["scene_mesh"]["scene_fdpose"] = str(scene_fdpose_path)

        if debug_level >= 3:
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