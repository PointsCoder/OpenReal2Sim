import os
import sys
import numpy as np
import time
import torch
from typing import List
import open3d as o3d
from graspnetAPI.grasp import GraspGroup
from pathlib import Path
base_dir = Path.cwd()
from graspness_unofficial.models.graspnet import GraspNet, pred_decode
from graspness_unofficial.dataset.graspnet_dataset import minkowski_collate_fn
from graspness_unofficial.utils.collision_detector import ModelFreeCollisionDetector
from graspness_unofficial.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
from scipy.spatial.transform import Rotation as R
import trimesh
grasp_cfgs = {
    "save_files": False,
    "checkpoint_path": base_dir / "third_party" / "graspness_unofficial" / "ckpt" / "minkuresunet_kinect.tar", #hardcode path
    "seed_feat_dim": 512,
    "camera": "kinect",
    "num_point": 80000,
    "batch_size": 1,
    "voxel_size": 0.001, # 0.005
    "collision_thresh": 0.00001,
    "voxel_size_cd": 0.01, # 0.01
    "infer": True,
}

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

class MyGraspNet():
    def __init__(self):
        cfgs = grasp_cfgs
        self.cfgs = cfgs
        self.net = GraspNet(seed_feat_dim=cfgs["seed_feat_dim"], is_training=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(cfgs["checkpoint_path"])
        self.net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (cfgs["checkpoint_path"], start_epoch))

        self.net.eval()

    def inference(self, pcs):

        data_dict = {'point_clouds': pcs.astype(np.float32),
                    'coors': pcs.astype(np.float32) / self.cfgs["voxel_size"],
                    'feats': np.ones_like(pcs).astype(np.float32)}
        batch_data = minkowski_collate_fn([data_dict])
        tic = time.time()
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self.device)
            else:
                batch_data[key] = batch_data[key].to(self.device)

        # Forward pass
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds = pred_decode(end_points)

        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)

        # collision detection
        if self.cfgs["collision_thresh"] > 0:
            cloud = data_dict['point_clouds']
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs["voxel_size_cd"])
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs["collision_thresh"])
            gg = gg[~collision_mask]

        toc = time.time()
        # print('inference time: %fs' % (toc - tic))
        return gg

def inference(cfgs, data_input):
    batch_data = minkowski_collate_fn([data_input])
    net = GraspNet(seed_feat_dim=cfgs["seed_feat_dim"], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Load checkpoint
    checkpoint = torch.load(cfgs["checkpoint_path"])
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs["checkpoint_path"], start_epoch))

    net.eval()
    tic = time.time()

    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)

    # Forward pass
    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)

    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)
    # collision detection
    if cfgs["collision_thresh"] > 0:
        cloud = data_input['point_clouds']
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs["voxel_size_cd"])
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs["collision_thresh"])
        gg = gg[~collision_mask]

    toc = time.time()
    # print('inference time: %fs' % (toc - tic))
    return gg

def pointcloud_to_grasp(cfgs, pcs):
    data_dict = {'point_clouds': pcs.astype(np.float32),
                'coors': pcs.astype(np.float32) / cfgs["voxel_size"],
                'feats': np.ones_like(pcs).astype(np.float32)}
    gg = inference(cfgs, data_dict)
    return gg

def vis_grasp(pcs, gg):
    gg = gg.nms()
    gg = gg.sort_by_score()
    keep = 1
    if gg.__len__() > keep:
        gg = gg[:keep]
    grippers = gg.to_open3d_geometry_list()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcs.astype(np.float32))
    o3d.visualization.draw_geometries([cloud, *grippers])

def grasp_to_pointcloud(grippers, gripper_points=1000, gripper_color=[1, 0, 0]):
    grippers_pcd = o3d.geometry.PointCloud()
    for gripper in grippers:
        g_pcd = gripper.sample_points_uniformly(gripper_points)
        grippers_pcd += g_pcd
    grippers_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([gripper_color]), (len(grippers_pcd.points), 1)))
    return grippers_pcd


def vis_save_grasp(points, gg, best_grasp, save_path, colors=None, grasp_position=None, place_position=None):
    # visualize grasp pos, place pos, grasp poses, and pcd
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))

    if colors is None:
        cloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0, 1, 0]]), (len(cloud.points), 1)))
    elif isinstance(colors, np.ndarray):
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    elif isinstance(colors, list):
        cloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([colors]), (len(cloud.points), 1)))

    if type(gg) == GraspGroup:
        pcd_w_grasp = grasp_to_pointcloud(gg.to_open3d_geometry_list())
    else:
        pcd_w_grasp = grasp_to_pointcloud([gg.to_open3d_geometry()])

    pcd_best_grasp = grasp_to_pointcloud([best_grasp.to_open3d_geometry()], gripper_color=[0, 0, 1])
    pcd_w_grasp += pcd_best_grasp

    if grasp_position is not None and place_position is not None:
        pick_pcd = o3d.geometry.PointCloud()
        place_pcd = o3d.geometry.PointCloud()
        pick_pcd.points = o3d.utility.Vector3dVector(np.array(grasp_position).reshape(1,3).astype(np.float32))
        place_pcd.points = o3d.utility.Vector3dVector(np.array(place_position).reshape(1,3).astype(np.float32))
        pick_pcd.colors = place_pcd.colors = o3d.utility.Vector3dVector(np.array([[0,0,1]]).astype(np.float32))
        pcd_w_grasp = pcd_w_grasp + pick_pcd + place_pcd

    pcd_w_grasp += cloud

    o3d.io.write_point_cloud(save_path, pcd_w_grasp)

def get_pose_from_grasp(best_grasp):
    grasp_position = best_grasp.translation
    # convert rotation to isaacgym convention
    delta_m = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    rotation = np.dot(best_grasp.rotation_matrix, delta_m)
    quaternion_grasp = R.from_matrix(rotation).as_quat()
    quaternion_grasp = np.array([quaternion_grasp[3],quaternion_grasp[0],quaternion_grasp[1],quaternion_grasp[2]])
    rotation_unit_vect = rotation[:,2]
    # grasp_position -= 0.03 * rotation_unit_vect
    return grasp_position, quaternion_grasp, rotation_unit_vect

def get_best_grasp(gg, position, max_dis=0.05):
    # get best grasp pose around the grasp position
    best_grasp = None
    for g in gg:
        if np.linalg.norm(g.translation - position) < max_dis:
            if best_grasp is None:
                best_grasp = g
            else:
                if g.score > best_grasp.score:
                    best_grasp = g
    if best_grasp is None:
        best_grasp = gg[0]
    return best_grasp

def get_best_grasp_z_aligned(gg):
    # get best grasp pose that is facing the -z axis (for top-down grasping)
    best_grasp = None
    best_angle = np.inf

    gravity_direction = np.array([0, 0, -1])  # -Z direction in world coordinates

    for g in gg:

        # approach vector is +X axis of the grasp frame, make it close to the -Z axis in the world frame
        approach_vector = g.rotation_matrix[:, 0]
        approach_vector /= np.linalg.norm(approach_vector)

        # compute the angle between the approach vector and the gravity direction
        angle = np.arccos(np.clip(np.dot(approach_vector, gravity_direction), -1.0, 1.0))

        if angle < best_angle:
            best_angle = angle
            best_grasp = g
        elif np.isclose(angle, best_angle) and g.score > best_grasp.score:
            best_grasp = g

    if best_grasp is None:
        print("No best grasp found, falling back to the first grasp.")
        best_grasp = gg[0]  # fallback

    return best_grasp

def get_best_grasp_z_aligned_and_y_aligned(gg,
                                           desired_approach_dir=np.array([0, 0, -1]),
                                           desired_width_dir=np.array([0, 1, 0]),
                                           angle_tolerance=0.2):
    """
    Pick the grasp whose:
      1) X-axis (approach) is closest to desired_approach_dir (default -Z),
      2) Y-axis (gripper width) is closest to desired_width_dir (default +Y),
      3) Score is highest if angles tie.

    Grasp coordinate system:
      - rotation_matrix[:, 0] → X-axis = Approach direction
      - rotation_matrix[:, 1] → Y-axis = Gripper width direction
      - rotation_matrix[:, 2] → Z-axis = Depth/thickness direction
    """

    best_grasp = None
    # Track best angles and score
    best_approach_angle = np.inf
    best_width_angle = np.inf
    best_score = -np.inf

    # Normalize desired directions
    desired_approach_dir = desired_approach_dir / np.linalg.norm(desired_approach_dir)
    desired_width_dir = desired_width_dir / np.linalg.norm(desired_width_dir)

    for g in gg:
        # 1) Approach vector angle
        approach_vec = g.rotation_matrix[:, 0]
        approach_vec /= np.linalg.norm(approach_vec)
        approach_angle = angle_between(approach_vec, desired_approach_dir)

        # 2) Width vector angle
        width_vec = g.rotation_matrix[:, 1]
        width_vec /= np.linalg.norm(width_vec)
        width_angle = angle_between(width_vec, desired_width_dir)

        # 3) Compare to the "best" so far in a hierarchical manner
        if approach_angle < best_approach_angle:
            # Definitely better in terms of approach alignment => choose this
            best_approach_angle = approach_angle
            best_width_angle = width_angle
            best_score = g.score
            best_grasp = g
        elif np.isclose(approach_angle, best_approach_angle, atol=angle_tolerance):
            # Approach angles are essentially tied, compare width alignment
            if width_angle < best_width_angle:
                best_width_angle = width_angle
                best_score = g.score
                best_grasp = g
            elif np.isclose(width_angle, best_width_angle, atol=angle_tolerance):
                # Both angles tied, pick the higher score
                if g.score > best_score:
                    best_score = g.score
                    best_grasp = g

    if best_grasp is None and len(gg) > 0:
        print("No valid grasp found using angle criteria. Falling back to the first grasp.")
        best_grasp = gg[0]

    return best_grasp


def get_best_grasp_with_hints(gg: GraspGroup, point: List[float] = None, direction: List[float] = None):
    """
    Rescore all grasps using optional spatial and directional hints, then return a new
    GraspGroup sorted by this custom score (best first). Does NOT mutate the original gg.

    Scoring terms in [0, 1], combined by a weighted sum:
      - dir_term: alignment between grasp approach (+X axis) and `direction`
      - pt_term : proximity to `point` (RBF over distance)
      - net_term: original network score normalized over gg

    Args:
        gg: GraspGroup from graspnetAPI.
        point: (3,) world point. If provided, grasps closer to this point are preferred.
        direction: (3,) world direction. If provided, grasps whose approach (+X) aligns
                   with this direction are preferred.

    Returns:
        GraspGroup: a *new* group sorted by the custom score (descending).
                    The best guess is result[0].
    """
    # --- Early exits ---
    if gg is None or len(gg) == 0:
        return gg

    # Internal weights (you can tweak if needed)
    w_dir = 1.0
    w_pt  = 1.0
    w_net = 0

    # If hints are missing, zero-out the corresponding weights
    if point is None:
        w_pt = 0.0
    if direction is None or (np.asarray(direction).shape != (3,)):
        w_dir = 0.0

    # Length-scale for the point proximity (meters). Similar to your 0.05 window.
    sigma = 0.05
    sigma2 = max(sigma * sigma, 1e-12)

    # --- Gather per-grasp attributes ---
    translations = []
    approach_axes = []  # grasp frame +X
    net_scores    = []
    for g in gg:
        translations.append(g.translation.astype(np.float64))
        # Normalize +X axis as approach direction
        ax = g.rotation_matrix[:, 0].astype(np.float64)
        n  = np.linalg.norm(ax)
        approach_axes.append(ax / n if n > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float64))
        net_scores.append(float(g.score))

    translations = np.vstack(translations)          # (N,3)
    approach_axes = np.vstack(approach_axes)        # (N,3)
    net_scores = np.asarray(net_scores, dtype=np.float64)  # (N,)

    # --- Normalize original network scores to [0,1] ---
    if np.isfinite(net_scores).all() and (net_scores.max() > net_scores.min()):
        net_term = (net_scores - net_scores.min()) / (net_scores.max() - net_scores.min())
    else:
        net_term = np.zeros_like(net_scores)

    # --- Direction alignment term (cosine mapped to [0,1]) ---
    if w_dir > 0.0:
        d = np.asarray(direction, dtype=np.float64)
        nd = np.linalg.norm(d)
        if nd > 0:
            d = d / nd
            cosv = np.clip((approach_axes * d[None, :]).sum(axis=1), -1.0, 1.0)
            dir_term = 0.5 * (cosv + 1.0)  # map [-1,1] -> [0,1]
        else:
            dir_term = np.zeros(len(gg), dtype=np.float64)
    else:
        dir_term = np.zeros(len(gg), dtype=np.float64)

    # --- Point proximity term (RBF over Euclidean distance) ---
    if w_pt > 0.0:
        p = np.asarray(point, dtype=np.float64).reshape(1, 3)
        dists = np.linalg.norm(translations - p, axis=1)
        pt_term = np.exp(-0.5 * (dists * dists) / sigma2)  # in (0,1]
    else:
        pt_term = np.zeros(len(gg), dtype=np.float64)

    # --- Combine ---
    total_score = w_dir * dir_term + w_pt * pt_term + w_net * net_term
    gg.scores = total_score
    gg.sort_by_score()  # Sort in-place by the new score (best first)
    return gg


def angle_between(v1, v2):
    """Utility to compute the angle between two normalized vectors in [0, π]."""
    dot_val = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot_val)
