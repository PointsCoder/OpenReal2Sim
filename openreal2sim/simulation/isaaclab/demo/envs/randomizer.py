import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from .task_cfg import TaskCfg, TaskType, SuccessMetric, SuccessMetricType, TrajectoryCfg

import torch
import transforms3d

def pose_to_mat(pos, quat):
    if torch.is_tensor(pos):  pos  = pos.cpu().numpy()
    if torch.is_tensor(quat): quat = quat.cpu().numpy()
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = transforms3d.quaternions.quat2mat(quat)
    m[:3,  3] = pos
    return m

def mat_to_pose(mat):
    pos  = mat[:3, 3]
    quat = transforms3d.quaternions.mat2quat(mat[:3, :3])
    return pos, quat



class Randomizer(TaskCfg):
    def __init__(self, task_cfg: TaskCfg):
        self.task_cfg = task_cfg

    
    def generate_randomized_scene_cfg(self, grid_dist: float, grid_num: int, angle_random_range: float, angle_random_num: int, traj_randomize_num:int, scene_randomize_num: int, robot_pose_randomize_range, robot_pose_randomize_angle, robot_pose_randomize_num, fix_end_pose: bool = False, fix_other_objects: bool = False, fix_robot_pose: bool = False, use_no_interpolation: bool = False):
        # Step 1: Generate candidate transforms
      
        candidate_transforms = []
        for i in range(-grid_num, grid_num):
            for j in range(-grid_num, grid_num):
                random_angles = np.random.uniform(-angle_random_range, angle_random_range, angle_random_num)
                for angle in random_angles:
                    orig = np.eye(4)
                    orig[0, 3] = i * grid_dist
                    orig[1, 3] = j * grid_dist
                    orig[0, 0] = np.cos(angle)
                    orig[0, 1] = -np.sin(angle)
                    orig[1, 0] = np.sin(angle)
                    orig[1, 1] = np.cos(angle)
                    candidate_transforms.append(orig)
        
        # Step 2: Generate traj_randomize_num combinations of (start_pose, end_pose)
        traj_pose_pairs = []
        for _ in range(traj_randomize_num):
            start_pose = random.choice(candidate_transforms)
            if fix_end_pose:
                #import pdb; pdb.set_trace()
                end_pose = np.eye(4)
            else:
                if self.task_cfg.task_type == TaskType.SIMPLE_PICK:
                    end_pose = start_pose.copy()  # For SIMPLE_PICK, end same as start
                else:
                    end_pose = random.choice(candidate_transforms)
            traj_pose_pairs.append((start_pose, end_pose))
        
        # Step 3: Generate scene_randomize_num combinations for other objects
        other_object_ids = [obj.object_id for obj in self.task_cfg.objects 
                           if obj.object_id != self.task_cfg.manipulated_oid and obj.object_id not in self.task_cfg.start_related and obj.object_id not in self.task_cfg.end_related]
        
        if fix_other_objects:
            scene_poses_combinations = [[(oid, [0,0,0,1,0,0,0]) for oid in other_object_ids]]
        else: 
            scene_poses_combinations = []
            for _ in range(scene_randomize_num):
                # Create list of (oid, pose) pairs
                other_object_poses = [(oid, random.choice(candidate_transforms)) 
                                    for oid in other_object_ids]
                scene_poses_combinations.append(other_object_poses)
        
        # Step 4: Generate robot_pose_randomize_num random robot poses
        # robot_pose format: [x, y, z, w, x, y, z] (position + quaternion wxyz)
        ## FIXME: hacking
        ref_traj = self.task_cfg.reference_trajectory[-1]
        assert ref_traj is not None, "Reference trajectory is not found"
        ref_robot_pose = np.array(ref_traj.robot_pose)
        robot_pose_mat = pose_to_mat(ref_robot_pose[:3], ref_robot_pose[3:7])
        
        if fix_robot_pose:
            robot_poses = [ref_robot_pose]
        else:
            robot_poses = []
            for _ in range(robot_pose_randomize_num):
                # Random translation within range
                random_trans = np.random.uniform(
                    -robot_pose_randomize_range, 
                    robot_pose_randomize_range, 
                    3
                )
                random_rot = np.random.uniform(
                    -robot_pose_randomize_angle, 
                    robot_pose_randomize_angle, 
                )

                rotate_matrix = np.eye(4)
                rotate_matrix[:3, :3] = R.from_euler('z', random_rot).as_matrix()
                new_robot_pose = rotate_matrix @ robot_pose_mat
                new_robot_pose[:3, 3] += random_trans
                # Combine position and quaternion [x, y, z, w, x, y, z]
                pos, quat = mat_to_pose(new_robot_pose)
                robot_pose_7d = np.concatenate([pos, quat])
                robot_poses.append(robot_pose_7d.tolist())
            
        # Step 5: Generate trajectories for all combinations
        generated_trajectories = []
        ref_traj = self.task_cfg.reference_trajectory[-1]
        for start_pose, end_pose in traj_pose_pairs:
            for other_object_poses in scene_poses_combinations:
                for robot_pose in robot_poses:
                    # Generate trajectory for manipulated object
                    if self.task_cfg.task_type == TaskType.SIMPLE_PICK:
                        # The reference trajectory poses are not guaranteed to be expressed
                        # relative to a canonical origin; anchor them to the reference start pose.
                        # This ensures the same transform that maps ref start -> new start also maps
                        # every waypoint (including the final goal).
                        ref_mats = []
                        for traj_pose_7d in ref_traj.object_trajectory:
                            ref_mats.append(
                                pose_to_mat(
                                    np.array(traj_pose_7d[:3]),
                                    np.array(traj_pose_7d[3:7]),
                                )
                            )
                        ref_mats = np.asarray(ref_mats, dtype=np.float32)
                        T_ref0_inv = np.linalg.inv(ref_mats[0]).astype(np.float32)

                        new_traj_mats = []
                        for pose_mat in ref_mats:
                            rel_mat = (pose_mat @ T_ref0_inv).astype(np.float32)
                            new_traj_mats.append((start_pose @ rel_mat).astype(np.float32))
                        # Convert back to 7D format
                        new_traj_7d = []
                        for mat in new_traj_mats:
                            pos, quat = mat_to_pose(mat)
                            new_traj_7d.append(np.concatenate([pos, quat]).tolist())
                        
                        # Transform pregrasp and grasp poses
                        if ref_traj.pregrasp_pose:
                            pre_ref = pose_to_mat(
                                np.array(ref_traj.pregrasp_pose[:3]),
                                np.array(ref_traj.pregrasp_pose[3:7]),
                            ).astype(np.float32)
                            pre_rel = (pre_ref @ T_ref0_inv).astype(np.float32)
                            pregrasp_mat = (start_pose @ pre_rel).astype(np.float32)
                            pos, quat = mat_to_pose(pregrasp_mat)
                            pregrasp_pose = np.concatenate([pos, quat]).tolist()
                        else:
                            pregrasp_pose = None
                            
                        if ref_traj.grasp_pose:
                            grasp_ref = pose_to_mat(
                                np.array(ref_traj.grasp_pose[:3]),
                                np.array(ref_traj.grasp_pose[3:7]),
                            ).astype(np.float32)
                            grasp_rel = (grasp_ref @ T_ref0_inv).astype(np.float32)
                            grasp_mat = (start_pose @ grasp_rel).astype(np.float32)
                            pos, quat = mat_to_pose(grasp_mat)
                            grasp_pose = np.concatenate([pos, quat]).tolist()
                        else:
                            grasp_pose = None
                    else:
                        # Convert reference trajectory to mat format
                        ref_traj_mats = []
                        for traj_pose_7d in ref_traj.object_trajectory:
                            ref_traj_mats.append(pose_to_mat(np.array(traj_pose_7d[:3]), np.array(traj_pose_7d[3:7])))
                        ref_traj_mats = np.array(ref_traj_mats, dtype=np.float32)

                        # Anchor reference trajectory to its first pose.
                        T_ref0_inv = np.linalg.inv(ref_traj_mats[0]).astype(np.float32)
                        ref_traj_mats_rel = np.matmul(ref_traj_mats, T_ref0_inv)
                        
                        if use_no_interpolation:
                            new_traj_mats = ref_traj_mats_rel.copy()
                            new_traj_mats[0] = (start_pose @ ref_traj_mats_rel[0]).astype(np.float32)
                            new_traj_mats[-1] = (end_pose @ ref_traj_mats_rel[-1]).astype(np.float32)
                        else:
                            new_traj_mats = self.compute_new_traj(start_pose, end_pose, ref_traj_mats_rel)
                            new_traj_mats = self.lift_traj(ref_traj_mats_rel, new_traj_mats)
                        # Convert back to 7D format
                        new_traj_7d = []
                        for mat in new_traj_mats:
                            pos, quat = mat_to_pose(mat)
                            new_traj_7d.append(np.concatenate([pos, quat]).tolist())
                        #new_traj_7d = lift_traj(new_traj_7d)
                        # Transform pregrasp and grasp poses
                        if ref_traj.pregrasp_pose:
                            pre_ref = pose_to_mat(
                                np.array(ref_traj.pregrasp_pose[:3]),
                                np.array(ref_traj.pregrasp_pose[3:7]),
                            ).astype(np.float32)
                            pre_rel = (pre_ref @ T_ref0_inv).astype(np.float32)
                            pregrasp_mat = (start_pose @ pre_rel).astype(np.float32)
                            pos, quat = mat_to_pose(pregrasp_mat)
                            pregrasp_pose = np.concatenate([pos, quat]).tolist()
                        else:
                            pregrasp_pose = None
                            
                        if ref_traj.grasp_pose:
                            grasp_ref = pose_to_mat(
                                np.array(ref_traj.grasp_pose[:3]),
                                np.array(ref_traj.grasp_pose[3:7]),
                            ).astype(np.float32)
                            grasp_rel = (grasp_ref @ T_ref0_inv).astype(np.float32)
                            grasp_mat = (start_pose @ grasp_rel).astype(np.float32)
                            pos, quat = mat_to_pose(grasp_mat)
                            grasp_pose = np.concatenate([pos, quat]).tolist()
                        else:
                            grasp_pose = None
                    
                    # Apply scene randomization to other objects (if needed, update object_poses here)
                    # other_object_poses is now a list of (oid, pose) pairs
                    
                    # Create success metric    

                    ref_end_pose_mat = pose_to_mat(
                        np.array(ref_traj.success_metric.end_pose[:3]), 
                        np.array(ref_traj.success_metric.end_pose[3:7])
                    )
                    end_pose_metric_mat = end_pose @ ref_end_pose_mat
                    pos, quat = mat_to_pose(end_pose_metric_mat)
                    end_pose_metric_7d = np.concatenate([pos, quat]).tolist()

                    success_metric_type = ref_traj.success_metric.success_metric_type
                    if success_metric_type == SuccessMetricType.SIMPLE_LIFT:
                        success_metric = SuccessMetric(
                            success_metric_type=SuccessMetricType.SIMPLE_LIFT,
                            lift_height=ref_traj.success_metric.lift_height,
                            final_gripper_close=ref_traj.success_metric.final_gripper_close,
                            end_pose=end_pose_metric_7d
                        )
                    elif success_metric_type == SuccessMetricType.TARGET_POINT:
                        # Convert end_pose from 7D to mat, transform, then back to 7D
                        success_metric = SuccessMetric(
                            success_metric_type=SuccessMetricType.TARGET_POINT,
                            final_gripper_close=ref_traj.success_metric.final_gripper_close,
                            end_pose=end_pose_metric_7d
                        )
                    elif success_metric_type == SuccessMetricType.TARGET_PLANE:
                        success_metric = SuccessMetric(
                            success_metric_type=SuccessMetricType.TARGET_PLANE,
                            ground_value=ref_traj.success_metric.ground_value,
                            final_gripper_close=ref_traj.success_metric.final_gripper_close,
                            end_pose=end_pose_metric_7d
                        )
                    
                    # Store object poses in 7D format
                    object_poses = {}
                    for oid, pose_mat in other_object_poses:
                        pos, quat = mat_to_pose(pose_mat)
                        object_poses[oid] = np.concatenate([pos, quat]).tolist()
                    
                    # Add start and end related objects
                    start_pos, start_quat = mat_to_pose(start_pose)
                    start_pose_7d = np.concatenate([start_pos, start_quat]).tolist()
                    for oid in self.task_cfg.start_related:
                        if not fix_other_objects:
                            object_poses[oid] = start_pose_7d
                        else:
                            object_poses[oid] = [0,0,0,1,0,0,0]
                    
                    end_pos, end_quat = mat_to_pose(end_pose)
                    end_pose_7d = np.concatenate([end_pos, end_quat]).tolist()
                    for oid in self.task_cfg.end_related:
                        if not fix_other_objects:
                            object_poses[oid] = end_pose_7d
                        else:
                            object_poses[oid] = [0,0,0,1,0,0,0]
                    
                    object_poses[self.task_cfg.manipulated_oid] = start_pose_7d
                    
                    robot_type = ref_traj.robot_type
                    generated_trajectories.append(TrajectoryCfg(
                        robot_pose=robot_pose,
                        object_poses=object_poses,
                        object_trajectory=new_traj_7d,
                        final_gripper_close=ref_traj.final_gripper_close,
                        pregrasp_pose=pregrasp_pose,
                        grasp_pose=grasp_pose,
                        success_metric=success_metric,
                        robot_type=robot_type
                    ))
        
        # # Total trajectories = traj_randomize_num * scene_randomize_num * robot_pose_randomize_num
        # assert len(generated_trajectories) == traj_randomize_num * scene_randomize_num * robot_pose_randomize_num, \
        #     f"Expected {traj_randomize_num * scene_randomize_num * robot_pose_randomize_num} trajectories, got {len(generated_trajectories)}"
      
        random.shuffle(generated_trajectories)
        return generated_trajectories

    @staticmethod
    def compute_new_traj(start_trans: np.ndarray, end_trans: np.ndarray, reference_traj: np.ndarray, 
                        interp_mode: str = 'slerp') -> np.ndarray:
        """
        Compute interpolated trajectory using Real2Render2Real's relative shape preservation method.
        
        This maintains the relative motion pattern of the reference trajectory by normalizing
        the progress along the reference and mapping it to the new start and end positions.
        
        Args:
            start_trans: 4x4 transformation matrix from original start to new start
            end_trans: 4x4 transformation matrix from original end to new end  
            reference_traj: Reference trajectory in SE(3), shape (N, 4, 4) or (N, 16)
            interp_mode: 'linear' or 'slerp' for rotation interpolation (default: 'slerp')
            
        Returns:
            New interpolated trajectory with same shape as reference_traj
        """
        # Ensure reference_traj is in (N, 4, 4) format
        if reference_traj.ndim == 2 and reference_traj.shape[1] == 16:
            reference_traj = reference_traj.reshape(-1, 4, 4)
        elif reference_traj.ndim == 3 and reference_traj.shape[1] == 4 and reference_traj.shape[2] == 4:
            reference_traj = reference_traj.copy()
        else:
            raise ValueError(f"Invalid reference_traj shape: {reference_traj.shape}, expected (N, 4, 4) or (N, 16)")
        
        N = len(reference_traj)
        if N == 0:
            return reference_traj.copy()
        
        # Get start and end poses from reference trajectory
        ref_start = reference_traj[0].copy()
        ref_end = reference_traj[-1].copy()
        
        # Compute new start and end poses
        new_start_mat = start_trans @ ref_start
        new_end_mat = end_trans @ ref_end
        
        # Convert to 7D format using mat_to_pose
        ref_traj_7d = []
        for pose_mat in reference_traj:
            pos, quat = mat_to_pose(pose_mat)
            ref_traj_7d.append(np.concatenate([pos, quat]))
        ref_traj_7d = np.array(ref_traj_7d)
        
        pos_start, quat_start = mat_to_pose(new_start_mat)
        pos_end, quat_end = mat_to_pose(new_end_mat)
        
        # Initialize new trajectory
        new_traj_7d = np.zeros_like(ref_traj_7d)
        
        # Normalize time steps
        t = np.linspace(0, 1, N)
        
        # Split into position and rotation components
        pos_orig = ref_traj_7d[:, :3]
        quat_orig = ref_traj_7d[:, 3:7]  # wxyz format (from transforms3d)
        
        ref_start_pos, ref_start_quat = mat_to_pose(ref_start)
        ref_end_pos, ref_end_quat = mat_to_pose(ref_end)
        
        # Interpolate positions: preserve relative shape from reference trajectory
        for axis in range(3):
            ref_range = ref_end_pos[axis] - ref_start_pos[axis]
            if np.abs(ref_range) > 1e-10:
                # Normalize progress along reference trajectory
                normalized_progress = (pos_orig[:, axis] - ref_start_pos[axis]) / ref_range
                # Map to new range
                new_traj_7d[:, axis] = pos_start[axis] + (pos_end[axis] - pos_start[axis]) * normalized_progress
            else:
                # If no change in reference, use direct transformation
                new_traj_7d[:, axis] = pos_orig[:, axis] + (pos_start[axis] - ref_start_pos[axis])
        
        # Interpolate rotations using SLERP
        if interp_mode == 'slerp':
            # Use scipy Slerp for spherical linear interpolation
            # Convert wxyz to xyzw for scipy
            quat_start_xyzw = np.array([quat_start[1], quat_start[2], quat_start[3], quat_start[0]])
            quat_end_xyzw = np.array([quat_end[1], quat_end[2], quat_end[3], quat_end[0]])
            
            # Create Slerp interpolator
            key_rots = R.from_quat([quat_start_xyzw, quat_end_xyzw])
            key_times = [0, 1]
            slerp = Slerp(key_times, key_rots)
            
            # Interpolate for all time steps
            interp_rots = slerp(t)
            quat_interp_xyzw = interp_rots.as_quat()
            
            # Convert back to wxyz format
            new_traj_7d[:, 3] = quat_interp_xyzw[:, 3]  # w
            new_traj_7d[:, 4] = quat_interp_xyzw[:, 0]  # x
            new_traj_7d[:, 5] = quat_interp_xyzw[:, 1]  # y
            new_traj_7d[:, 6] = quat_interp_xyzw[:, 2]  # z
        else:  # linear
            # Linear interpolation (needs normalization)
            for i in range(N):
                new_traj_7d[i, 3:7] = (1 - t[i]) * quat_start + t[i] * quat_end
                new_traj_7d[i, 3:7] /= np.linalg.norm(new_traj_7d[i, 3:7])
        
        # Convert back to SE(3) matrices using pose_to_mat
        new_traj = []
        for pose_7d in new_traj_7d:
            pose_mat = pose_to_mat(pose_7d[:3], pose_7d[3:7])
            new_traj.append(pose_mat)
        new_traj = np.array(new_traj)
        
        return new_traj


    def lift_traj(self, old_traj, new_traj):
        T = len(old_traj)
        renewed_traj = []
        for t in range(T):
            old_pose = old_traj[t]
            new_pose = new_traj[t]
            old_pose_z = old_pose[2,3]
            new_pose_z = new_pose[2,3]
            if old_pose_z > new_pose_z:
               new_pose[2,3] = old_pose_z
            renewed_traj.append(new_pose)
        return renewed_traj

    def select_randomized_cfg(self):
        return random.choice(self.task_cfg.generated_trajectories)




