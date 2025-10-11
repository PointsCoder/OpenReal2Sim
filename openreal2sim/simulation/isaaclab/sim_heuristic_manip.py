"""
Heuristic manipulation policy in Isaac Lab simulation.
Using grasping and motion planning to perform object manipulation tasks.
"""
from __future__ import annotations

# ─────────── AppLauncher ───────────
import argparse, os, json, random, transforms3d
from pathlib import Path
import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher

# ─────────── CLI ───────────
parser = argparse.ArgumentParser("sim_policy")
parser.add_argument("--key", type=str, default="lab16", help="scene key (outputs/<key>)")
parser.add_argument("--robot", type=str, default="franka")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_trials", type=int, default=1)
parser.add_argument("--teleop_device", type=str, default="keyboard")
parser.add_argument("--sensitivity", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = False  # headless mode for batch execution
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# ─────────── Runtime imports ───────────
import isaaclab.sim as sim_utils
from isaaclab.utils.math import subtract_frame_transforms

from graspnetAPI.grasp import GraspGroup


# ─────────── Simulation environments ───────────
from sim_base import BaseSimulator, get_next_demo_id
from sim_env_factory import make_env
from preprocess.grasping_utils import get_best_grasp_with_hints
from utils.transform_utils import pose_to_mat, mat_to_pose, grasp_to_world, grasp_approach_axis_batch
from utils.sim_utils import load_sim_parameters

BASE_DIR   = Path.cwd()
img_folder = args_cli.key
out_dir    = BASE_DIR / "outputs"


# ──────────────────────────── Heuristic Manipulation ────────────────────────────
class HeuristicManipulation(BaseSimulator):
    """
    Physical trial-and-error grasping with approach-axis perturbation:
      • Multiple grasp proposals executed in parallel;
      • Every attempt does reset → pre → grasp → close → lift → check;
      • Early stops when any env succeeds; then re-exec for logging.
    """
    def __init__(self, sim, scene, sim_cfgs: dict):
        robot_pose = torch.tensor(
            sim_cfgs["robot_cfg"]["robot_pose"],
            dtype=torch.float32,
            device=sim.device
        )
        super().__init__(
            sim=sim, scene=scene, args=args_cli,
            robot_pose=robot_pose, cam_dict=sim_cfgs["cam_cfg"],
            out_dir=out_dir, img_folder=img_folder,
            enable_motion_planning=True,
            set_physics_props=True, debug_level=0,
        )

        self.selected_object_id = sim_cfgs["demo_cfg"]["manip_object_id"]
        self.traj_key = sim_cfgs["demo_cfg"]["traj_key"]
        self.traj_path = sim_cfgs["demo_cfg"]["traj_path"]
        self.goal_offset = [0, 0, sim_cfgs["demo_cfg"]["goal_offset"]]
        self.grasp_path = sim_cfgs["demo_cfg"]["grasp_path"]
        self.grasp_idx = sim_cfgs["demo_cfg"]["grasp_idx"]
        self.grasp_pre = sim_cfgs["demo_cfg"]["grasp_pre"]
        self.grasp_delta = sim_cfgs["demo_cfg"]["grasp_delta"]
        self.load_obj_goal_traj()

    def load_obj_goal_traj(self):
        """
        Load the relative trajectory Δ_w (T,4,4) and precompute the absolute
        object goal trajectory for each env using the *actual current* object pose
        in the scene as T_obj_init (not env_origin).
          T_obj_goal[t] = Δ_w[t] @ T_obj_init

        Sets:
          self.obj_rel_traj   : np.ndarray or None, shape (T,4,4)
          self.obj_goal_traj_w: np.ndarray or None, shape (B,T,4,4)
        """
        # —— 1) Load Δ_w ——
        rel = np.load(self.traj_path).astype(np.float32)
        self.obj_rel_traj = rel  # (T,4,4)

        self.reset()

        # —— 2) Read current object initial pose per env as T_obj_init ——
        B = self.scene.num_envs
        # obj_state = self.object_prim.data.root_com_state_w[:, :7]  # [B,7], pos(3)+quat(wxyz)(4)
        obj_state = self.object_prim.data.root_state_w[:, :7]  # [B,7], pos(3)+quat(wxyz)(4)
        self.show_goal(obj_state[:, :3], obj_state[:, 3:7])

        obj_state_np = obj_state.detach().cpu().numpy().astype(np.float32)
        offset_np = np.asarray(self.goal_offset, dtype=np.float32).reshape(3)
        obj_state_np[:, :3] += offset_np  # raise a bit to avoid collision

        # Note: here the relative traj Δ_w is defined in world frame with origin (0,0,0),
        # Hence, we need to normalize it to each env's origin frame.
        origins = self.scene.env_origins.detach().cpu().numpy().astype(np.float32)  # (B,3)
        obj_state_np[:, :3] -= origins # normalize to env origin frame

        # —— 3) Precompute absolute object goals for all envs ——
        T = rel.shape[0]
        obj_goal = np.zeros((B, T, 4, 4), dtype=np.float32)
        for b in range(B):
            T_init = pose_to_mat(obj_state_np[b, :3], obj_state_np[b, 3:7])  # (4,4)
            for t in range(T):
                goal = rel[t] @ T_init
                goal[:3, 3] += origins[b]  # back to world frame
                obj_goal[b, t] = goal

        self.obj_goal_traj_w = obj_goal  # [B, T, 4, 4]

    def follow_object_goals(self, start_joint_pos, sample_step=1, visualize=True):
        """
        follow precompute object absolute trajectory: self.obj_goal_traj_w:
          T_obj_goal[t] = Δ_w[t] @ T_obj_init

        EE-object transform is fixed:
          T_ee_goal[t] = T_obj_goal[t] @ (T_obj_grasp^{-1} @ T_ee_grasp)
        Here T_obj_grasp / T_ee_grasp is the transform at the moment of grasping.
        """

        B = self.scene.num_envs
        obj_goal_all = self.obj_goal_traj_w  # [B, T, 4, 4]
        T = obj_goal_all.shape[1]

        ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B,7]
        # obj_w = self.object_prim.data.root_com_state_w[:, :7]                                 # [B,7]
        obj_w = self.object_prim.data.root_state_w[:, :7]                                 # [B,7]

        T_ee_in_obj = []
        for b in range(B):
            T_ee_w  = pose_to_mat(ee_w[b, :3],  ee_w[b, 3:7])
            T_obj_w = pose_to_mat(obj_w[b, :3], obj_w[b, 3:7])
            T_ee_in_obj.append((np.linalg.inv(T_obj_w) @ T_ee_w).astype(np.float32))

        joint_pos = start_joint_pos
        root_w = self.robot.data.root_state_w[:, 0:7]  # robot base poses per env

        t_iter = list(range(0, T, sample_step))
        t_iter = t_iter + [T-1] if t_iter[-1] != T-1 else t_iter

        for t in t_iter:
            goal_pos_list, goal_quat_list = [], []
            print(f"[INFO] follow object goal step {t}/{T}")
            for b in range(B):
                T_obj_goal = obj_goal_all[b, t]            # (4,4)
                T_ee_goal  = T_obj_goal @ T_ee_in_obj[b]   # (4,4)
                pos_b, quat_b = mat_to_pose(T_ee_goal)
                goal_pos_list.append(pos_b.astype(np.float32))
                goal_quat_list.append(quat_b.astype(np.float32))

            goal_pos  = torch.as_tensor(np.stack(goal_pos_list),  dtype=torch.float32, device=self.sim.device)
            goal_quat = torch.as_tensor(np.stack(goal_quat_list), dtype=torch.float32, device=self.sim.device)

            if visualize:
                self.show_goal(goal_pos, goal_quat)
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_w[:, :3], root_w[:, 3:7], goal_pos, goal_quat
            )
            joint_pos, success = self.move_to(ee_pos_b, ee_quat_b, gripper_open=False)
            self.save_dict["actions"].append(np.concatenate([ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), np.ones((B, 1))], axis=1))


        joint_pos = self.wait(gripper_open=True, steps=10)
        return joint_pos

    def viz_object_goals(self, sample_step=1, hold_steps=20):
        self.reset()
        self.wait(gripper_open=True, steps=10)
        B = self.scene.num_envs
        env_ids = torch.arange(B, device=self.object_prim.device, dtype=torch.long)
        goals = self.obj_goal_traj_w
        t_iter = list(range(0, goals.shape[1], sample_step))
        t_iter = t_iter + [goals.shape[1]-1] if t_iter[-1] != goals.shape[1]-1 else t_iter
        for t in t_iter:
            print(f"[INFO] viz object goal step {t}/{goals.shape[1]}")
            pos_list, quat_list = [], []
            for b in range(B):
                pos, quat = mat_to_pose(goals[b, t])
                pos_list.append(pos.astype(np.float32))
                quat_list.append(quat.astype(np.float32))
            pose = self.object_prim.data.root_state_w[:, :7]
            # pose = self.object_prim.data.root_com_state_w[:, :7]
            pose[:, :3]   = torch.tensor(np.stack(pos_list),  dtype=torch.float32, device=pose.device)
            pose[:, 3:7]  = torch.tensor(np.stack(quat_list), dtype=torch.float32, device=pose.device)
            self.show_goal(pose[:, :3], pose[:, 3:7])

            for _ in range(hold_steps):
                self.object_prim.write_root_pose_to_sim(pose, env_ids=env_ids)
                self.object_prim.write_data_to_sim()
                self.step()

    # ---------- Helpers ----------
    def _to_base(self, pos_w: np.ndarray | torch.Tensor, quat_w: np.ndarray | torch.Tensor):
        """World → robot base frame for all envs."""
        root = self.robot.data.root_state_w[:, 0:7]  # [B,7]
        p_w, q_w = self._ensure_batch_pose(pos_w, quat_w)
        pb, qb = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7], p_w, q_w
        )
        return pb, qb  # [B,3], [B,4]

    # ---------- Batched execution & lift-check ----------
    def execute_and_lift_once_batch(self, info: dict, lift_height=0.12) -> tuple[np.ndarray, np.ndarray]:
        """
        Reset → pre → grasp → close → lift → hold; return (success[B], score[B]).
        """
        B = self.scene.num_envs
        self.reset()

        # open gripper buffer
        jp = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        self.wait(gripper_open=True, steps=4)

        # pre-grasp
        jp, success = self.move_to(info["pre_p_b"], info["pre_q_b"], gripper_open=True)
        if torch.any(success==False): return np.zeros(B, bool), np.zeros(B, np.float32)
        jp = self.wait(gripper_open=True, steps=3)

        # grasp
        jp, success = self.move_to(info["p_b"], info["q_b"], gripper_open=True)
        if torch.any(success==False): return np.zeros(B, bool), np.zeros(B, np.float32)
        jp = self.wait(gripper_open=True, steps=2)

        # close
        jp = self.wait(gripper_open=False, steps=6)

        # initial heights
        obj0 = self.object_prim.data.root_com_pos_w[:, 0:3]     # [B,3]
        ee_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]  # [B,7]
        ee_p0 = ee_w[:, :3]
        robot_ee_z0 = ee_p0[:, 2].clone()
        obj_z0 = obj0[:, 2].clone()
        print(f"[INFO] mean object z0={obj_z0.mean().item():.3f} m, mean EE z0={robot_ee_z0.mean().item():.3f} m")

        # lift: keep orientation, add height
        ee_q = ee_w[:, 3:7]
        target_p = ee_p0.clone()
        target_p[:, 2] += lift_height

        root = self.robot.data.root_state_w[:, 0:7]
        p_lift_b, q_lift_b = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7],
            target_p, ee_q
        )
        jp, success = self.move_to(p_lift_b, q_lift_b, gripper_open=False)
        if torch.any(success==False): return np.zeros(B, bool), np.zeros(B, np.float32)
        jp = self.wait(gripper_open=False, steps=8)

        # final heights
        obj1 = self.object_prim.data.root_com_pos_w[:, 0:3]
        ee_w1 = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        robot_ee_z1 = ee_w1[:, 2]
        obj_z1 = obj1[:, 2]
        print(f"[INFO] mean object z1={obj_z1.mean().item():.3f} m, mean EE z1={robot_ee_z1.mean().item():.3f} m")

        # lifted if EE and object rise similarly (tight coupling)
        ee_diff  = robot_ee_z1 - robot_ee_z0
        obj_diff = obj_z1 - obj_z0
        lifted = (torch.abs(ee_diff - obj_diff) <= 0.01) & \
            (torch.abs(ee_diff) >= 0.5 * lift_height) & \
            (torch.abs(obj_diff) >= 0.5 * lift_height)  # [B] bool

        score = torch.zeros_like(ee_diff)
        score[lifted] = 1000.0
        return lifted.detach().cpu().numpy().astype(bool), score.detach().cpu().numpy().astype(np.float32)

    def lift_up(self, height=0.12, gripper_open=False):
        """
        Lift up by a certain height (m) from current EE pose.
        """
        ee_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        target_p = ee_w[:, :3].clone()
        target_p[:, 2] += height

        root = self.robot.data.root_state_w[:, 0:7]
        p_lift_b, q_lift_b = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7],
            target_p, ee_w[:, 3:7]
        )
        jp, success = self.move_to(p_lift_b, q_lift_b, gripper_open=gripper_open)
        jp = self.wait(gripper_open=gripper_open, steps=8)
        return jp

    def is_success(self) -> bool:
        ee_w  = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        obj_w = self.object_prim.data.root_com_pos_w[:, 0:3]
        dist = torch.norm(obj_w[:, :3] - ee_w[:, :3], dim=1).mean().item()
        print("Success!" if dist < 0.10 else "Fail!")
        return dist < 0.10

    def build_grasp_info(
        self,
        grasp_pos_w_batch: np.ndarray,   # (B,3)  GraspNet proposal in world frame
        grasp_quat_w_batch: np.ndarray,  # (B,4)  wxyz
        pre_dist_batch: np.ndarray,      # (B,)
        delta_batch: np.ndarray          # (B,)
    ) -> dict:
        """
        返回与 _build_info 相同结构，但每个 env 的抓取都可不同。
        """
        B = self.scene.num_envs
        p_w   = np.asarray(grasp_pos_w_batch,  dtype=np.float32).reshape(B, 3)
        q_w   = np.asarray(grasp_quat_w_batch, dtype=np.float32).reshape(B, 4)
        pre_d = np.asarray(pre_dist_batch,     dtype=np.float32).reshape(B)
        delt  = np.asarray(delta_batch,        dtype=np.float32).reshape(B)

        a_batch = grasp_approach_axis_batch(q_w)  # (B,3)

        pre_p_w = (p_w - pre_d[:, None] * a_batch).astype(np.float32)
        gra_p_w = (p_w + delt[:,  None] * a_batch).astype(np.float32)

        origins = self.scene.env_origins.detach().cpu().numpy().astype(np.float32)  # (B,3)
        pre_p_w = pre_p_w + origins
        gra_p_w = gra_p_w + origins

        pre_pb, pre_qb = self._to_base(pre_p_w, q_w)
        gra_pb, gra_qb = self._to_base(gra_p_w, q_w)

        return {
            "pre_p_w": pre_p_w, "p_w": gra_p_w, "q_w": q_w,
            "pre_p_b": pre_pb,  "pre_q_b": pre_qb,
            "p_b": gra_pb,      "q_b": gra_qb,
            "pre_dist": pre_d,  "delta": delt,
        }


    def grasp_trials(self, gg, std: float = 0.0005):

        B = self.scene.num_envs
        idx_all = list(range(len(gg)))
        if len(idx_all) == 0:
            print("[ERR] empty grasp list.")
            return False

        rng = np.random.default_rng()

        pre_dist_const = 0.12  # m

        success = False
        chosen_pose_w = None    # (p_w, q_w)
        chosen_pre    = None
        chosen_delta  = None

        # assign different grasp proposals to different envs
        for start in range(0, len(idx_all), B):
            block = idx_all[start : start + B]
            if len(block) < B:
                block = block + [block[-1]] * (B - len(block))

            grasp_pos_w_batch, grasp_quat_w_batch = [], []
            for idx in block:
                p_w, q_w = grasp_to_world(gg[int(idx)])
                grasp_pos_w_batch.append(p_w.astype(np.float32))
                grasp_quat_w_batch.append(q_w.astype(np.float32))
            grasp_pos_w_batch  = np.stack(grasp_pos_w_batch,  axis=0)  # (B,3)
            grasp_quat_w_batch = np.stack(grasp_quat_w_batch, axis=0)  # (B,4)
            self.show_goal(grasp_pos_w_batch, grasp_quat_w_batch)
            # random disturbance along approach axis
            pre_dist_batch = np.full((B,), pre_dist_const, dtype=np.float32)
            delta_batch    = rng.normal(0.0, std, size=(B,)).astype(np.float32)

            info = self.build_grasp_info(grasp_pos_w_batch, grasp_quat_w_batch,
                                          pre_dist_batch, delta_batch)

            ok_batch, score_batch = self.execute_and_lift_once_batch(info)
            ok_cnt = int(ok_batch.sum())
            print(f"[SEARCH] block[{start}:{start+B}] -> success {ok_cnt}/{B}")
            if ok_cnt > 0:
                winner = int(np.argmax(score_batch))
                chosen_pose_w = (grasp_pos_w_batch[winner], grasp_quat_w_batch[winner])
                chosen_pre    = float(pre_dist_batch[winner])
                chosen_delta  = float(delta_batch[winner])
                success = True
                return {
                    "success": success,
                    "chosen_pose_w": chosen_pose_w,
                    "chosen_pre": chosen_pre,
                    "chosen_delta": chosen_delta,
                }

        if not success:
            print("[ERR] no proposal succeeded to lift after full search.")
            return {
                "success": success,
                "chosen_pose_w": None,
                "chosen_pre": None,
                "chosen_delta": None,
            }

    def replay_actions(self, actions: np.ndarray):
        """
        Replay a sequence of recorded actions: (p[B,3], q[B,4], gripper[B,1])
        """
        n_steps = actions.shape[0]

        self.reset()
        self.wait(gripper_open=True, steps=10)

        for t in range(n_steps):
            print(f"[INFO] replay step {t}/{n_steps}")
            act = actions[t:t+1]
            p_b = torch.as_tensor(act[:, 0:3], dtype=torch.float32, device=self.sim.device)
            q_b = torch.as_tensor(act[:, 3:7], dtype=torch.float32, device=self.sim.device)
            g_b = act[:, 7] < 0.5
            jp, success = self.move_to(p_b, q_b, gripper_open=g_b)
            if torch.any(success==False):
                print(f"[ERR] replay step {t} failed.")
                return False
            jp = self.wait(gripper_open=g_b, steps=3)
        return True

    def inference(self, std: float = 0.0) -> bool:
        """
        Main function of the heuristic manipulation policy.
        Physical trial-and-error grasping with approach-axis perturbation:
          • Multiple grasp proposals executed in parallel;
          • Every attempt does reset → pre → grasp → close → lift → check;
          • Early stops when any env succeeds; then re-exec for logging.
        """
        B = self.scene.num_envs

        self.wait(gripper_open=True, steps=10)

        # read grasp proposals
        npy_path = self.grasp_path
        if npy_path is None or (not os.path.exists(npy_path)):
            print(f"[ERR] grasps npy not found: {npy_path}")
            return False
        gg = GraspGroup().from_npy(npy_file_path=npy_path)
        gg = get_best_grasp_with_hints(gg, point=None, direction=[0, 0, -1]) # [0, 0, -1]

        if self.grasp_idx >= 0:
            if self.grasp_idx >= len(gg):
                print(f"[ERR] grasp_idx {self.grasp_idx} out of range [0,{len(gg)})")
                return False
            print(f"[INFO] using fixed grasp index {self.grasp_idx} for all envs.")
            p_w, q_w = grasp_to_world(gg[int(self.grasp_idx)])
            ret = {
                "success": True,
                "chosen_pose_w": (p_w.astype(np.float32), q_w.astype(np.float32)),
                "chosen_pre": self.grasp_pre if self.grasp_pre is not None else 0.12,
                "chosen_delta": self.grasp_delta if self.grasp_delta is not None else 0.0,
            }
            print(f"[INFO] grasp delta (m): {ret['chosen_delta']:.4f}")
        else:
            ret = self.grasp_trials(gg, std=std)

        print("[INFO] Re-exec all envs with the winning grasp, then follow object goals.")

        p_win, q_win = ret["chosen_pose_w"]
        p_all   = np.repeat(p_win.reshape(1, 3), B, axis=0)
        q_all   = np.repeat(q_win.reshape(1, 4), B, axis=0)
        pre_all = np.full((B,), ret["chosen_pre"],   dtype=np.float32)
        del_all = np.full((B,), ret["chosen_delta"], dtype=np.float32)

        info_all = self.build_grasp_info(p_all, q_all, pre_all, del_all)

        # reset and conduct main process: open→pre→grasp→close→follow_object_goals
        self.reset()

        cam_p = self.camera.data.pos_w
        cam_q = self.camera.data.quat_w_ros
        gp_w  = torch.as_tensor(info_all["p_w"],     dtype=torch.float32, device=self.sim.device)
        gq_w  = torch.as_tensor(info_all["q_w"],     dtype=torch.float32, device=self.sim.device)
        pre_w = torch.as_tensor(info_all["pre_p_w"], dtype=torch.float32, device=self.sim.device)
        gp_cam,  gq_cam  = subtract_frame_transforms(cam_p, cam_q, gp_w,  gq_w)
        pre_cam, pre_qcm = subtract_frame_transforms(cam_p, cam_q, pre_w, gq_w)
        self.save_dict["grasp_pose_cam"]    = torch.cat([gp_cam,  gq_cam],  dim=1).unsqueeze(0).cpu().numpy()
        self.save_dict["pregrasp_pose_cam"] = torch.cat([pre_cam, pre_qcm], dim=1).unsqueeze(0).cpu().numpy()

        jp = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        self.wait(gripper_open=True, steps=4)

        # pre → grasp
        jp, success = self.move_to(info_all["pre_p_b"], info_all["pre_q_b"], gripper_open=True)
        if torch.any(success==False): return False
        self.save_dict["actions"].append(np.concatenate([info_all["pre_p_b"].cpu().numpy(), info_all["pre_q_b"].cpu().numpy(), np.zeros((B, 1))], axis=1))
        jp = self.wait(gripper_open=True, steps=3)

        jp, success = self.move_to(info_all["p_b"], info_all["q_b"], gripper_open=True)
        if torch.any(success==False): return False
        self.save_dict["actions"].append(np.concatenate([info_all["p_b"].cpu().numpy(), info_all["q_b"].cpu().numpy(), np.zeros((B, 1))], axis=1))

        # close gripper
        jp = self.wait(gripper_open=False, steps=50)
        self.save_dict["actions"].append(np.concatenate([info_all["p_b"].cpu().numpy(), info_all["q_b"].cpu().numpy(), np.ones((B, 1))], axis=1))

        # object goal following
        # self.lift_up(height=0.05, gripper_open=False)
        jp = self.follow_object_goals(jp, sample_step=5, visualize=True)

        self.save_data(ignore_keys=["segmask", "depth"])
        return True

# ──────────────────────────── Entry Point ────────────────────────────
def main():
    sim_cfgs = load_sim_parameters(BASE_DIR, args_cli.key)
    env, _ = make_env(
        cfgs=sim_cfgs, num_envs=args_cli.num_envs,
        device=args_cli.device,
        bg_simplify=False,
    )
    sim, scene = env.sim, env.scene

    my_sim = HeuristicManipulation(sim, scene, sim_cfgs=sim_cfgs)

    demo_root = (out_dir / img_folder / "demos").resolve()

    for _ in range(args_cli.num_trials):

        robot_pose = torch.tensor(sim_cfgs["robot_cfg"]["robot_pose"], dtype=torch.float32, device=my_sim.sim.device)  # [7], pos(3)+quat(wxyz)(4)
        my_sim.set_robot_pose(robot_pose)
        my_sim.demo_id = get_next_demo_id(demo_root)
        my_sim.reset()
        print(f"[INFO] start simulation demo_{my_sim.demo_id}")
        # Note: if you set viz_object_goals(), remember to disable gravity and collision for object
        # my_sim.viz_object_goals(sample_step=10, hold_steps=40)
        # my_sim.inference()
        actions = np.load("outputs/lab16/demos/demo_5/env_000/actions.npy")
        my_sim.replay_actions(actions)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
    os.system("quit()")
    simulation_app.close()
