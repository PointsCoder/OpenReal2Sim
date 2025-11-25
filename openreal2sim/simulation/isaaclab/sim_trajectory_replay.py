"""
Replay existing trajectories in Isaac Lab simulation.
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
parser.add_argument("--demo_dir", type=str, help="directory of robot trajectory")
parser.add_argument("--key", type=str, default=None, help="scene key (outputs/<key>)")
parser.add_argument("--robot", type=str, default="franka_panda")
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
from sim_preprocess.grasp_utils import get_best_grasp_with_hints
from sim_utils.transform_utils import pose_to_mat, mat_to_pose, grasp_to_world, grasp_approach_axis_batch
from sim_utils.sim_utils import load_sim_parameters

BASE_DIR   = Path.cwd()
sim_cfgs = json.load(open(Path(args_cli.demo_dir) / "config.json", "r"))
args_cli.key = sim_cfgs["key"]
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
            sim=sim, sim_cfgs=sim_cfgs, scene=scene, args=args_cli,
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

    # ---------- Helpers ----------
    def _to_base(self, pos_w: np.ndarray | torch.Tensor, quat_w: np.ndarray | torch.Tensor):
        """World → robot base frame for all envs."""
        root = self.robot.data.root_state_w[:, 0:7]  # [B,7]
        p_w, q_w = self._ensure_batch_pose(pos_w, quat_w)
        pb, qb = subtract_frame_transforms(
            root[:, 0:3], root[:, 3:7], p_w, q_w
        )
        return pb, qb  # [B,3], [B,4]

    def replay_actions(self, actions: np.ndarray):
        """
        Replay a sequence of recorded actions: (p[B,3], q[B,4], gripper[B,1])
        """
        n_steps = actions.shape[0]

        self.reset()
        self.wait(gripper_open=True, steps=10)

        prev_gripper_open = True
        for t in range(n_steps):
            print(f"[INFO] replay step {t}/{n_steps}")
            act = actions[t:t+1]
            p_b = torch.as_tensor(act[:, 0:3], dtype=torch.float32, device=self.sim.device)
            q_b = torch.as_tensor(act[:, 3:7], dtype=torch.float32, device=self.sim.device)
            g_b = bool(act[0, 7] < 0.5)  # True = open, False = closed

            jp, success = self.move_to(p_b, q_b, gripper_open=g_b)
            if torch.any(success==False):
                print(f"[ERR] replay step {t} failed.")
                return False

            # If gripper just closed (transition from open to closed), wait longer for full grasp
            if prev_gripper_open and not g_b:
                print(f"[INFO] Gripper closing detected at step {t}, waiting 50 steps for grasp...")
                jp = self.wait(gripper_open=g_b, steps=50)
            else:
                jp = self.wait(gripper_open=g_b, steps=3)

            prev_gripper_open = g_b
        return True

# ──────────────────────────── Entry Point ────────────────────────────
def main():
    sim_cfgs["robot_type"] = args_cli.robot
    env, _ = make_env(
        cfgs=sim_cfgs, num_envs=args_cli.num_envs,
        device=args_cli.device,
        bg_simplify=False,
        robot_type=args_cli.robot,
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
        actions = np.load(Path(args_cli.demo_dir) / "actions.npy")
        my_sim.replay_actions(actions)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
    os.system("quit()")
    simulation_app.close()
