import mplib
import numpy as np
import sapien
import trimesh
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.pose import to_sapien_pose as to_sapien_pose_mani_skill
from mani_skill.utils.structs.actor import Actor
from transforms3d import quaternions
from transforms3d.quaternions import mat2quat
import torch


class BaseMotionPlanningSolver:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose_mani_skill(base_pose)

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info

        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        move_group = self.MOVE_GROUP if hasattr(self, "MOVE_GROUP") else "eef"
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=move_group,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        planner.joint_vel_limits = (
            np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        )
        planner.joint_acc_limits = (
            np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        )
        return planner

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        return None

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        return target

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel])
            else:
                action = np.hstack([qpos])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def move_to_pose_with_RRTStar(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose_mani_skill(pose)
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            rrt_range=0.0,
            planning_time=1,
            planner_name="RRTstar",
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose_mani_skill(pose)
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose_mani_skill(pose)
        # try screw two times before giving up
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass


class TwoFingerGripperMotionPlanningSolver(BaseMotionPlanningSolver):
    OPEN = 1
    CLOSED = -1

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        super().__init__(
            env,
            debug,
            vis,
            base_pose,
            print_env_info,
            joint_vel_limits,
            joint_acc_limits,
        )
        self.gripper_state = self.OPEN
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_two_finger_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp_pose)

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target)

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def open_gripper(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.OPEN
        self.gripper_state = gripper_state
        qpos = (
            self.robot.get_qpos()[0, : len(self.planner.joint_vel_limits)].cpu().numpy()
        )
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.CLOSED
        self.gripper_state = gripper_state
        qpos = (
            self.robot.get_qpos()[0, : len(self.planner.joint_vel_limits)].cpu().numpy()
        )
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info


def build_two_finger_gripper_grasp_pose_visual(scene: ManiSkillScene):
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7]),
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual


class HeuristicManipulationAgent:
    """
    An agent that performs heuristic grasp trials and follows a trajectory.
    """

    def __init__(
        self,
        env: BaseEnv,
        planner: "PandaArmMotionPlanningSolver",
        lift_height: float = 0.1,
    ):
        self.env = env
        self.planner = planner
        self.lift_height = lift_height

    def check_grasp_success(self, target_object_id: str) -> bool:
        """
        Checks if the object is successfully grasped by using the agent's
        built-in contact force checking method.
        """
        target_object: Actor = self.env.unwrapped.object_actors[target_object_id]
        agent: BaseAgent = self.env.unwrapped.agent

        # This check needs to be done over a few steps to be reliable
        for _ in range(5):
            self.env.step(None)  # Let physics settle

        is_grasping = agent.is_grasping(target_object)

        if is_grasping:
            print("Contact forces detected. Grasp is likely successful.")
            # Lift the gripper to confirm
            initial_gripper_pose = self.env.unwrapped.agent.tcp.pose
            lift_pose = (
                Pose.create_from_pq(p=[0, 0, self.lift_height]) * initial_gripper_pose
            )
            self.planner.move_to_pose_with_screw(lift_pose)
        else:
            # TODO: check the pre_grasp_pose to see if it is close to the object
            print("No significant contact forces detected. Grasp failed.")
            initial_gripper_pose = self.env.unwrapped.agent.tcp.pose
            # Move back to a safe position
            pre_grasp_pose = Pose.create_from_pq(
                p=initial_gripper_pose.p + np.array([0, 0, 0.1]).reshape(1, 3),
                q=initial_gripper_pose.q,
            )
            self.planner.open_gripper()
            self.planner.move_to_pose_with_RRTConnect(pre_grasp_pose)

        return is_grasping

    def attempt_grasp(
        self, grasp_pose_world: Pose, pre_grasp_offset: float = 0.1
    ) -> None:
        """
        Executes a grasp attempt from a pre-grasp position.
        """
        # Calculate pre-grasp pose (offset along the grasp's +X axis)
        # unsqueeze to 4x4 matrix (N, 4, 4) -> (4, 4)
        approach_dir = grasp_pose_world.to_transformation_matrix().reshape(4, 4)[:3, 0]

        pre_grasp_p = grasp_pose_world.p - approach_dir * pre_grasp_offset
        pre_grasp_pose = Pose.create_from_pq(p=pre_grasp_p, q=grasp_pose_world.q)

        # Execute the motion sequence
        self.planner.open_gripper()
        print("Moving to pre-grasp pose...")
        self.planner.move_to_pose_with_RRTConnect(pre_grasp_pose)
        print("Moving to grasp pose...")
        self.planner.move_to_pose_with_screw(grasp_pose_world)
        self.planner.close_gripper()

    def follow_trajectory(self, target_object_id: str, trajectory: list[Pose]) -> None:
        """
        Follows a given 6D object trajectory.
        """
        target_object: Actor = self.env.unwrapped.object_actors[target_object_id]
        initial_ee_pose_world: Pose = self.env.unwrapped.agent.tcp.pose
        initial_obj_pose_world: Pose = target_object.pose

        # Calculate the fixed transform from the object to the EE frame
        T_world_obj = initial_obj_pose_world.to_transformation_matrix()
        T_world_ee = initial_ee_pose_world.to_transformation_matrix()

        T_obj_ee = torch.linalg.inv(T_world_obj) @ T_world_ee

        print("Starting trajectory following...")
        for i, target_obj_pose_world in enumerate(trajectory):
            print(f"  Waypoint {i + 1}/{len(trajectory)}")
            T_world_obj_target = target_obj_pose_world.to_transformation_matrix()
            T_world_ee_target = T_world_obj_target @ T_obj_ee
            T_world_ee_target = T_world_ee_target.reshape(4, 4)

            # turn into numpy array
            T_world_ee_target = T_world_ee_target.cpu().numpy()

            ee_target_pose = Pose.create_from_pq(
                p=T_world_ee_target[:3, 3],
                q=mat2quat(T_world_ee_target[:3, :3]),
            )

            self.planner.move_to_pose_with_screw(ee_target_pose, refine_steps=0)

        print("Trajectory following complete.")


class PandaArmMotionPlanningSolver(TwoFingerGripperMotionPlanningSolver):
    OPEN = 1
    CLOSED = -1
    MOVE_GROUP = "panda_hand_tcp"

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        super().__init__(
            env,
            debug,
            vis,
            base_pose,
            visualize_target_grasp_pose,
            print_env_info,
            joint_vel_limits,
            joint_acc_limits,
        )
