from __future__ import annotations

from typing import Any, Dict, Optional


# ─────────── AppLauncher ───────────
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import torch

# ─────────── CLI ───────────
parser = argparse.ArgumentParser(description="Test scene and object loading")
parser.add_argument(
    "--key",
    type=str,
    default="demo_video",
    help="Scene key (outputs/<key>/simulation/scene.json)",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=100,
    help="Number of simulation steps to run",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False  # No camera needed
args_cli.headless = True  # Headless mode for batch execution
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app
import isaaclab.sim as sim_utils
# ─────────── Runtime imports ───────────
# Import here after AppLauncher to avoid issues if isaaclab is not available
from sim_env_factory_calib import make_env, load_scene_json


class BaseSimulator:
    """
    Base class for object simulation.

    Attributes:
        sim: Simulation context.
        scene: Interactive scene.
        sim_dt: Simulation timestep.
        object_prim: Primary object primitive.
        other_object_prims: List of other object primitives.
        background_prim: Background primitive.
        count: Step counter.
        _all_env_ids: All environment IDs tensor.
    """

    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        scene: Any,  # InteractiveScene
        *,
        args: Optional[Any] = None,
        sim_cfgs: Dict,
        set_physics_props: bool = True,
        debug_level: int = 1,
    ) -> None:
        """Initialize the base simulator.

        Args:
            sim: Simulation context.
            scene: Interactive scene containing objects.
            args: Optional arguments (for compatibility).
            sim_cfgs: Simulation configuration dictionary.
            set_physics_props: Whether to set physics properties on objects.
            debug_level: Debug level (default: 1).
        """
        # Basic simulation setup
        self.sim: sim_utils.SimulationContext = sim
        self.sim_cfgs = sim_cfgs
        self.scene = scene
        self.sim_dt = sim.get_physics_dt()
        self.debug_level = debug_level

        # Scene entities
        self.object_prim = scene["object_00"]
        self.other_object_prims = [
            scene[key]
            for key in scene.keys()
            if "object_" in key and key != "object_00"
        ]
        self.background_prim = scene["background"]

        # Initialize oid to object prim mapping and pose/com tracking
        self.oid_to_prim = {}  # oid (int) -> RigidObject
        self.oid_to_pose_com = {}  # oid (int) -> {"pose": tensor[B,7], "com_center": tensor[B,3]}
        
        # Build oid to prim mapping from scene.json
        # Objects in scene are named object_00, object_01, etc., ordered by their appearance in scene.json
        if "scene_cfg" in sim_cfgs:
            scene_cfg = sim_cfgs["scene_cfg"]
            if "objects" in scene_cfg:
                # Get all object keys from scene, sorted to match order
                # Filter keys that start with "object_" to avoid other entities
                all_scene_keys = list(scene.keys())
                obj_keys = sorted([k for k in all_scene_keys if isinstance(k, str) and k.startswith("object_")])
                
                # Get objects from scene.json, sorted by their key (oid string)
                objects_items = sorted(scene_cfg["objects"].items(), key=lambda x: int(x[0]))
                
                # Map each oid to corresponding scene object prim
                for idx, (oid_str, obj_info) in enumerate(objects_items):
                    oid = int(obj_info.get("oid", oid_str))
                    if idx < len(obj_keys):
                        scene_key = obj_keys[idx]
                        # Use try-except to safely access scene entity
                        try:
                            prim = scene[scene_key]
                            self.oid_to_prim[oid] = prim
                        except (KeyError, TypeError):
                            # Skip if scene_key doesn't exist or is invalid
                            continue

        # Initialize counters and IDs
        self.count = 0
        self._all_env_ids = torch.arange(
            scene.num_envs, device=scene.device, dtype=torch.long
        )

        # Physics properties
        if set_physics_props:
            static_friction = 5.0
            dynamic_friction = 5.0
            restitution = 0.0

            # Object: rigid prim -> has root_physx_view
            if (
                hasattr(self.object_prim, "root_physx_view")
                and self.object_prim.root_physx_view is not None
            ):
                obj_view = self.object_prim.root_physx_view
                obj_mats = obj_view.get_material_properties()
                vals_obj = torch.tensor(
                    [static_friction, dynamic_friction, restitution],
                    device=obj_mats.device,
                    dtype=obj_mats.dtype,
                )
                obj_mats[:] = vals_obj
                obj_view.set_material_properties(
                    obj_mats, self._all_env_ids.to(obj_mats.device)
                )

    # ---------- Environment Step ----------
    def step(self) -> None:
        """Perform one simulation step."""
        self.scene.write_data_to_sim()
        self.sim.step()
        self.count += 1
        self.scene.update(self.sim_dt)
        # Update object poses and com centers
        self._update_object_poses_coms()

    # ---------- Reset Envs ----------
    def reset(self, env_ids: Optional[Any] = None) -> None:
        """Reset all environments or only those in env_ids.

        Args:
            env_ids: Optional list of environment IDs to reset. If None, resets all.
        """
        device = self.object_prim.device
        if env_ids is None:
            env_ids_t = self._all_env_ids.to(device)  # (B,)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long).view(
                -1
            )  # (M,)
        M = int(env_ids_t.shape[0])

        # Object pose/vel: set object at env origins with identity quaternion
        env_origins = self.scene.env_origins.to(device)[env_ids_t]  # (M, 3)
        object_pose = torch.zeros((M, 7), device=device, dtype=torch.float32)
        object_pose[:, :3] = env_origins
        object_pose[:, 3] = 1.0  # wxyz = [1, 0, 0, 0]
        self.object_prim.write_root_pose_to_sim(object_pose, env_ids=env_ids_t)
        self.object_prim.write_root_velocity_to_sim(
            torch.zeros((M, 6), device=device, dtype=torch.float32), env_ids=env_ids_t
        )
        self.object_prim.write_data_to_sim()

        # Reset other objects
        for prim in self.other_object_prims:
            prim.write_root_pose_to_sim(object_pose, env_ids=env_ids_t)
            prim.write_root_velocity_to_sim(
                torch.zeros((M, 6), device=device, dtype=torch.float32),
                env_ids=env_ids_t,
            )
            prim.write_data_to_sim()
        
        # Update object poses and com centers after reset
        self._update_object_poses_coms()

    def _update_object_poses_coms(self) -> None:
        """Update the pose and com center for all tracked objects."""
        for oid, prim in self.oid_to_prim.items():
            if hasattr(prim, "data"):
                # Get root state (pose): [B, 7] (pos[3] + quat[4] in wxyz format)
                if hasattr(prim.data, "root_state_w"):
                    pose = prim.data.root_state_w[:, :7]  # [B, 7]
                elif hasattr(prim.data, "root_com_state_w"):
                    pose = prim.data.root_com_state_w[:, :7]  # [B, 7]
                else:
                    continue
                
                # Get com center: [B, 3]
                if hasattr(prim.data, "root_com_pos_w"):
                    com_center = prim.data.root_com_pos_w[:, :3]  # [B, 3]
                elif hasattr(prim.data, "root_state_w"):
                    com_center = prim.data.root_state_w[:, :3]  # [B, 3]
                else:
                    continue
                
                self.oid_to_pose_com[oid] = {
                    "pose": pose.clone(),  # [B, 7]
                    "com_center": com_center.clone(),  # [B, 3]
                }

    def get_object_pose(self, oid: int, env_id: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Get object pose for a given oid.
        
        Args:
            oid: Object ID
            env_id: Optional environment ID. If None, returns all envs [B, 7]
        
        Returns:
            Pose tensor [7] if env_id specified, or [B, 7] if None
        """
        if oid not in self.oid_to_pose_com:
            return None
        pose = self.oid_to_pose_com[oid]["pose"]  # [B, 7]
        if env_id is not None:
            return pose[env_id]  # [7]
        return pose  # [B, 7]

    def get_object_com_center(self, oid: int, env_id: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Get object com center for a given oid.
        
        Args:
            oid: Object ID
            env_id: Optional environment ID. If None, returns all envs [B, 3]
        
        Returns:
            COM center tensor [3] if env_id specified, or [B, 3] if None
        """
        if oid not in self.oid_to_pose_com:
            return None
        com_center = self.oid_to_pose_com[oid]["com_center"]  # [B, 3]
        if env_id is not None:
            return com_center[env_id]  # [3]
        return com_center  # [B, 3]

    def get_all_object_poses_coms(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Get all tracked object poses and com centers.
        
        Returns:
            Dictionary mapping oid -> {"pose": [B, 7], "com_center": [B, 3]}
        """
        return self.oid_to_pose_com.copy()

    def save_object_poses_to_scene_json(self, key: str, env_id: int = 0) -> None:
        """
        Save all tracked object poses to scene.json file.
        Also updates object_min, object_max, and object_center by transforming
        the optimized mesh with the current pose.
        
        Args:
            key: Scene key (e.g., "demo_video")
            env_id: Environment ID to save pose from (default: 0)
        """
        import json
        import numpy as np
        from pathlib import Path
        import trimesh
        
        # Update poses before saving
        self._update_object_poses_coms()
        
        # Load current scene.json
        scene_json_path = Path.cwd() / "outputs" / key / "simulation" / "scene.json"
        if not scene_json_path.exists():
            raise FileNotFoundError(f"Scene JSON not found: {scene_json_path}")
        
        with open(scene_json_path, "r", encoding="utf-8") as f:
            scene_dict = json.load(f)
        
        # Update poses for each object
        if "objects" not in scene_dict:
            scene_dict["objects"] = {}
        
        for oid, pose_com_data in self.oid_to_pose_com.items():
            oid_str = str(oid)
            
            # Get pose for the specified environment
            pose = pose_com_data["pose"]  # [B, 7]
            
            if env_id < pose.shape[0]:
                pose_env = pose[env_id].cpu().numpy()  # [7]: [x, y, z, w, x, y, z]
            else:
                # If env_id is out of range, use first environment
                pose_env = pose[0].cpu().numpy()
            
            # Ensure object entry exists (preserve existing fields if any)
            if oid_str not in scene_dict["objects"]:
                scene_dict["objects"][oid_str] = {}
            
            # Update or add pose field only, preserving all other existing fields
            # pose: [x, y, z, w, x, y, z] (position + quaternion in wxyz format)
            scene_dict["objects"][oid_str]["pose"] = pose_env.tolist()
            
            # Update object_min, object_max, and object_center by transforming optimized mesh
            # Backup original optimized file as "original", then overwrite optimized with transformed mesh
            if "optimized" in scene_dict["objects"][oid_str]:
                optimized_path = scene_dict["objects"][oid_str]["optimized"]
                # Handle absolute paths (remove /app prefix if present)
                if optimized_path.startswith("/app/"):
                    optimized_path_abs = Path.cwd() / optimized_path[5:]  # Remove /app prefix
                elif Path(optimized_path).is_absolute():
                    optimized_path_abs = Path(optimized_path)
                else:
                    # Relative path, make it relative to current working directory
                    optimized_path_abs = Path.cwd() / optimized_path
                
                if optimized_path_abs.exists():
                    try:
                        # Step 1: Backup original optimized file and object parameters (only if not already backed up)
                        simulation_dir = optimized_path_abs.parent
                        obj_name = scene_dict["objects"][oid_str].get("name", f"object_{oid_str}")
                        original_filename = f"{oid_str}_{obj_name}_original.glb"
                        original_path = simulation_dir / original_filename
                        
                        # Only backup if original doesn't exist yet
                        if not original_path.exists():
                            import shutil
                            shutil.copy2(optimized_path_abs, original_path)
                            # Update scene.json with original path
                            original_path_str = f"/app/{original_path.relative_to(Path.cwd())}"
                            scene_dict["objects"][oid_str]["original"] = original_path_str
                            
                            # Backup original object parameters (min, max, center)
                            if "object_min" in scene_dict["objects"][oid_str]:
                                scene_dict["objects"][oid_str]["original_object_min"] = scene_dict["objects"][oid_str]["object_min"]
                            if "object_max" in scene_dict["objects"][oid_str]:
                                scene_dict["objects"][oid_str]["original_object_max"] = scene_dict["objects"][oid_str]["object_max"]
                            if "object_center" in scene_dict["objects"][oid_str]:
                                scene_dict["objects"][oid_str]["original_object_center"] = scene_dict["objects"][oid_str]["object_center"]
                            
                            print(f"[INFO] Backed up original optimized mesh and parameters to {original_path}")
                        
                        # Step 2: Load optimized mesh
                        mesh = trimesh.load(str(optimized_path_abs))
                        
                        # Step 3: Extract position and quaternion from pose
                        position = pose_env[:3]  # [x, y, z]
                        quaternion_wxyz = pose_env[3:]  # [w, x, y, z]
                        
                        # Step 4: Convert quaternion (wxyz) to rotation matrix
                        # scipy.spatial.transform.Rotation.from_quat accepts [x, y, z, w] format
                        from scipy.spatial.transform import Rotation
                        quaternion_xyzw = np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]])  # [x, y, z, w]
                        rotation = Rotation.from_quat(quaternion_xyzw)
                        rotation_matrix = rotation.as_matrix()  # 3x3 rotation matrix
                        
                        # Step 5: Create 4x4 transformation matrix
                        transform_matrix = np.eye(4)
                        transform_matrix[:3, :3] = rotation_matrix
                        transform_matrix[:3, 3] = position
                        
                        # Step 6: Apply transform to mesh
                        mesh.apply_transform(transform_matrix)
                        
                        # Step 7: Calculate bounds after transform
                        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
                        object_min = bounds[0].tolist()  # [min_x, min_y, min_z]
                        object_max = bounds[1].tolist()  # [max_x, max_y, max_z]
                        object_center = ((bounds[0] + bounds[1]) / 2.0).tolist()  # [center_x, center_y, center_z]
                        
                        # Step 8: Overwrite optimized file with transformed mesh
                        mesh.export(str(optimized_path_abs))
                        
                        # Step 9: Update scene.json with bounds
                        scene_dict["objects"][oid_str]["object_min"] = object_min
                        scene_dict["objects"][oid_str]["object_max"] = object_max
                        scene_dict["objects"][oid_str]["object_center"] = object_center
                        
                        print(f"[INFO] Overwrote optimized mesh for object {oid_str} with transformed version")
                        print(f"[INFO] Updated bounds: min={object_min}, max={object_max}, center={object_center}")
                    except Exception as e:
                        print(f"[WARN] Failed to process mesh for object {oid_str}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[WARN] Optimized mesh not found: {optimized_path_abs}")
        
        # Save updated scene.json
        with open(scene_json_path, "w", encoding="utf-8") as f:
            json.dump(scene_dict, f, indent=2)
        
        print(f"[INFO] Saved object poses and bounds to {scene_json_path}")


# --------------------------------------------------------------------------------------
# Test main function
# --------------------------------------------------------------------------------------


def main():
    """Test function to load scene and objects."""
    print(f"[INFO] Loading scene from outputs/{args_cli.key}/simulation/scene.json")
    
    # Load scene configuration
    scene_cfg = load_scene_json(args_cli.key)
    
    # Create physics configuration (default values)
    physics_cfg = {
        "obj_physics": None,  # Use defaults from scene.json
        "bg_physics": None,   # Use defaults from scene.json
    }
    
    # Create configuration dictionary
    cfgs = {
        "scene_cfg": scene_cfg,
        "physics_cfg": physics_cfg,
    }

    print(f"[INFO] Creating environment with {args_cli.num_envs} env(s) on {args_cli.device}")
    
    # Create environment
    env, env_cfg = make_env(
        cfgs=cfgs,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        bg_simplify=False,
    )
    
    sim = env.sim
    scene = env.scene

    print(f"[INFO] Creating BaseSimulator")
    
    # Create simulator
    simulator = BaseSimulator(
        sim=sim,
        scene=scene,
        sim_cfgs=cfgs,
        set_physics_props=True,
        debug_level=1,
    )

    print(f"[INFO] Scene loaded successfully!")
    print(f"  - Number of environments: {scene.num_envs}")
    # Get object names from scene keys (RigidObject doesn't have .name attribute)
    object_keys = [key for key in scene.keys() if "object_" in key]
    primary_object_key = "object_00" if "object_00" in scene.keys() else (object_keys[0] if object_keys else "unknown")
    print(f"  - Primary object: {primary_object_key}")
    print(f"  - Other objects: {len(simulator.other_object_prims)}")
    background_key = "background" if "background" in scene.keys() else "unknown"
    print(f"  - Background: {background_key}")
    
    # Print tracked object oids and initialize pose/com tracking
    if simulator.oid_to_prim:
        print(f"  - Tracked object OIDs: {sorted(simulator.oid_to_prim.keys())}")
        # Update once to initialize the pose/com data
        simulator._update_object_poses_coms()
        print(f"  - Object pose/com tracking initialized")
        # Print example pose and com for first object
        first_oid = sorted(simulator.oid_to_prim.keys())[0]
        pose = simulator.get_object_pose(first_oid, env_id=0)
        com = simulator.get_object_com_center(first_oid, env_id=0)
        if pose is not None and com is not None:
            print(f"  - Example (OID {first_oid}, env 0): pose={pose.cpu().numpy()}, com={com.cpu().numpy()}")

    # Reset environment
    print(f"[INFO] Resetting environment...")
    simulator.reset()

    # Run simulation steps
    print(f"[INFO] Running {args_cli.num_steps} simulation steps...")
    for i in range(args_cli.num_steps):
        simulator.step()
        if (i + 1) % 10 == 0:
            print(f"  Step {i + 1}/{args_cli.num_steps}")

    print(f"[INFO] Test completed successfully!")
    print(f"  - Total steps executed: {simulator.count}")
    
    # Save object poses to scene.json
    print(f"[INFO] Saving object poses to scene.json...")
    try:
        simulator.save_object_poses_to_scene_json(args_cli.key, env_id=0)
        print(f"[INFO] Object poses saved successfully!")
    except Exception as e:
        print(f"[WARN] Failed to save object poses: {e}")
    
    # Clean up
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
