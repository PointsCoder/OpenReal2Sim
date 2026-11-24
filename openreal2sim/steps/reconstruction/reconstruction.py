#!/usr/bin/env python3
"""Run the reconstruction steps for every key in the config, or for a single key."""

import argparse
import contextlib
import json
import os
import pickle
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml

# The runner can be started as a script, so put the repository root on sys.path.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openreal2sim.utils.compose_config import compose_configs
from openreal2sim.utils.notification import notify_started, notify_failed, notify_success

# Opt-in step: replaces the manual annotator, selected by name only, so the
# default run keeps using the masks a human already annotated.
OPTIONAL_STEPS = ["auto_segmentation"]

# Steps that only run when their reconstruction config flag is on, or when --stage names
# them. background_mesh_generation completes the depth of the object region with the ground
# plane itself, so the point completion is only needed for a non-planar background
# (bg_completion_mode: "moge").
FLAGGED_STEPS = {"background_point_inpainting": "bg_point_inpainting"}

STEPS = [
    "mask_propagation",
    "background_pixel_inpainting",
    "background_point_inpainting",
    "background_mesh_generation",
    "object_mesh_generation",
    "scenario_construction",
    "scenario_fdpose_optimization",
    "scenario_collision_optimization",
]


def update_metadata(metadata_path: Path, reconstruction_time: int, status: str = "success"):
    """Create or update the metadata YAML of a reconstruction run."""
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                existing = yaml.safe_load(f) or {}
                metadata.update(existing)
        except Exception:
            pass
    metadata["reconstruction_time"] = reconstruction_time
    metadata["reconstruction_status"] = status
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f)


@contextlib.contextmanager
def key_logs(key: str):
    """Send stdout and stderr, including the ones of child processes, to the key log files."""
    timestamp = datetime.now().strftime("%m-%d-%H:%M")
    log_dir = Path("outputs") / key / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{timestamp}_stdout.log"
    stderr_log = log_dir / f"{timestamp}_stderr.log"
    saved_out, saved_err = os.dup(1), os.dup(2)
    with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(out.fileno(), 1)
        os.dup2(err.fileno(), 2)
        try:
            yield stdout_log, stderr_log
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


class ReconAgent:
    def __init__(self, keys, cfg, stage=None):
        print('[Info] Initializing ReconAgent...')
        self.base_dir = Path.cwd()
        self.keys = keys
        self.key_cfgs = {key: compose_configs(key, cfg, "reconstruction") for key in self.keys}
        self.key_scene_dicts = {}
        for key in self.keys:
            scene_pkl = self.base_dir / f'outputs/{key}/scene/scene.pkl'
            with open(scene_pkl, 'rb') as f:
                scene_dict = pickle.load(f)
            self.key_scene_dicts[key] = scene_dict
        self.steps = list(STEPS)
        if stage is not None:
            if stage in OPTIONAL_STEPS:
                self.steps = [stage] + self.steps
            elif stage in self.steps:
                self.steps = self.steps[self.steps.index(stage):]
            else:
                print(f"[Warning] Step '{stage}' not found. It must be one of the {OPTIONAL_STEPS + self.steps}. Running all steps by default.")
        self.steps = [step for step in self.steps if self.step_enabled(step, stage)]
        print('[Info] ReconAgent initialized.')

    def step_enabled(self, step: str, stage: str = None) -> bool:
        """Whether a flagged step runs: a key asks for it, or --stage named it."""
        flag = FLAGGED_STEPS.get(step)
        if flag is None or step == stage or any(cfg[flag] for cfg in self.key_cfgs.values()):
            return True
        print(f"[Info] Skipping '{step}': '{flag}' is off in the configuration.")
        return False

    def save_scene_dicts(self):
        for key, scene_dict in self.key_scene_dicts.items():
            with open(self.base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
                pickle.dump(scene_dict, f)
        print('[Info] Scene dictionaries saved.')

    def save_assets(self):
        for key, scene_dict in self.key_scene_dicts.items():
            scene_json = scene_dict["info"]
            save_dir = self.base_dir / f'outputs/{key}/simulation'
            save_dir.mkdir(parents=True, exist_ok=True)

            background_image_path = self.base_dir / f'outputs/{key}/reconstruction/background.jpg'
            new_background_image_path = save_dir / 'background.jpg'
            scene_json["background_image"] = str(new_background_image_path)
            shutil.copy(background_image_path, new_background_image_path)

            bg_glb_path = scene_json["background"]["registered"]
            new_bg_glb_path = save_dir / Path(bg_glb_path).name
            scene_json["background"] = {"registered": str(new_bg_glb_path)}
            shutil.copy(bg_glb_path, new_bg_glb_path)

            scene_glb_path = scene_json["scene_mesh"]["optimized"]
            new_scene_glb_path = save_dir / Path(scene_glb_path).name
            scene_json["scene_mesh"] = {"optimized": str(new_scene_glb_path)}
            shutil.copy(scene_glb_path, new_scene_glb_path)

            for oid, obj in scene_json["objects"].items():
                obj_glb_path = obj["optimized"]
                new_obj_glb_path = save_dir / Path(obj_glb_path).name
                shutil.copy(obj_glb_path, new_obj_glb_path)
                obj_fdpose_trajs_path = obj["fdpose_trajs"]
                new_obj_fdpose_trajs_path = save_dir / Path(obj_fdpose_trajs_path).name
                shutil.copy(obj_fdpose_trajs_path, new_obj_fdpose_trajs_path)
                obj_simple_trajs_path = obj["simple_trajs"]
                new_obj_simple_trajs_path = save_dir / Path(obj_simple_trajs_path).name
                shutil.copy(obj_simple_trajs_path, new_obj_simple_trajs_path)
                obj_hybrid_trajs_path = obj["hybrid_trajs"]
                new_obj_hybrid_trajs_path = save_dir / Path(obj_hybrid_trajs_path).name
                shutil.copy(obj_hybrid_trajs_path, new_obj_hybrid_trajs_path)

                scene_json["objects"][oid] = {
                    "oid": obj["oid"],
                    "name": obj["name"],
                    "object_center": obj["object_center"],
                    "object_min": obj["object_min"],
                    "object_max": obj["object_max"],
                    "object_to_world": obj.get("object_to_world"),
                    "optimized": str(new_obj_glb_path),
                    "fdpose_trajs": str(new_obj_fdpose_trajs_path),
                    "simple_trajs": str(new_obj_simple_trajs_path),
                    "hybrid_trajs": str(new_obj_hybrid_trajs_path)
                }
            json_path = save_dir / "scene.json"
            with open(json_path, 'w') as f:
                json.dump(scene_json, f, indent=2)
        print('[Info] Scene JSON files saved.')

    def auto_segmentation(self):
        from openreal2sim.steps.reconstruction.steps.segmentation_auto import auto_segmentation
        self.key_scene_dicts = auto_segmentation(self.keys, self.key_scene_dicts)
        print("[Info] Automatic frame-0 segmentation completed.")

    def mask_propagation(self):
        from openreal2sim.steps.reconstruction.steps.segmentation_mask_propagation import mask_propagation
        self.key_scene_dicts = mask_propagation(self.keys, self.key_scene_dicts)
        print("[Info] Mask propagation completed.")

    def background_pixel_inpainting(self):
        from openreal2sim.steps.reconstruction.steps.background_pixel_inpainting import background_pixel_inpainting
        self.key_scene_dicts = background_pixel_inpainting(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Background inpainting completed.')

    def background_point_inpainting(self):
        from openreal2sim.steps.reconstruction.steps.background_point_inpainting import background_point_inpainting
        self.key_scene_dicts = background_point_inpainting(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Background point inpainting completed.')

    def background_mesh_generation(self):
        from openreal2sim.steps.reconstruction.steps.background_mesh_generation import background_mesh_generation
        self.key_scene_dicts = background_mesh_generation(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Background mesh generation completed.')

    def object_mesh_generation(self):
        from openreal2sim.steps.reconstruction.steps.sam_object_mesh_generation import sam_object_mesh_generation
        self.key_scene_dicts = sam_object_mesh_generation(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Object mesh generation completed.')

    def scenario_construction(self):
        from openreal2sim.steps.reconstruction.steps.scenario_construction import scenario_construction
        self.key_scene_dicts = scenario_construction(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Scenario construction completed.')

    def scenario_fdpose_optimization(self):
        from openreal2sim.steps.reconstruction.steps.scenario_fdpose_optimization import scenario_fdpose_optimization
        self.key_scene_dicts = scenario_fdpose_optimization(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Scenario foundation pose optimization completed.')

    def scenario_collision_optimization(self):
        from openreal2sim.steps.reconstruction.steps.scenario_collision_optimization import scenario_collision_optimization
        self.key_scene_dicts = scenario_collision_optimization(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Scenario collision optimization completed.')

    def run(self):
        for step in self.steps:
            getattr(self, step)()
        self.save_assets()
        print('[Info] ReconAgent run completed.')
        return self.key_scene_dicts


def free_gpu_memory():
    """Give the next key a clean allocator, now that the keys share one process."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_key(key: str, cfg: dict, stage: str = None) -> bool:
    """Reconstruct one key, with its own log files and metadata entry.

    The keys share this process, so one key must not take the others down:
    everything but KeyboardInterrupt is recorded as a failed key, including
    SystemExit raised by a step that calls sys.exit.
    """
    start_time = time.time()
    metadata_path = Path("outputs") / key / "metadata.yaml"
    print(f"Running {key} .........", end='', flush=True)
    try:
        with key_logs(key):
            try:
                notify_started(key)
                agent = ReconAgent([key], cfg, stage=stage)
                agent.run()
                notify_success(key)
            except KeyboardInterrupt:
                raise
            except BaseException:
                # Still inside key_logs, so the stderr log of the key keeps it.
                traceback.print_exc()
                raise
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        elapsed = int(time.time() - start_time)
        print(f" [{elapsed}s] - Failed!")
        print(f"[Error] {error}")
        print(f"[Error] Check the logs in outputs/{key}/logs")
        update_metadata(metadata_path, elapsed, "fail")
        notify_failed(key)
        return False
    finally:
        free_gpu_memory()
    elapsed = int(time.time() - start_time)
    print(f" [{elapsed}s] - Done!")
    update_metadata(metadata_path, elapsed, "success")
    return True


def main(config_file: str = "config/config.yaml", key: str = None, stage: str = None) -> bool:
    cfg = yaml.safe_load(Path(config_file).open("r"))
    keys = [key] if key is not None else cfg["keys"]

    print("[Info] Reconstruction starting...")
    print(f"[Info] Processing {len(keys)} keys: {keys}")

    results = {}
    for key in keys:
        print(f"\n{'_' * 60}")
        results[key] = run_key(key, cfg, stage)
        print(f"{'_' * 60}\n")

    if not all(results.values()):
        return False
    print("\n[Info] Reconstruction completed successfully!")
    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML with keys: [key1, key2, ...]")
    parser.add_argument("--key", type=str, default=None, help="Process a single key instead of all keys from config")
    parser.add_argument("--stage", type=str, default=None, help="Start from this step, one of %s" % ", ".join(OPTIONAL_STEPS + STEPS))
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    sys.exit(0 if main(args.config, args.key, args.stage) else 1)
