from pathlib import Path
import sys
import pickle
import yaml
import json
import argparse

from modules.utils.compose_config import compose_configs
from utils.notification import notify_started, notify_failed, notify_success

class ReconAgent:
    def __init__(self, stage=None):
        print('[Info] Initializing ReconAgent...')
        self.base_dir = Path.cwd()
        cfg_path = self.base_dir / "config" / "config.yaml"
        cfg = yaml.safe_load(cfg_path.open("r"))
        self.keys = cfg["keys"]
        self.key_cfgs = {key: compose_configs(key, cfg) for key in self.keys}
        self.key_scene_dicts = {}
        for key in self.keys:
            scene_pkl = self.base_dir / f'outputs/{key}/scene/scene.pkl'
            with open(scene_pkl, 'rb') as f:
                scene_dict = pickle.load(f)
            self.key_scene_dicts[key] = scene_dict
        self.stages = [
            "background_pixel_inpainting",
            "background_point_inpainting",
            "background_mesh_generation",
            "object_mesh_generation",
            "scenario_construction",
            "scenario_fdpose_optimization",
            "scenario_collision_optimization"
        ]
        if stage is not None:
            if stage in self.stages:
                start_idx = self.stages.index(stage)
                self.stages = self.stages[start_idx:]
            else:
                print(f"[Warning] Stage '{stage}' not found. It must be one of the {self.stages}. Running all stages by default.")
        print('[Info] ReconAgent initialized.')
    
    def save_scene_dicts(self):
        for key, scene_dict in self.key_scene_dicts.items():
            with open(self.base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
                pickle.dump(scene_dict, f)
        print('[Info] Scene dictionaries saved.')

    def save_scene_jsons(self):
        for key, scene_dict in self.key_scene_dicts.items():
            scene_json = scene_dict["info"]
            json_path = self.base_dir / f'outputs/{key}/scene/scene.json'
            with open(json_path, 'w') as f:
                json.dump(scene_json, f, indent=2)
        print('[Info] Scene JSON files saved.')

    def background_pixel_inpainting(self):
        from modules.background_pixel_inpainting import background_pixel_inpainting
        self.key_scene_dicts = background_pixel_inpainting(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Background inpainting completed.')

    def background_point_inpainting(self):
        from modules.background_point_inpainting import background_point_inpainting
        self.key_scene_dicts = background_point_inpainting(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Background point inpainting completed.')

    def background_mesh_generation(self):
        from modules.background_mesh_generation import background_mesh_generation
        self.key_scene_dicts = background_mesh_generation(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Background mesh generation completed.')

    def object_mesh_generation(self):
        from modules.object_mesh_generation import object_mesh_generation
        self.key_scene_dicts = object_mesh_generation(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Object mesh generation completed.')

    def scenario_construction(self):
        from modules.scenario_construction import scenario_construction
        self.key_scene_dicts = scenario_construction(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Scenario construction completed.')

    def scenario_fdpose_optimization(self):
        from modules.scenario_fdpose_optimization import scenario_fdpose_optimization
        self.key_scene_dicts = scenario_fdpose_optimization(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Scenario foundation pose optimization completed.')

    def scenario_collision_optimization(self):
        from modules.scenario_collision_optimization import scenario_collision_optimization
        self.key_scene_dicts = scenario_collision_optimization(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Scenario collision optimization completed.')

    def run(self):
        if "background_pixel_inpainting" in self.stages:
            self.background_pixel_inpainting()
        if "background_point_inpainting" in self.stages:
            self.background_point_inpainting()
        if "background_mesh_generation" in self.stages:
            self.background_mesh_generation()
        if "object_mesh_generation" in self.stages:
            self.object_mesh_generation()
        if "scenario_construction" in self.stages:
            self.scenario_construction()
        if "scenario_fdpose_optimization" in self.stages:
            self.scenario_fdpose_optimization()
        if "scenario_collision_optimization" in self.stages:
            self.scenario_collision_optimization()
        self.save_scene_jsons()
        print('[Info] ReconAgent run completed.')
        return self.key_scene_dicts

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--stage', type=str, default=None, help='Starting from a certain stage')
    args.add_argument('--label', type=str, default=None, help='Optional label for notifications')
    args = args.parse_args()

    if args.label:
        notify_started(args.label)

    try:
        agent = ReconAgent(args.stage)
        scene_dicts = agent.run()

        if args.label:
            notify_success(args.label)
    except Exception as e:
        if args.label:
            notify_failed(args.label)
        raise