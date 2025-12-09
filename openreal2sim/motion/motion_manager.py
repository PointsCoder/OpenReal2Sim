from pathlib import Path
import sys
import pickle
import yaml
import argparse
import json
sys.path.append(str(Path.cwd() / "openreal2sim" / "motion" / "modules"))
sys.path.append(str(Path.cwd() / "openreal2sim" / "motion" / "utils"))

from utils.compose_config import compose_configs

class MotionAgent:
    def __init__(self, stage=None, key=None):
        print('[Info] Initializing MotionAgent...')
        self.base_dir = Path.cwd()
        cfg_path = self.base_dir / "config" / "config.yaml"
        cfg = yaml.safe_load(cfg_path.open("r"))
        self.keys = [key] if key is not None else cfg["keys"]
        self.key_cfgs = {key: compose_configs(key, cfg) for key in self.keys}
        self.key_scene_dicts = {}
        for key in self.keys:
            scene_pkl = self.base_dir / f'outputs/{key}/scene/scene.pkl'
            with open(scene_pkl, 'rb') as f:
                scene_dict = pickle.load(f)
            self.key_scene_dicts[key] = scene_dict
        self.key_scene_json_dicts = {}
        for key in self.keys:
            scene_json_path = self.base_dir / f'outputs/{key}/simulation/scene.json'
            with open(scene_json_path, "r") as f:
                scene_json_dict = json.load(f)
            self.key_scene_json_dicts[key] = scene_json_dict
        self.stages = [
            "hand_extraction",
            "demo_motion_process",
            "grasp_point_extraction",
            "grasp_generation",
        ]
        if stage is not None:
            if stage in self.stages:
                start_idx = self.stages.index(stage)
                self.stages = self.stages[start_idx:]
            else:
                print(f"[Warning] Stage '{stage}' not found. It must be one of {self.stages}. Running all stages by default.")
        print('[Info] MotionAgent initialized.')
    
    def save_scene_dicts(self):
        for key, scene_dict in self.key_scene_dicts.items():
            with open(self.base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
                pickle.dump(scene_dict, f)
        print('[Info] Scene dictionaries saved.')

    def hand_extraction(self):
        from modules.hand_extraction import hand_extraction
        self.key_scene_dicts = hand_extraction(self.keys, self.key_scene_dicts, self.key_cfgs)
        print("[Info] Hand extraction completed.")

    def demo_motion_process(self):
        from modules.demo_motion_process import demo_motion_process
        self.key_scene_dicts = demo_motion_process(self.keys, self.key_scene_dicts, self.key_cfgs)
        print("[Info] Demo motion process completed.")

    def grasp_point_extraction(self):
        from modules.grasp_point_extraction import grasp_point_extraction
        self.key_scene_dicts = grasp_point_extraction(self.keys, self.key_scene_dicts, self.key_scene_json_dicts, self.key_cfgs)
        print("[Info] Grasp point extraction completed.")

    def grasp_generation(self):
        from modules.grasp_generation import grasp_generation
        grasp_generation(self.keys)
        print("[Info] Grasp generation completed.")

    def run(self):
        if "hand_extraction" in self.stages:
            self.hand_extraction()
        if "demo_motion_process" in self.stages:
            self.demo_motion_process()
        if "grasp_point_extraction" in self.stages:
            self.grasp_point_extraction()
        if "grasp_generation" in self.stages:
            self.grasp_generation()
        self.save_scene_dicts()
        print('[Info] MotionAgent run completed.')
        return self.key_scene_dicts

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--stage', type=str, default=None, help='Starting from a certain stage')
    args.add_argument('--key', type=str, default=None, help='Process a single key instead of all keys from config')
    args = args.parse_args()

    agent = MotionAgent(stage=args.stage, key=args.key)
    scene_dicts = agent.run()
