#!/usr/bin/env python3
"""Run the motion steps for every key in the config, or for a single key."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import yaml

# The runner can be started as a script, so put the repository root on sys.path.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openreal2sim.utils.compose_config import compose_configs

STEPS = [
    "hand_extraction",
    "demo_motion_process",
    "grasp_point_extraction",
    "grasp_generation",
]


class MotionAgent:
    def __init__(self, config_file: str = "config/config.yaml", key=None, stage=None):
        print('[Info] Initializing MotionAgent...')
        self.base_dir = Path.cwd()
        cfg_path = self.base_dir / config_file
        cfg = yaml.safe_load(cfg_path.open("r"))
        self.keys = [key] if key is not None else cfg["keys"]
        self.key_cfgs = {key: compose_configs(key, cfg, "motion") for key in self.keys}
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
        self.steps = list(STEPS)
        if stage is not None:
            if stage in self.steps:
                self.steps = self.steps[self.steps.index(stage):]
            else:
                print(f"[Warning] Step '{stage}' not found. It must be one of {self.steps}. Running all steps by default.")
        print('[Info] MotionAgent initialized.')

    def save_scene_dicts(self):
        for key, scene_dict in self.key_scene_dicts.items():
            with open(self.base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
                pickle.dump(scene_dict, f)
        print('[Info] Scene dictionaries saved.')

    def hand_extraction(self):
        from openreal2sim.steps.motion.steps.hand_extraction import hand_extraction
        self.key_scene_dicts = hand_extraction(self.keys, self.key_scene_dicts, self.key_cfgs)
        print("[Info] Hand extraction completed.")

    def demo_motion_process(self):
        from openreal2sim.steps.motion.steps.demo_motion_process import demo_motion_process
        self.key_scene_dicts = demo_motion_process(self.keys, self.key_scene_dicts, self.key_cfgs)
        print("[Info] Demo motion process completed.")

    def grasp_point_extraction(self):
        from openreal2sim.steps.motion.steps.grasp_point_extraction import grasp_point_extraction
        self.key_scene_dicts = grasp_point_extraction(self.keys, self.key_scene_dicts, self.key_scene_json_dicts, self.key_cfgs)
        print("[Info] Grasp point extraction completed.")

    def grasp_generation(self):
        from openreal2sim.steps.motion.steps.grasp_generation import grasp_generation
        grasp_generation(self.keys)
        print("[Info] Grasp generation completed.")

    def run(self):
        for step in self.steps:
            getattr(self, step)()
        self.save_scene_dicts()
        print('[Info] MotionAgent run completed.')
        return self.key_scene_dicts


def main(config_file: str = "config/config.yaml", key: str = None, stage: str = None):
    agent = MotionAgent(config_file=config_file, key=key, stage=stage)
    agent.run()
    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML with keys: [key1, key2, ...]")
    parser.add_argument('--key', type=str, default=None, help='Process a single key instead of all keys from config')
    parser.add_argument('--stage', type=str, default=None, help="Start from this step, one of %s" % ", ".join(STEPS))
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    sys.exit(0 if main(args.config, args.key, args.stage) else 1)
