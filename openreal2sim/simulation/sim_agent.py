### isaaclab

from pathlib import Path
import sys
import pickle
import yaml
import json
import argparse
import shutil
import numpy as np
from isaaclab.app import AppLauncher
import typing
from typing import Optional
sys.path.append(str(Path.cwd() / "IsaacLab"))
sys.path.append(str(Path.cwd() / "openreal2sim" / "simulation"))
from utils.compose_config import compose_configs
from utils.notification import notify_started, notify_failed, notify_success

class ProprocessAgent:
    def __init__(self, stage=None, key=None):
        print('[Info] Initializing ProprocessAgent...')
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
        self.stages = [
            # "hand_extraction",
            #"demo_motion_process",
            # "grasp_point_generation",
            # "grasp_generation"

        ]
        if stage is not None:
            if stage in self.stages:
                start_idx = self.stages.index(stage)
                self.stages = self.stages[start_idx:]
            else:
                print(f"[Warning] Stage '{stage}' not found. It must be one of the {self.stages}. Running all stages by default.")
        print('[Info] ProprocessAgent initialized.')
    
 
    
    def hand_extraction(self):
        from modules.hand_preprocess.hand_extraction import hand_extraction
        self.key_scene_dicts = hand_extraction(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Hand extraction completed.')
    
    def demo_motion_process(self):
        from modules.demo_motion_process import demo_motion_process
        self.key_scene_dicts = demo_motion_process(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Demo motion process completed.')
    
    # def grasp_point_generation(self):
    #     from modules.grasp_point_generation import grasp_point_generation
    #     self.key_scene_dicts = grasp_point_generation(self.keys, self.key_scene_dicts, self.key_cfgs)
    #     print('[Info] Grasp point generation completed.')
    
    def grasp_generation(self):
        from modules.grasp_preprocess.grasp_generation import grasp_generation
        grasp_generation(self.keys)
        print('[Info] Grasp proposal generation completed.')
    
    def run(self):
        if "hand_extraction" in self.stages:
            self.hand_extraction()
        if "demo_motion_process" in self.stages:
            self.demo_motion_process()
        # if "grasp_point_generation" in self.stages:
        #     self.grasp_point_generation()
        if "grasp_generation" in self.stages:
            self.grasp_generation()
       
        print('[Info] ProprocessAgent run completed.')
        return self.key_scene_dicts
    

        


def create_args_from_config(config_path: Optional[str] = None, config_dict: Optional[dict] = None) -> argparse.Namespace:
    """
    Create an argparse.Namespace from a config file (yaml/json) or dictionary.
    
    Args:
        config_path: Path to yaml or json config file
        config_dict: Dictionary with config parameters
        
    Returns:
        argparse.Namespace with the config parameters
        
    Example config file (yaml):
        key: "demo_video"
        robot: "franka"
        num_envs: 4
        num_trials: 10
        teleop_device: "keyboard"
        sensitivity: 1.0
        device: "cuda:0"
        enable_cameras: true
        headless: true
    """

    if config_dict is None:
        config_dict = {
            'robot': 'franka',
            'num_envs': 5,
            'num_trials': 10,
            'teleop_device': 'keyboard',
            'sensitivity': 1.0,
            'device': 'cuda:0',
            'enable_cameras': True,
            'headless': True,
        }
    else:
        config_dict = {
            'robot': config_dict['robot'],
            'num_envs': config_dict['num_envs'],
            'num_trials': config_dict['num_trials'],
            'teleop_device': config_dict['teleop_device'],
            'sensitivity': config_dict['sensitivity'],
            'device': config_dict['device'],
            'enable_cameras': config_dict['enable_cameras'],
            'headless': config_dict['headless'],
        }
    
    return argparse.Namespace(**config_dict)


class IsaacAgent:
    def __init__(self, stage=None, key=None):
        print('[Info] Initializing IsaacAgent...')
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
        self.stages = [
            # "usd_conversion",
            #"grasp_generation",
            #"sim_heuristic_manip",
            "sim_randomize_rollout",

        ]
        if stage is not None:
            if stage in self.stages:
                start_idx = self.stages.index(stage)
                self.stages = self.stages[start_idx:]
            else:
                print(f"[Warning] Stage '{stage}' not found. It must be one of the {self.stages}. Running all stages by default.")
        print('[Info] IsaacAgent initialized.')
    
    def launch_isaaclab(self):

        self.args_cli = create_args_from_config()
        #AppLauncher.add_app_launcher_args(self.args_cli)
        self.args_cli.enable_cameras = True
        self.args_cli.headless = True
        self.app_launcher = AppLauncher(vars(self.args_cli))
        self.simulation_app = self.app_launcher.app
        return self.simulation_app
    
    def close_isaaclab(self):
        self.simulation_app.close()

    def usd_conversion(self):
        from isaaclab.sim_preprocess.usd_conversion import usd_conversion
        usd_conversion(self.keys)
        print('[Info] USD conversion completed.')
    
    def sim_heuristic_manip(self):
        self.launch_isaaclab()
        from isaaclab.sim_heuristic_manip import sim_heuristic_manip
        sim_heuristic_manip(self.keys, args_cli=self.args_cli)
        print('[Info] Heuristic manipulation simulation completed.')
        self.close_isaaclab()
    

    def sim_randomize_rollout(self):
        self.launch_isaaclab()
        from isaaclab.sim_randomize_rollout import sim_randomize_rollout
        sim_randomize_rollout(self.keys, args_cli=self.args_cli)
        print('[Info] Randomize rollout completed.')
        self.close_isaaclab()

    def grasp_generation(self):
        from modules.grasp_preprocess.grasp_generation import grasp_generation
        grasp_generation(self.keys)
        print('[Info] Grasp proposal generation completed.')
    
    def run(self):
        if "usd_conversion" in self.stages:
            self.usd_conversion()
        if "grasp_generation" in self.stages:
            self.grasp_generation()
        if "launch_isaaclab" in self.stages:
            self.launch_isaaclab()
        if "sim_heuristic_manip" in self.stages:
            self.sim_heuristic_manip()

        if "sim_randomize_rollout" in self.stages:
            self.sim_randomize_rollout()
        if "close_isaaclab" in self.stages:
            self.close_isaaclab()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--stage', type=str, default=None, help='Starting from a certain stage')
    args.add_argument('--key', type=str, default=None, help='Process a single key instead of all keys from config')
    args.add_argument('--label', type=str, default=None, help='Optional label for notifications')
    args = args.parse_args()

    if args.label:
        notify_started(args.label)

    try:
        agent = ProprocessAgent(stage=args.stage, key=args.key)
        scene_dicts = agent.run()

        if args.label:
            notify_success(args.label)
    except Exception as e:
        if args.label:
            notify_failed(args.label)
        raise

    try:
        agent = IsaacAgent(stage=args.stage, key=args.key)
        scene_dicts = agent.run()

        if args.label:
            notify_success(args.label)
    except Exception as e:
        if args.label:
            notify_failed(args.label)
        raise