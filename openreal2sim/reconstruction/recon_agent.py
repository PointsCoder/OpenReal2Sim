import os
import glob
import torch
from pathlib import Path
import sys
from PIL import Image
import numpy as np
import pickle
import cv2
import yaml
import json

from modules.utils.compose_config import compose_configs

class ReconAgent:
    def __init__(self):
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
        print('[Info] ReconAgent initialized.')
    
    def save_scene_dicts(self):
        for key, scene_dict in self.key_scene_dicts.items():
            with open(self.base_dir / f'outputs/{key}/scene/scene.pkl', 'wb') as f:
                pickle.dump(scene_dict, f)
        print('[Info] Scene dictionaries saved.')

    def save_scene_jsons(self):
        for key, scene_dict in self.key_scene_dicts.items():
            try:
                scene_json = json.load(open(self.base_dir / f'outputs/{key}/scene/scene.json', "r"))
            except FileNotFoundError:
                scene_json = {}
            scene_json = self.update_camera_json(scene_json, scene_dict)
            json_path = self.base_dir / f'outputs/{key}/scene/scene.json'
            with open(json_path, 'w') as f:
                json.dump(scene_json, f, indent=2)
        print('[Info] Scene JSON files saved.')

    def update_camera_json(self, scene_json, scene_dict):
        if "depths" not in scene_dict or "intrinsics" not in scene_dict:
            return scene_json
        H = scene_dict["depths"][0].shape[0]
        W = scene_dict["depths"][0].shape[1]
        K = scene_dict["intrinsics"]
        scene_json["camera"] = {
            "width": W, "height": H,
            "fx": float(K[0,0]),
            "fy": float(K[1,1]),
            "cx": float(K[0,2]),
            "cy": float(K[1,2])
        }
        return scene_json

    def background_pixel_inpainting(self):
        from openreal2sim.reconstruction.modules.background_pixel_inpainting import background_pixel_inpainting
        self.key_scene_dicts = background_pixel_inpainting(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Background inpainting completed.')

    def background_point_inpainting(self):
        from openreal2sim.reconstruction.modules.background_point_inpainting import background_point_inpainting
        self.key_scene_dicts = background_point_inpainting(self.keys, self.key_scene_dicts, self.key_cfgs)
        print('[Info] Background point inpainting completed.')

    def run(self):
        self.background_pixel_inpainting()
        self.background_point_inpainting()
        print('[Info] ReconAgent run completed.')
        return self.key_scene_dicts

if __name__ == '__main__':
    agent = ReconAgent()
    scene_dicts = agent.run()