#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
from pathlib import Path


class PreprocessManager:
    def __init__(self, config_file: str = "config/config.yaml", key_name: str = None):
        self.config_file = config_file
        self.base_dir = Path.cwd()
        self.key_name = key_name

        # Verify config file exists
        config_path = self.base_dir / config_file
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    def image_extraction(self):
        from openreal2sim.preprocess.image_extraction import main as image_extraction_main
        image_extraction_main(self.config_file, self.key_name)

    def depth_prediction(self):
        from openreal2sim.preprocess.depth_prediction import main as depth_prediction_main
        depth_prediction_main(self.config_file, self.key_name)

    def depth_calibration(self):
        from openreal2sim.preprocess.depth_calibration import main as depth_calibration_main
        depth_calibration_main(self.config_file, self.key_name)

    def run(self):
        print(f"Starting Preprocess manager with config file {self.config_file}")
        try:
            self.image_extraction()
            self.depth_prediction()
            self.depth_calibration()
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False


def main(config_file: str = "config/config.yaml", key_name: str = None):
    manager = PreprocessManager(config_file=config_file, key_name=key_name)
    success = manager.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file with keys: [lab1, ...]"
    )
    parser.add_argument("--key_name", type=str, default=None, help="If set, run only this key")
    args = parser.parse_args()

    main(args.config, args.key_name)
