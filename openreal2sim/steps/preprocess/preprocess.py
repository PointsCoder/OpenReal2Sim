#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the preprocess steps for every key, or for a single key."""

import argparse
import sys
from pathlib import Path

# The runner can be started as a script, so put the repository root on sys.path.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEPS = [
    "image_extraction",
    "depth_prediction",
    "depth_calibration",
]


class PreprocessManager:
    def __init__(self, config_file: str = "config/config.yaml", key: str = None, stage: str = None):
        self.config_file = config_file
        self.base_dir = Path.cwd()
        self.key = key

        config_path = self.base_dir / config_file
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.steps = list(STEPS)
        if stage is not None:
            if stage in self.steps:
                self.steps = self.steps[self.steps.index(stage):]
            else:
                print(f"[Warning] Step '{stage}' not found. It must be one of {self.steps}. Running all steps by default.")

    def image_extraction(self):
        from openreal2sim.steps.preprocess.steps.image_extraction import main as image_extraction_main
        image_extraction_main(self.config_file, self.key)

    def depth_prediction(self):
        from openreal2sim.steps.preprocess.steps.depth_prediction import main as depth_prediction_main
        depth_prediction_main(self.config_file, self.key)

    def depth_calibration(self):
        from openreal2sim.steps.preprocess.steps.depth_calibration import main as depth_calibration_main
        depth_calibration_main(self.config_file, self.key)

    def run(self):
        print(f"Starting Preprocess manager with config file {self.config_file}")
        try:
            for step in self.steps:
                getattr(self, step)()
            return True

        except Exception:
            import traceback
            traceback.print_exc()
            return False


def main(config_file: str = "config/config.yaml", key: str = None, stage: str = None):
    manager = PreprocessManager(config_file=config_file, key=key, stage=stage)
    return manager.run()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file with keys: [lab1, ...]"
    )
    parser.add_argument("--key", type=str, default=None, help="If set, run only this key")
    parser.add_argument("--stage", type=str, default=None, help="Start from this step, one of %s" % ", ".join(STEPS))
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    sys.exit(0 if main(args.config, args.key, args.stage) else 1)
