#!/usr/bin/env python3
"""Run the simulation steps for every key in the config, or for a single key.

The IsaacLab backend builds its ``AppLauncher`` at import time, and a process can
only start IsaacSim once, so every step runs in its own subprocess, exactly as
the old ``isaaclab/sim_agent.sh`` did.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

# The runner can be started as a script, so put the repository root on sys.path.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEPS = [
    "usd_conversion",
    "sim_heuristic_manip",
    "sim_randomize_rollout",
]

BACKEND_DIR = Path(__file__).resolve().parent / "isaaclab"

SCRIPTS = {
    "usd_conversion": BACKEND_DIR / "sim_utils" / "usd_conversion.py",
    "sim_heuristic_manip": BACKEND_DIR / "sim_heuristic_manip.py",
    "sim_randomize_rollout": BACKEND_DIR / "sim_randomize_rollout.py",
}


class SimulationAgent:
    def __init__(self, config_file: str = "config/config.yaml", key: str = None, stage: str = None):
        self.config_file = config_file
        self.base_dir = Path.cwd()

        config_path = self.base_dir / config_file
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg = yaml.safe_load(config_path.open("r"))
        self.keys = [key] if key is not None else cfg["keys"]

        self.steps = list(STEPS)
        if stage is not None:
            if stage in self.steps:
                self.steps = self.steps[self.steps.index(stage):]
            else:
                print(f"[Warning] Step '{stage}' not found. It must be one of {self.steps}. Running all steps by default.")

    def run_step(self, step: str, key: str) -> bool:
        """Run one backend script for one key in a fresh process."""
        script = SCRIPTS[step]
        command = [sys.executable, str(script), "--key", key]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        # Without this the step output sits in the child's pipe buffer, so a long
        # run looks hung and the log only arrives when the step ends.
        env["PYTHONUNBUFFERED"] = "1"
        print("=" * 30)
        print(f"[Run] Step: {step} | Key: {key}")
        print("=" * 30, flush=True)
        result = subprocess.run(command, cwd=self.base_dir, env=env)
        if result.returncode != 0:
            print(f"[Error] Step {step} failed for key {key} with code {result.returncode}", file=sys.stderr)
            return False
        return True

    def run(self) -> bool:
        print(f"Starting Simulation manager with config file {self.config_file}")
        print(f"[Info] Keys: {self.keys}")
        print(f"[Info] Steps: {self.steps}")
        results = {}
        for key in self.keys:
            results[key] = all(self.run_step(step, key) for step in self.steps)
        for key, ok in results.items():
            print(f"[Info] {key}: {'success' if ok else 'fail'}")
        return all(results.values())


def main(config_file: str = "config/config.yaml", key: str = None, stage: str = None) -> bool:
    agent = SimulationAgent(config_file=config_file, key=key, stage=stage)
    return agent.run()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML with keys: [key1, key2, ...]")
    parser.add_argument("--key", type=str, default=None, help="Process a single key instead of all keys from config")
    parser.add_argument("--stage", type=str, default=None, help="Start from this step, one of %s" % ", ".join(STEPS))
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    sys.exit(0 if main(args.config, args.key, args.stage) else 1)
