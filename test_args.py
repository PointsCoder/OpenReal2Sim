from dataclasses import dataclass
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("sim_policy")
parser.add_argument("--key", type=str, default="demo_video", help="scene key")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--device", "cuda:6"])
print(args_cli.device)
