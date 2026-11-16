#!/usr/bin/env python3
"""Run a stage of the pipeline, or every stage in order.

Usage:
    python openreal2sim/openreal2sim.py <stage> [--config C] [--key K] [--stage S]

Stages: preprocess, reconstruction, motion, all. The simulation backends are
chosen separately and are not part of "all"; see openreal2sim/steps/simulation.
"""

import argparse
import sys
from pathlib import Path

# The runner lives inside the package, so the directory python puts on sys.path
# for a script would shadow the openreal2sim package. Use the repository root.
HERE = str(Path(__file__).resolve().parent)
ROOT = str(Path(__file__).resolve().parents[1])
sys.path = [entry for entry in sys.path if entry != HERE]
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STAGES = ["preprocess", "reconstruction", "motion"]


def run_stage(stage: str, config: str, key: str, step: str) -> bool:
    if stage == "preprocess":
        from openreal2sim.steps.preprocess.preprocess import main as stage_main
    elif stage == "reconstruction":
        from openreal2sim.steps.reconstruction.reconstruction import main as stage_main
    elif stage == "motion":
        from openreal2sim.steps.motion.motion import main as stage_main
    else:
        raise ValueError("unknown stage: %s" % stage)
    print(f"[Info] Running stage {stage}")
    return stage_main(config, key, step)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline_stage", choices=STAGES + ["all"], help="stage to run, or all for %s in order" % ", ".join(STAGES))
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML with keys: [key1, key2, ...]")
    parser.add_argument("--key", type=str, default=None, help="Process a single key instead of all keys from config")
    parser.add_argument("--stage", type=str, default=None, help="Start the stage from this step")
    args = parser.parse_args(argv)

    stages = STAGES if args.pipeline_stage == "all" else [args.pipeline_stage]
    if args.stage is not None and len(stages) > 1:
        parser.error("--stage needs a single stage, not all")

    for stage in stages:
        if not run_stage(stage, args.config, args.key, args.stage):
            print(f"[Error] Stage {stage} failed", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
