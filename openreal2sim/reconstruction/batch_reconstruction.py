import subprocess
import sys
import time
import threading
import yaml
import argparse
from pathlib import Path
from datetime import datetime

def updateMetadata(metadata_path: Path, reconstruction_time: int, status: str = "success", use_sam: bool = False):
    """Create or update metadata YAML for a reconstruction run."""
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                existing = yaml.safe_load(f) or {}
                metadata.update(existing)
        except Exception:
            pass
    metadata["reconstruction_time"] = reconstruction_time
    metadata["reconstruction_status"] = status
    metadata["use_sam"] = use_sam
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f)

def run_recon_agent_for_key(key, stage=None, use_sam=False):
    """Run recon_agent.py for a specific key"""
    script_dir = Path(__file__).parent
    recon_agent_path = script_dir / "recon_agent.py"

    
    timestamp = datetime.now().strftime("%m-%d-%H:%M")
    log_dir = Path("outputs") / key / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{timestamp}_stdout.log"
    stderr_log = log_dir / f"{timestamp}_stderr.log"

    cmd = [sys.executable, str(recon_agent_path), "--key", key, "--label", key]
    if stage:
        cmd.extend(["--stage", stage])
    if use_sam:
        cmd.append("--use-sam")

    stop_timer = threading.Event()
    start_time = time.time()

    def print_timer():
        """Print elapsed time while the process runs"""
        while not stop_timer.is_set():
            elapsed = int(time.time() - start_time)
            print(f"\rRunning {key} ......... [{elapsed}s]", end='', flush=True)
            time.sleep(1)

    timer_thread = threading.Thread(target=print_timer, daemon=True)
    timer_thread.start()
    metadata_path = Path("outputs") / key / "metadata.yaml"

    try:
        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            result = subprocess.run(cmd, check=True, stdout=stdout_file, stderr=stderr_file)
        stop_timer.set()
        timer_thread.join(timeout=1)
        elapsed = int(time.time() - start_time)
        print(f"\rRunning {key} ......... [{elapsed}s] - Done!")
        updateMetadata(metadata_path, elapsed, "success", use_sam)
        return True
    except subprocess.CalledProcessError as e:
        stop_timer.set()
        timer_thread.join(timeout=1)
        elapsed = int(time.time() - start_time)
        print(f"\rRunning {key} ......... [{elapsed}s] - Failed!")
        print(f"[Error] Exit code: {e.returncode}")
        print(f"[Error] Check logs at: {stdout_log} and {stderr_log}")
        updateMetadata(metadata_path, elapsed, "fail", use_sam)
        return False

def main(config_file: str = "config/config.yaml", stage: str = None, use_sam: bool = False):
    """Main function: load config and process keys."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    keys = config["keys"]

    print(f"[Info] SuperAgent starting...")
    print(f"[Info] Processing {len(keys)} keys: {keys}")

    results = {}
    for key in keys:
        print(f"\n{'_'*60}")
        success = run_recon_agent_for_key(key, stage, use_sam)
        results[key] = success
        print(f"{'_'*60}\n")

    if not all(results.values()):
        sys.exit(1)

    print("\n[Info] SuperAgent completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML with keys: [key1, key2, ...]")
    parser.add_argument("--stage", type=str, default=None, help="Optional stage to start from")
    parser.add_argument("--use-sam", action="store_true", help="Use SAM-3D for object mesh generation")
    args = parser.parse_args()

    main(args.config, args.stage, args.use_sam)
