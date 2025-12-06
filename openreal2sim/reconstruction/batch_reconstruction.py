import subprocess
import sys
import time
import threading
import yaml
import argparse
from pathlib import Path
from datetime import datetime

def updateMetadata(metadata_path: Path, reconstruction_time: int, status: str = "success"):
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
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f)

def run_recon_agent_for_key(key, stage=None):
    """Run recon_agent.py for a specific key"""
    # Get the path to recon_agent.py (same directory as this script)
    script_dir = Path(__file__).parent
    recon_agent_path = script_dir / "recon_agent.py"

    # Create log directory for this key
    log_dir = Path("outputs") / key / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for log files (month-day-hour:minute)
    timestamp = datetime.now().strftime("%m-%d-%H:%M")

    # Define log file paths
    stdout_log = log_dir / f"{timestamp}_stdout.log"
    stderr_log = log_dir / f"{timestamp}_stderr.log"

    # Build the command
    cmd = [sys.executable, str(recon_agent_path), "--key", key, "--label", key]
    if stage:
        cmd.extend(["--stage", stage])

    # Flag to control the timer thread
    stop_timer = threading.Event()
    start_time = time.time()

    def print_timer():
        """Print elapsed time while the process runs"""
        while not stop_timer.is_set():
            elapsed = int(time.time() - start_time)
            print(f"\rRunning {key} ......... [{elapsed}s]", end='', flush=True)
            time.sleep(1)

    # Start the timer thread
    timer_thread = threading.Thread(target=print_timer, daemon=True)
    timer_thread.start()
    metadata_path = Path("outputs") / key / "metadata.yaml"

    try:
        # Open log files and run the command
        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            result = subprocess.run(cmd, check=True, stdout=stdout_file, stderr=stderr_file)
        stop_timer.set()
        timer_thread.join(timeout=1)
        elapsed = int(time.time() - start_time)
        print(f"\rRunning {key} ......... [{elapsed}s] - Done!")
        # Write metadata with reconstruction info
        updateMetadata(metadata_path, elapsed, "success")
        return True
    except subprocess.CalledProcessError as e:
        stop_timer.set()
        timer_thread.join(timeout=1)
        elapsed = int(time.time() - start_time)
        print(f"\rRunning {key} ......... [{elapsed}s] - Failed!")
        print(f"[Error] Exit code: {e.returncode}")
        print(f"[Error] Check logs at: {stdout_log} and {stderr_log}")
        updateMetadata(metadata_path, elapsed, "fail")
        return False

def main(config_file: str = "config/config.yaml", stage: str = None):
    """Main function: load config and process keys."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    keys = config["keys"]

    print(f"[Info] SuperAgent starting...")
    print(f"[Info] Processing {len(keys)} keys: {keys}")

    results = {}
    for key in keys:
        print(f"\n{'_'*60}")
        success = run_recon_agent_for_key(key, stage)
        results[key] = success
        print(f"{'_'*60}\n")

    # Exit with error if any key failed
    if not all(results.values()):
        sys.exit(1)

    print("\n[Info] SuperAgent completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="YAML with keys: [key1, key2, ...]")
    parser.add_argument("--stage", type=str, default=None, help="Optional stage to start from")
    args = parser.parse_args()

    main(args.config, args.stage)
