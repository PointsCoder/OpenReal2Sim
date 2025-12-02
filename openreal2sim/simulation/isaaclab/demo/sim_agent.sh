#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd /app && pwd)"

# Default parameters
gpu_id="1"
HEADLESS="--headless"
CONFIG_PATH="${ROOT_DIR}/config/config.yaml"

usage() {
  cat <<'EOF'
Usage: sim_agent.sh [options]

Options:
  --config <path>        Path to config.yaml file (default: /app/config/config.yaml)
  --no-headless          Disable headless mode (passes through to AppLauncher)
  -h, --help             Show this message and exit

The script loads keys from config.yaml and runs for each key:
  1. USD conversion (`isaaclab/sim_preprocess/usd_conversion.py`)
  2. Heuristic manipulation (`isaaclab/sim_heuristic_manip.py`)
  3. Randomized rollout (`isaaclab/sim_randomize_rollout.py`)
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:?Missing value for --config}"
      shift 2
      ;;
    --no-headless)
      HEADLESS=""
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERR] Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Load keys from config.yaml
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERR] Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# Extract keys from YAML using Python (more reliable than parsing YAML in bash)
KEYS=($(python3 -c "
import yaml
import sys
try:
    with open('${CONFIG_PATH}', 'r') as f:
        cfg = yaml.safe_load(f)
        keys = cfg.get('keys', [])
        if not keys:
            print('[ERR] No keys found in config.yaml', file=sys.stderr)
            sys.exit(1)
        print(' '.join(keys))
except Exception as e:
    print(f'[ERR] Failed to load config: {e}', file=sys.stderr)
    sys.exit(1)
"))

if [[ ${#KEYS[@]} -eq 0 ]]; then
  echo "[ERR] No keys found in config file: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[INFO] Loaded ${#KEYS[@]} key(s) from ${CONFIG_PATH}: ${KEYS[*]}"

# Determine which stages to run
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
cd "${ROOT_DIR}"

declare -a PIPELINE=(  "sim_randomize_rollout")

HEADLESS_ARGS=()
if [[ -n "${HEADLESS}" ]]; then
  HEADLESS_ARGS+=("${HEADLESS}")
fi

run_stage() {
  local stage="$1"
  local key="$2"
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  echo "=============================="
  echo "[RUN] Stage: ${stage} | Key: ${key}"
  echo "=============================="

  case "${stage}" in
    usd_conversion)
      python openreal2sim/simulation/isaaclab/sim_preprocess/usd_conversion.py "${HEADLESS_ARGS[@]}"
      ;;
    sim_heuristic_manip)
      echo "Setting CUDA_VISIBLE_DEVICES to ${CUDA_VISIBLE_DEVICES}"
      python openreal2sim/simulation/isaaclab/demo/sim_heuristic_manip.py \
        --key "${key}" \
        "${HEADLESS_ARGS[@]}"
      ;;
    sim_randomize_rollout)
      python openreal2sim/simulation/isaaclab/demo/sim_randomize_rollout.py \
        --key "${key}" \
        "${HEADLESS_ARGS[@]}"
      ;;
    *)
      echo "[ERR] Unsupported stage '${stage}'" >&2
      exit 1
      ;;
  esac
}

for k in "${KEYS[@]}"; do
  echo "########## Processing key: ${k} ##########"
  for stage in "${PIPELINE[@]}"; do
    run_stage "${stage}" "${k}"
  done
done

echo "[DONE] Processed keys: ${KEYS[*]}"

