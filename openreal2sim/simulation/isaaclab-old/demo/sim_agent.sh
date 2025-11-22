#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd /app && pwd)"

# Default parameters
KEY=""
KEY_FOLDER="${ROOT_DIR}/outputs"
gpu_id="1"
HEADLESS="--headless"

usage() {
  cat <<'EOF'
Usage: sim_agent.sh [options]

Options:
  --key <name>           Single scene key to process (optional)
  --folder <path>        Directory whose immediate subfolders are treated as keys
  --device <str>         Device for simulations, e.g. cuda:0 (default: cuda:0)
  --no-headless          Disable headless mode (passes through to AppLauncher)
  -h, --help             Show this message and exit

For each resolved key, the script runs:
  1. USD conversion (`isaaclab/sim_preprocess/usd_conversion.py`)
  2. Heuristic manipulation (`isaaclab/sim_heuristic_manip.py`)
  3. Randomized rollout (`isaaclab/sim_randomize_rollout.py`)
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --key)
      KEY="${2:?Missing value for --key}"
      shift 2
      ;;
    --folder)
      KEY_FOLDER="${2:?Missing value for --folder}"
      shift 2
      ;;
    --device)
      DEVICE="${2:?Missing value for --device}"
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

# Determine which stages to run
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
cd "${ROOT_DIR}"

declare -a PIPELINE=("usd_conversion" "sim_heuristic_manip" "sim_randomize_rollout")

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

collect_keys() {
  local -n _out=$1
  if [[ -n "${KEY}" ]]; then
    _out+=("${KEY}")
  fi
  if [[ -n "${KEY_FOLDER}" ]]; then
    local folder_abs
    folder_abs="$(cd "${KEY_FOLDER}" && pwd)"
    while IFS= read -r -d '' subdir; do
      _out+=("$(basename "${subdir}")")
    done < <(find "${folder_abs}" -mindepth 1 -maxdepth 1 -type d -print0)
  fi
}

declare -a KEYS=()
collect_keys KEYS

if [[ ${#KEYS[@]} -eq 0 ]]; then
  echo "[ERR] Please specify --key and/or --folder with at least one subdirectory." >&2
  exit 1
fi

for k in "${KEYS[@]}"; do
  echo "########## Processing key: ${k} ##########"
  for stage in "${PIPELINE[@]}"; do
    run_stage "${stage}" "${k}"
  done
done

echo "[DONE] Processed keys: ${KEYS[*]}"

