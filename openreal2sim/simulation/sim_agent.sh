#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default parameters
KEY="demo_video"
STAGE=""
NUM_ENVS=1
NUM_TRIALS=1
DEVICE="cuda:0"
HEADLESS="--headless"
TOTAL_NUM=10

usage() {
  cat <<'EOF'
Usage: sim_agent.sh [options]

Options:
  --key <name>           Scene key to process (default: demo_video)
  --stage <stage>        Starting stage: usd_conversion | sim_heuristic_manip | sim_randomize_rollout
  --num_envs <int>       Number of parallel environments for heuristic/randomize (default: 1)
  --num_trials <int>     Number of trials for heuristic stage (default: 1)
  --device <str>         Device for simulations, e.g. cuda:0 (default: cuda:0)
  --no-headless          Disable headless mode (passes through to AppLauncher)
  -h, --help             Show this message and exit

The script runs the following stages in order (starting from --stage if provided):
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
    --stage)
      STAGE="${2:?Missing value for --stage}"
      shift 2
      ;;
    --num_envs)
      NUM_ENVS="${2:?Missing value for --num_envs}"
      shift 2
      ;;
    --num_trials)
      NUM_TRIALS="${2:?Missing value for --num_trials}"
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
declare -a PIPELINE=("usd_conversion" "sim_heuristic_manip" "sim_randomize_rollout")
declare -a RUN_STAGES=()
if [[ -z "${STAGE}" ]]; then
  RUN_STAGES=("${PIPELINE[@]}")
else
  found=""
  for stage in "${PIPELINE[@]}"; do
    if [[ -z "${found}" && "${stage}" == "${STAGE}" ]]; then
      found="yes"
    fi
    if [[ -n "${found}" ]]; then
      RUN_STAGES+=("${stage}")
    fi
  done
  if [[ ${#RUN_STAGES[@]} -eq 0 ]]; then
    echo "[ERR] Invalid --stage '${STAGE}'. Valid options: ${PIPELINE[*]}" >&2
    exit 1
  fi
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
cd "${ROOT_DIR}"

run_stage() {
  local stage="$1"
  echo "=============================="
  echo "[RUN] Stage: ${stage}"
  echo "=============================="
  case "${stage}" in
    usd_conversion)
      python pi_i/simulation/isaaclab/sim_preprocess/usd_conversion.py ${HEADLESS}
      ;;
    sim_heuristic_manip)
      python pi_i/simulation/isaaclab/sim_heuristic_manip.py \
        --key "${KEY}" \
        --num_envs "${NUM_ENVS}" \
        --num_trials "${NUM_TRIALS}" \
        --device "${DEVICE}" \
        ${HEADLESS}
      ;;
    sim_randomize_rollout)
      python pi_i/simulation/isaaclab/sim_randomize_rollout.py \
        --key "${KEY}" \
        --num_envs "${NUM_ENVS}" \
        --device "${DEVICE}" \
        -- total_num "${TOTAL_NUM}" \
        ${HEADLESS}
      ;;
    *)
      echo "[ERR] Unsupported stage '${stage}'" >&2
      exit 1
      ;;
  esac
}

for stage in "${RUN_STAGES[@]}"; do
  run_stage "${stage}"
done

echo "[DONE] Pipeline completed: ${RUN_STAGES[*]}"

