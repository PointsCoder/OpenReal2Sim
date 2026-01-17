#!/usr/bin/env bash
# sim_agent.sh - Runs the full pipeline for each key sequentially
# Each key runs through all stages (usd_conversion, sim_heuristic_manip, sim_randomize_rollout)
# Per-key logs are saved to logs/<key>.log
# If a key fails, the script continues to the next key

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd /app && pwd)"

# Default parameters
gpu_id="1"
CONFIG_PATH="${ROOT_DIR}/config/config.yaml"
declare -a PIPELINE=()
START_IDX=""
END_IDX=""

# Logging directory
LOG_DIR="${ROOT_DIR}/logs"
SUMMARY_FILE="${LOG_DIR}/pipeline_summary_$(date +%Y%m%d_%H%M%S).csv"

usage() {
  cat <<'EOF'
Usage: sim_agent.sh [options]

Options:
  --config <path>        Path to config.yaml file (default: /app/config/config.yaml)
  --stage <stage>         Stage to run (can be specified multiple times)
                          Available stages:
                            - usd_conversion
                            - sim_heuristic_manip
                            - sim_randomize_rollout
                          If not specified, runs all stages in order
  --start <index>         Start index (0-based) for keys to process (inclusive)
  --end <index>           End index (0-based) for keys to process (exclusive)
  -h, --help             Show this message and exit

Per-key logs are saved to logs/<key>.log
A summary CSV is saved to logs/pipeline_summary_<timestamp>.csv

Examples:
  sim_agent.sh --start 10 --end 20
  sim_agent.sh --start 5
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:?Missing value for --config}"
      shift 2
      ;;
    --stage)
      STAGE="${2:?Missing value for --stage}"
      case "${STAGE}" in
        usd_conversion|sim_heuristic_manip|sim_randomize_rollout)
          PIPELINE+=("${STAGE}")
          ;;
        *)
          echo "[ERR] Invalid stage: ${STAGE}" >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --start)
      START_IDX="${2:?Missing value for --start}"
      if ! [[ "${START_IDX}" =~ ^[0-9]+$ ]]; then
        echo "[ERR] --start must be a non-negative integer" >&2
        exit 1
      fi
      shift 2
      ;;
    --end)
      END_IDX="${2:?Missing value for --end}"
      if ! [[ "${END_IDX}" =~ ^[0-9]+$ ]]; then
        echo "[ERR] --end must be a non-negative integer" >&2
        exit 1
      fi
      shift 2
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

KEYS=($(python3 -c "
import yaml
import sys
try:
    with open('${CONFIG_PATH}', 'r') as f:
        cfg = yaml.safe_load(f)
        all_keys = cfg.get('keys', [])
        if not all_keys:
            print('[ERR] No keys found in config.yaml', file=sys.stderr)
            sys.exit(1)
        
        start_idx = ${START_IDX:--1}
        end_idx = ${END_IDX:--1}
        
        if start_idx >= 0 and end_idx >= 0:
            keys = all_keys[start_idx:end_idx]
        elif start_idx >= 0:
            keys = all_keys[start_idx:]
        elif end_idx >= 0:
            keys = all_keys[:end_idx]
        else:
            keys = all_keys
        
        if not keys:
            print('[ERR] No keys in specified range', file=sys.stderr)
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

# Setup
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

# Default pipeline
if [[ ${#PIPELINE[@]} -eq 0 ]]; then
  PIPELINE=("usd_conversion" "sim_heuristic_manip" "sim_randomize_rollout")
fi

echo "========================================"
echo "  sim_agent.sh - Pipeline Runner"
echo "========================================"
echo "Keys to process: ${#KEYS[@]}"
echo "Stages: ${PIPELINE[*]}"
echo "Log directory: ${LOG_DIR}"
echo "========================================"
echo ""

# Initialize summary CSV
echo "key,status,failed_stage,duration_seconds,timestamp" > "${SUMMARY_FILE}"

# Counters
TOTAL_KEYS=${#KEYS[@]}
SUCCESSFUL_KEYS=0
FAILED_KEYS=0

# Process each key
for key in "${KEYS[@]}"; do
  KEY_LOG="${LOG_DIR}/${key}.log"
  KEY_START_TIME=$(date +%s)
  KEY_STATUS="SUCCESS"
  FAILED_STAGE=""
  
  echo ""
  echo "########################################################"
  echo "# Processing key: ${key}"
  echo "# Log file: ${KEY_LOG}"
  echo "########################################################"
  echo ""
  
  # Clear/create the key log file
  echo "========================================" > "${KEY_LOG}"
  echo "Key: ${key}" >> "${KEY_LOG}"
  echo "Started: $(date)" >> "${KEY_LOG}"
  echo "Stages: ${PIPELINE[*]}" >> "${KEY_LOG}"
  echo "========================================" >> "${KEY_LOG}"
  echo "" >> "${KEY_LOG}"
  
  # Run each stage for this key
  for stage in "${PIPELINE[@]}"; do
    echo "--- Running stage: ${stage} ---"
    echo "" >> "${KEY_LOG}"
    echo "========================================" >> "${KEY_LOG}"
    echo "Stage: ${stage}" >> "${KEY_LOG}"
    echo "Started: $(date)" >> "${KEY_LOG}"
    echo "========================================" >> "${KEY_LOG}"
    
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    stage_exit_code=0
    
    case "${stage}" in
      usd_conversion)
        # Pipe 'Yes' to auto-accept NVIDIA EULA
        echo "Yes" | python openreal2sim/simulation/isaaclab/demo/sim_utils_demo/usd_conversion.py \
          --key "${key}" --headless 2>&1 | tee -a "${KEY_LOG}"
        stage_exit_code=${PIPESTATUS[1]}
        ;;
      sim_heuristic_manip)
        python openreal2sim/simulation/isaaclab/demo/sim_heuristic_manip.py \
          --key "${key}" --headless 2>&1 | tee -a "${KEY_LOG}"
        stage_exit_code=${PIPESTATUS[0]}
        ;;
      sim_randomize_rollout)
        python openreal2sim/simulation/isaaclab/demo/sim_randomize_rollout.py \
          --key "${key}" --headless 2>&1 | tee -a "${KEY_LOG}"
        stage_exit_code=${PIPESTATUS[0]}
        ;;
      *)
        echo "[ERR] Unknown stage: ${stage}" | tee -a "${KEY_LOG}"
        stage_exit_code=1
        ;;
    esac
    
    echo "" >> "${KEY_LOG}"
    echo "Stage ${stage} exit code: ${stage_exit_code}" >> "${KEY_LOG}"
    
    if [[ ${stage_exit_code} -ne 0 ]]; then
      echo "[✗] Stage '${stage}' FAILED for key '${key}' (exit code: ${stage_exit_code})"
      echo "[✗] FAILED: ${stage}" >> "${KEY_LOG}"
      KEY_STATUS="FAILED"
      FAILED_STAGE="${stage}"
      break
    else
      echo "[✓] Stage '${stage}' completed for key '${key}'"
      echo "[✓] SUCCESS: ${stage}" >> "${KEY_LOG}"
    fi
  done
  
  # Finalize key log
  KEY_END_TIME=$(date +%s)
  KEY_DURATION=$((KEY_END_TIME - KEY_START_TIME))
  
  echo "" >> "${KEY_LOG}"
  echo "========================================" >> "${KEY_LOG}"
  echo "Key: ${key}" >> "${KEY_LOG}"
  echo "Status: ${KEY_STATUS}" >> "${KEY_LOG}"
  echo "Duration: ${KEY_DURATION} seconds" >> "${KEY_LOG}"
  echo "Finished: $(date)" >> "${KEY_LOG}"
  echo "========================================" >> "${KEY_LOG}"
  
  # Update counters and summary
  if [[ "${KEY_STATUS}" == "SUCCESS" ]]; then
    ((SUCCESSFUL_KEYS++))
    echo "[✓✓✓] Key '${key}' completed successfully in ${KEY_DURATION}s"
  else
    ((FAILED_KEYS++))
    echo "[✗✗✗] Key '${key}' FAILED at stage '${FAILED_STAGE}' after ${KEY_DURATION}s"
  fi
  
  # Log to summary CSV
  echo "\"${key}\",\"${KEY_STATUS}\",\"${FAILED_STAGE}\",${KEY_DURATION},\"$(date +%Y-%m-%dT%H:%M:%S)\"" >> "${SUMMARY_FILE}"
  
  echo ""
  echo "--- Moving to next key ---"
  echo ""
done

# Final summary
echo ""
echo "========================================"
echo "           EXECUTION SUMMARY            "
echo "========================================"
echo "Total keys: ${TOTAL_KEYS}"
echo "Successful: ${SUCCESSFUL_KEYS}"
echo "Failed: ${FAILED_KEYS}"
echo ""
echo "Summary file: ${SUMMARY_FILE}"
echo "Individual logs: ${LOG_DIR}/<key>.log"
echo ""

if [[ ${FAILED_KEYS} -gt 0 ]]; then
  echo "Failed keys:"
  grep 'FAILED' "${SUMMARY_FILE}" | cut -d',' -f1 | tr -d '"' | while read -r k; do
    echo "  - ${k}"
  done
fi

echo ""
echo "[DONE] Pipeline complete."
