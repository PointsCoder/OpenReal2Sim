#!/usr/bin/env bash
# motion_agent.sh - Runs the motion pipeline for each key sequentially
# Simulates sim_agent.sh behavior but for the motion module
# Logs execution status and errors to logs/motion_pipeline_summary.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd /app && pwd)"

# Default parameters
CONFIG_PATH="${ROOT_DIR}/config/config.yaml"
START_IDX=""
END_IDX=""
STAGE=""

# Logging directory
LOG_DIR="${ROOT_DIR}/logs"
SUMMARY_FILE="${LOG_DIR}/motion_pipeline_summary.csv"

usage() {
  cat <<'EOF'
Usage: motion_agent.sh [options]

Options:
  --config <path>        Path to config.yaml file (default: /app/config/config.yaml)
  --stage <stage>         Starting stage (passed to motion_manager.py)
  --start <index>         Start index (0-based) for keys to process (inclusive)
  --end <index>           End index (0-based) for keys to process (exclusive)
  -h, --help             Show this message and exit

Per-key logs are saved to logs/<key>_motion.log
A summary CSV is saved to logs/motion_pipeline_summary.csv

Examples:
  motion_agent.sh --start 10 --end 20
  motion_agent.sh --start 5 --stage hand_extraction
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
      shift 2
      ;;
    --start)
      START_IDX="${2:?Missing value for --start}"
      shift 2
      ;;
    --end)
      END_IDX="${2:?Missing value for --end}"
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
  echo "[ERR] No keys found to process." >&2
  exit 1
fi

# Setup
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
cd "${ROOT_DIR}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "  motion_agent.sh - Motion Pipeline Runner"
echo "========================================"
echo "Keys to process: ${#KEYS[@]}"
echo "Log directory: ${LOG_DIR}"
echo "Summary file: ${SUMMARY_FILE}"
echo "========================================"
echo ""

# Initialize summary CSV if it doesn't exist, otherwise append
if [[ ! -f "${SUMMARY_FILE}" ]]; then
    echo "key,status,duration_seconds,timestamp,error_message" > "${SUMMARY_FILE}"
fi

# Counters
TOTAL_KEYS=${#KEYS[@]}
SUCCESSFUL_KEYS=0
FAILED_KEYS=0

# Process each key
for key in "${KEYS[@]}"; do
  KEY_LOG="${LOG_DIR}/${key}_motion.log"
  KEY_START_TIME=$(date +%s)
  KEY_STATUS="SUCCESS"
  ERROR_MSG=""
  
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
  echo "========================================" >> "${KEY_LOG}"
  
  # Construct command
  CMD_ARGS="--key ${key}"
  if [[ -n "${STAGE}" ]]; then
      CMD_ARGS="${CMD_ARGS} --stage ${STAGE}"
  fi
  
  echo "Running: python openreal2sim/motion/motion_manager.py ${CMD_ARGS}" | tee -a "${KEY_LOG}"
  
  # Run motion_manager
  # We pipe output to tee to show it and save it.
  # We use pipefail to catch python exit code.
  set -o pipefail
  python openreal2sim/motion/motion_manager.py ${CMD_ARGS} 2>&1 | tee -a "${KEY_LOG}"
  EXIT_CODE=$?
  set +o pipefail
  
  echo "" >> "${KEY_LOG}"
  echo "Exit code: ${EXIT_CODE}" >> "${KEY_LOG}"
  
  if [[ ${EXIT_CODE} -ne 0 ]]; then
      echo "[✗] Motion pipeline FAILED for key '${key}' (exit code: ${EXIT_CODE})"
      echo "[✗] FAILED" >> "${KEY_LOG}"
      KEY_STATUS="FAILED"
      ((FAILED_KEYS++))
      
      # Extract full log as error message, escape quotes for CSV
      # Replacing newlines with spaces to keep CSV row on one line for safety
      ERROR_MSG=$(cat "${KEY_LOG}" | tr '\n' ' ' | sed 's/"/""/g')
  else
      echo "[✓] Motion pipeline completed for key '${key}'"
      echo "[✓] SUCCESS" >> "${KEY_LOG}"
      ((SUCCESSFUL_KEYS++))
  fi
  
  KEY_END_TIME=$(date +%s)
  KEY_DURATION=$((KEY_END_TIME - KEY_START_TIME))
  
  # Log to summary CSV
  # CSV Format: key, status, duration, timestamp, "error_message"
  echo "\"${key}\",\"${KEY_STATUS}\",${KEY_DURATION},\"$(date +%Y-%m-%dT%H:%M:%S)\",\"${ERROR_MSG}\"" >> "${SUMMARY_FILE}"
  
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
echo ""
echo "[DONE] Pipeline complete."
