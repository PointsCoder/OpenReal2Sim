#!/usr/bin/env bash
# sim_agent.sh - Runs the full pipeline for each key sequentially
# Each key runs through all stages (usd_conversion, sim_heuristic_manip, sim_randomize_rollout)
# Per-key logs are saved to logs/<key>.log
# If a key fails, the script continues to the next key

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd /app && pwd)"

# Default parameters
gpu_id="0"  # Will be dynamically selected
CONFIG_PATH="${ROOT_DIR}/config/config.yaml"
declare -a PIPELINE=()
START_IDX=""
END_IDX=""

# Maximum time (in seconds) allowed per key before timeout
# 3 hours = 10800 seconds
MAX_KEY_TIMEOUT=10800

# Function to select GPU with most free memory
select_best_gpu() {
  # Query nvidia-smi for GPU memory free (in MiB)
  # Format: gpu_id,memory_free_mib
  local best_gpu="0"
  local max_free="0"
  
  while IFS=',' read -r gpu_idx mem_free; do
    # Remove any whitespace
    gpu_idx=$(echo "$gpu_idx" | tr -d ' ')
    mem_free=$(echo "$mem_free" | tr -d ' MiB')
    
    if [[ -n "$mem_free" && "$mem_free" -gt "$max_free" ]]; then
      max_free="$mem_free"
      best_gpu="$gpu_idx"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
  
  echo "$best_gpu"
}

# Function to check GPU health via a quick CUDA operation
gpu_health_check() {
  local gpu="$1"
  echo "[INFO] Running GPU health check on GPU ${gpu}..."
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -c "
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print('[GPU-CHECK] CUDA not available', file=sys.stderr)
        sys.exit(1)
    # Quick tensor operation to verify GPU is functional
    a = torch.randn(64, 64, device='cuda:0')
    b = torch.randn(64, 64, device='cuda:0')
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    del a, b, c
    torch.cuda.empty_cache()
    print('[GPU-CHECK] GPU ${gpu} is healthy')
except Exception as e:
    print(f'[GPU-CHECK] GPU health check failed: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1
  return $?
}

# Patterns that identify OUR pipeline processes (used to catch reparented orphans)
# These are specific enough to never match other users' or our other work
SIM_PROCESS_PATTERNS="sim_heuristic_manip|sim_randomize_rollout|usd_conversion"

# Function to clean up all processes from a pipeline key execution
# Phase 1: Kill the process tree by parent PID (catches normal children)
# Phase 2: Kill reparented orphans by matching our pipeline script names
#           (only targets current user's processes matching our specific scripts)
cleanup_key_processes() {
  local parent_pid="$1"
  local current_user
  current_user=$(whoami)
  
  echo "[INFO] Phase 1: Killing process tree rooted at PID ${parent_pid}..."
  
  # Recursively collect all descendant PIDs
  local all_pids=""
  local queue="${parent_pid}"
  while [[ -n "$queue" ]]; do
    local next_queue=""
    for ppid in $queue; do
      local children
      children=$(pgrep -P "$ppid" 2>/dev/null || true)
      if [[ -n "$children" ]]; then
        all_pids="$all_pids $children"
        next_queue="$next_queue $children"
      fi
    done
    queue="$next_queue"
  done
  all_pids="${parent_pid} ${all_pids}"
  
  # Kill live ones
  local live_pids=""
  for pid in $all_pids; do
    kill -0 "$pid" 2>/dev/null && live_pids="$live_pids $pid"
  done
  
  if [[ -n "$live_pids" ]]; then
    echo "[INFO] Tree PIDs:${live_pids}"
    for pid in $(echo "$live_pids" | tr ' ' '\n' | tac); do
      kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in $live_pids; do
      kill -0 "$pid" 2>/dev/null && { kill -9 "$pid" 2>/dev/null || true; }
    done
  else
    echo "[INFO] No live processes in tree"
  fi
  
  sleep 2
  
  # Phase 2: Catch reparented orphans matching our pipeline scripts
  # Use pgrep to find python processes whose cmdline contains our script names
  # IMPORTANT: Exclude our own PID ($$) and parent to avoid self-kill
  echo "[INFO] Phase 2: Scanning for reparented orphans (user: ${current_user})..."
  local my_pid=$$
  local orphan_pids
  orphan_pids=$(pgrep -u "${current_user}" -f "${SIM_PROCESS_PATTERNS}" 2>/dev/null \
    | grep -v -w -e "${my_pid}" -e "${parent_pid}" -e "$PPID" \
    || true)
  
  # Further filter: only keep PIDs that are actually python processes, not our shell script
  local confirmed_orphans=""
  for pid in $orphan_pids; do
    local proc_name
    proc_name=$(ps -o comm= -p "$pid" 2>/dev/null || echo "")
    # Only kill python/python3 processes, not bash/sh (which could be our script)
    if [[ "$proc_name" == python* ]]; then
      confirmed_orphans="$confirmed_orphans $pid"
    fi
  done
  
  if [[ -z "$confirmed_orphans" ]]; then
    echo "[INFO] No reparented orphans found"
  else
    echo "[WARN] Found reparented orphans:${confirmed_orphans}"
    for pid in $confirmed_orphans; do
      local cmdline
      cmdline=$(ps -o args= -p "$pid" 2>/dev/null || echo "unknown")
      echo "[INFO] Killing orphan PID ${pid}: ${cmdline}"
      kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in $confirmed_orphans; do
      if kill -0 "$pid" 2>/dev/null; then
        echo "[WARN] SIGKILL orphan PID ${pid}"
        kill -9 "$pid" 2>/dev/null || true
      fi
    done
  fi
  
  sleep 2
  echo "[INFO] Key process cleanup complete"
}

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
echo "Max timeout per key: ${MAX_KEY_TIMEOUT}s ($(( MAX_KEY_TIMEOUT / 3600 ))h)"
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
  
  # Dynamically select the GPU with most free memory
  gpu_id=$(select_best_gpu)
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  echo "# Selected GPU: ${gpu_id} (most free memory)"
  echo "########################################################"
  echo ""
  
  # Clear/create the key log file
  echo "========================================" > "${KEY_LOG}"
  echo "Key: ${key}" >> "${KEY_LOG}"
  echo "Started: $(date)" >> "${KEY_LOG}"
  echo "Stages: ${PIPELINE[*]}" >> "${KEY_LOG}"
  echo "Selected GPU: ${gpu_id}" >> "${KEY_LOG}"
  echo "Max timeout: ${MAX_KEY_TIMEOUT}s" >> "${KEY_LOG}"
  echo "========================================" >> "${KEY_LOG}"
  echo "" >> "${KEY_LOG}"
  
  # Pre-key GPU health check — don't waste time if GPU is already dead
  echo "[INFO] Pre-key GPU health check..." | tee -a "${KEY_LOG}"
  GPU_HEALTHY=false
  if gpu_health_check "${gpu_id}" >> "${KEY_LOG}" 2>&1; then
    GPU_HEALTHY=true
    echo "[✓] GPU ${gpu_id} is healthy, proceeding" | tee -a "${KEY_LOG}"
  else
    echo "[WARN] GPU ${gpu_id} failed health check, trying other GPUs..." | tee -a "${KEY_LOG}"
    # Try each available GPU
    while IFS=',' read -r try_gpu _; do
      try_gpu=$(echo "$try_gpu" | tr -d ' ')
      [[ "$try_gpu" == "$gpu_id" ]] && continue  # Skip the one we already tried
      echo "[INFO] Trying GPU ${try_gpu}..." | tee -a "${KEY_LOG}"
      if gpu_health_check "${try_gpu}" >> "${KEY_LOG}" 2>&1; then
        gpu_id="${try_gpu}"
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        GPU_HEALTHY=true
        echo "[✓] GPU ${gpu_id} is healthy, switching to it" | tee -a "${KEY_LOG}"
        break
      fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
  fi
  
  if [[ "${GPU_HEALTHY}" != "true" ]]; then
    echo "[CRITICAL] No healthy GPU found! All GPUs are in bad state." | tee -a "${KEY_LOG}"
    echo "[CRITICAL] Skipping key '${key}' and aborting pipeline." | tee -a "${KEY_LOG}"
    echo "[CRITICAL] You must restart the Docker container to recover." | tee -a "${KEY_LOG}"
    KEY_STATUS="GPU_DEAD"
    FAILED_STAGE="pre_gpu_check"
    
    # Finalize this key's log and summary
    KEY_END_TIME=$(date +%s)
    KEY_DURATION=$((KEY_END_TIME - KEY_START_TIME))
    echo "" >> "${KEY_LOG}"
    echo "========================================" >> "${KEY_LOG}"
    echo "Key: ${key}" >> "${KEY_LOG}"
    echo "Status: ${KEY_STATUS}" >> "${KEY_LOG}"
    echo "Duration: ${KEY_DURATION} seconds" >> "${KEY_LOG}"
    echo "Finished: $(date)" >> "${KEY_LOG}"
    echo "========================================" >> "${KEY_LOG}"
    ((FAILED_KEYS++))
    echo "[💀] Key '${key}' SKIPPED — GPU is dead (${KEY_DURATION}s)"
    echo "\"${key}\",\"${KEY_STATUS}\",\"${FAILED_STAGE}\",${KEY_DURATION},\"$(date +%Y-%m-%dT%H:%M:%S)\"" >> "${SUMMARY_FILE}"
    
    echo ""
    echo "[CRITICAL] Aborting pipeline — no functional GPU available."
    echo "[CRITICAL] Restart the Docker container and re-run."
    break
  fi
  
  # Run all stages for this key inside a timeout wrapper
  # If the key exceeds MAX_KEY_TIMEOUT, it gets killed
  # timeout sends SIGTERM first, then SIGKILL after 60s if still alive
  # We run in background to capture the PID for targeted cleanup
  timeout --signal=TERM --kill-after=60 "${MAX_KEY_TIMEOUT}" bash -c '
    KEY_LOG="'"${KEY_LOG}"'"
    key="'"${key}"'"
    STAGES=('"$(printf "'%s' " "${PIPELINE[@]}")"')
    
    for stage in "${STAGES[@]}"; do
      echo "--- Running stage: ${stage} ---"
      echo "" >> "${KEY_LOG}"
      echo "========================================" >> "${KEY_LOG}"
      echo "Stage: ${stage}" >> "${KEY_LOG}"
      echo "Started: $(date)" >> "${KEY_LOG}"
      echo "========================================" >> "${KEY_LOG}"
      
      stage_exit_code=0
      
      case "${stage}" in
        usd_conversion)
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
        echo "[\u2717] Stage '"'"'${stage}'"'"' FAILED for key '"'"'${key}'"'"' (exit code: ${stage_exit_code})"
        echo "[\u2717] FAILED: ${stage}" >> "${KEY_LOG}"
        exit 1
      else
        echo "[\u2713] Stage '"'"'${stage}'"'"' completed for key '"'"'${key}'"'"'"
        echo "[\u2713] SUCCESS: ${stage}" >> "${KEY_LOG}"
        
        echo "[INFO] Cleaning up CUDA state..." >> "${KEY_LOG}"
        python3 -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize() if torch.cuda.is_available() else None" 2>/dev/null || true
        sleep 2
      fi
    done
  ' &
  TIMEOUT_PID=$!
  echo "[INFO] Timeout PID: ${TIMEOUT_PID}" >> "${KEY_LOG}"
  wait ${TIMEOUT_PID}
  SUBSHELL_EXIT=$?
  
  # Check if the subshell was killed by timeout
  # timeout sends SIGTERM (exit 124) or SIGKILL (exit 137)
  if [[ ${SUBSHELL_EXIT} -eq 124 || ${SUBSHELL_EXIT} -eq 137 ]]; then
    echo "[\u23f0] Key '${key}' TIMED OUT after ${MAX_KEY_TIMEOUT}s ($(( MAX_KEY_TIMEOUT / 3600 ))h)"
    echo "" >> "${KEY_LOG}"
    echo "[\u23f0] TIMEOUT: Key exceeded ${MAX_KEY_TIMEOUT}s limit" >> "${KEY_LOG}"
    KEY_STATUS="TIMEOUT"
    FAILED_STAGE="timeout"
  elif [[ ${SUBSHELL_EXIT} -ne 0 ]]; then
    KEY_STATUS="FAILED"
    # FAILED_STAGE was set inside the subshell but won't propagate; 
    # grep the log to find which stage failed
    FAILED_STAGE=$(grep -o '\[\u2717\] FAILED: [a-z_]*' "${KEY_LOG}" | tail -1 | sed 's/\[\u2717\] FAILED: //')
    [[ -z "${FAILED_STAGE}" ]] && FAILED_STAGE="unknown"
  fi
  
  # After timeout or failure, kill only the orphaned child processes from THIS key
  # This is targeted: only kills descendants of our timeout PID, not other user work
  if [[ "${KEY_STATUS}" != "SUCCESS" ]]; then
    echo "[INFO] Cleaning up orphaned processes from this key (tree root: ${TIMEOUT_PID})..." | tee -a "${KEY_LOG}"
    cleanup_key_processes "${TIMEOUT_PID}" 2>&1 | tee -a "${KEY_LOG}"
  fi
  
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
  elif [[ "${KEY_STATUS}" == "TIMEOUT" ]]; then
    ((FAILED_KEYS++))
    echo "[⏰⏰⏰] Key '${key}' TIMED OUT after ${KEY_DURATION}s"
  else
    ((FAILED_KEYS++))
    echo "[✗✗✗] Key '${key}' FAILED at stage '${FAILED_STAGE}' after ${KEY_DURATION}s"
  fi
  
  # Log to summary CSV
  echo "\"${key}\",\"${KEY_STATUS}\",\"${FAILED_STAGE}\",${KEY_DURATION},\"$(date +%Y-%m-%dT%H:%M:%S)\"" >> "${SUMMARY_FILE}"
  
  # GPU health check after each key to ensure clean state for the next key
  echo "" >> "${KEY_LOG}"
  echo "[INFO] Post-key GPU health check..." >> "${KEY_LOG}"
  if gpu_health_check "${gpu_id}" >> "${KEY_LOG}" 2>&1; then
    echo "[✓] GPU health check passed" | tee -a "${KEY_LOG}"
  else
    echo "[WARN] GPU health check FAILED on GPU ${gpu_id}" | tee -a "${KEY_LOG}"
    echo "[INFO] Waiting for GPU driver to release resources..." | tee -a "${KEY_LOG}"
    
    # Give the driver extra time to reclaim memory from killed processes
    sleep 10
    
    # Re-select the best GPU and retry health check
    gpu_id=$(select_best_gpu)
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    echo "[INFO] Re-selected GPU ${gpu_id} after cleanup" | tee -a "${KEY_LOG}"
    
    if gpu_health_check "${gpu_id}" >> "${KEY_LOG}" 2>&1; then
      echo "[✓] GPU health check passed after process cleanup" | tee -a "${KEY_LOG}"
    else
      echo "[CRITICAL] GPU still in bad state after killing all processes!" | tee -a "${KEY_LOG}"
      echo "[CRITICAL] Consider restarting the container if errors persist." | tee -a "${KEY_LOG}"
    fi
  fi
  
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
