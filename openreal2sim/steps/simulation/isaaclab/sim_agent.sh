#!/usr/bin/env bash
# Thin wrapper over openreal2sim/steps/simulation/simulation.py that keeps the
# key slicing of the old script. The runner owns the stage list and the per step
# subprocesses; run it directly when you do not need --start or --end.
#set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
RUNNER="${ROOT_DIR}/openreal2sim/steps/simulation/simulation.py"

CONFIG_PATH="${ROOT_DIR}/config/config.yaml"
STAGE=""
START_IDX=""
END_IDX=""

usage() {
  cat <<'EOF'
Usage: sim_agent.sh [options]

Options:
  --config <path>        Path to config.yaml file (default: <repo>/config/config.yaml)
  --stage <stage>        Start from this step, one of
                            - usd_conversion
                            - sim_heuristic_manip
                            - sim_randomize_rollout
                         If not specified, runs all steps in order
  --start <index>        Start index (0-based) for keys to process (inclusive)
  --end <index>          End index (0-based) for keys to process (exclusive)
                         If specified, only processes keys[start:end]
  -h, --help             Show this message and exit

The script loads keys from config.yaml and calls the runner once per key:
  python openreal2sim/steps/simulation/simulation.py --key <key>

Examples:
  # Process keys from index 10 to 20 (exclusive)
  sim_agent.sh --start 10 --end 20

  # Process keys from index 5 to the end
  sim_agent.sh --start 5
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="${2:?Missing value for --config}"
      shift 2
      ;;
    --stage)
      STAGE="${2:?Missing value for --stage}"
      case "${STAGE}" in
        usd_conversion|sim_heuristic_manip|sim_randomize_rollout) ;;
        *)
          echo "[ERR] Invalid stage: ${STAGE}" >&2
          echo "[ERR] Valid stages: usd_conversion, sim_heuristic_manip, sim_randomize_rollout" >&2
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

RANGE_INFO=""
if [[ -n "${START_IDX}" ]] || [[ -n "${END_IDX}" ]]; then
  RANGE_INFO=" (range: ${START_IDX:-0}:${END_IDX:-end})"
fi
echo "[INFO] Loaded ${#KEYS[@]} key(s) from ${CONFIG_PATH}${RANGE_INFO}: ${KEYS[*]}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
cd "${ROOT_DIR}"

for k in "${KEYS[@]}"; do
  echo "########## Processing key: ${k} ##########"
  if [[ -n "${STAGE}" ]]; then
    python3 "${RUNNER}" --config "${CONFIG_PATH}" --key "${k}" --stage "${STAGE}" || exit 1
  else
    python3 "${RUNNER}" --config "${CONFIG_PATH}" --key "${k}" || exit 1
  fi
done

echo "[DONE] Processed keys: ${KEYS[*]}"
