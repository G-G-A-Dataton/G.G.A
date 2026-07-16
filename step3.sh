#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DUMP_PATH=""
COMPETITION_DATA_PATH=""
OUT_PATH=""

usage() {
  printf '%s\n' \
    'Usage: bash step3.sh --model_dump_path PATH --competition_data_path PATH --out_path FILE'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_dump_path)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      MODEL_DUMP_PATH="$2"
      shift 2
      ;;
    --competition_data_path)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      COMPETITION_DATA_PATH="$2"
      shift 2
      ;;
    --out_path)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      OUT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "${MODEL_DUMP_PATH}" && -n "${COMPETITION_DATA_PATH}" && -n "${OUT_PATH}" ]] || {
  usage >&2
  exit 2
}

PYTHON_BIN="${GGA_PYTHON:-${GGA_ENV_PATH:-${ROOT_DIR}/.solution_venv}/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || {
  printf 'Solution environment is missing. Run bash step1.sh first.\n' >&2
  exit 1
}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/gga-matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/delivery/run_offline_inference.py" \
  --model-dump-path "${MODEL_DUMP_PATH}" \
  --competition-data-path "${COMPETITION_DATA_PATH}" \
  --out-path "${OUT_PATH}"
