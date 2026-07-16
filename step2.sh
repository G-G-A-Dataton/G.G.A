#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPETITION_DATA_PATH=""
EXTRA_DATA_PATH=""
MODEL_DUMP_PATH=""

usage() {
  printf '%s\n' \
    'Usage: bash step2.sh --competition_data_path PATH --extra_data_path PATH --model_dump_path PATH'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --competition_data_path)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      COMPETITION_DATA_PATH="$2"
      shift 2
      ;;
    --extra_data_path)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      EXTRA_DATA_PATH="$2"
      shift 2
      ;;
    --model_dump_path)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      MODEL_DUMP_PATH="$2"
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

[[ -n "${COMPETITION_DATA_PATH}" && -n "${EXTRA_DATA_PATH}" && -n "${MODEL_DUMP_PATH}" ]] || {
  usage >&2
  exit 2
}

PYTHON_BIN="${GGA_PYTHON:-${GGA_ENV_PATH:-${ROOT_DIR}/.solution_venv}/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || {
  printf 'Solution environment is missing. Run bash step1.sh first.\n' >&2
  exit 1
}
[[ -d "${COMPETITION_DATA_PATH}" ]] || {
  printf 'Competition data directory does not exist: %s\n' "${COMPETITION_DATA_PATH}" >&2
  exit 1
}
if [[ -d "${MODEL_DUMP_PATH}" ]] && find "${MODEL_DUMP_PATH}" -mindepth 1 -print -quit | grep -q .; then
  printf 'Model dump directory must be empty: %s\n' "${MODEL_DUMP_PATH}" >&2
  exit 1
fi
mkdir -p "${EXTRA_DATA_PATH}" "${MODEL_DUMP_PATH}"

if [[ -f "${ROOT_DIR}/source_revision.txt" ]]; then
  CODE_REVISION="$(tr -d '[:space:]' < "${ROOT_DIR}/source_revision.txt")"
else
  [[ -z "$(git -C "${ROOT_DIR}" status --porcelain)" ]] || {
    printf 'Refusing to train from a dirty source worktree.\n' >&2
    exit 1
  }
  CODE_REVISION="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
fi
[[ "${CODE_REVISION}" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'Invalid source revision: %s\n' "${CODE_REVISION}" >&2
  exit 1
}

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/training/run_model_shortlist.py" \
  --data-dir "${COMPETITION_DATA_PATH}" \
  --artifact-dir "${MODEL_DUMP_PATH}" \
  --candidate-output "${EXTRA_DATA_PATH}/generated_training_candidates.csv" \
  --code-revision "${CODE_REVISION}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/delivery/write_deploy_decision.py" \
  --competition-data-path "${COMPETITION_DATA_PATH}" \
  --model-dump-path "${MODEL_DUMP_PATH}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/delivery/write_extra_data_manifest.py" \
  --candidate-path "${EXTRA_DATA_PATH}/generated_training_candidates.csv" \
  --model-dump-path "${MODEL_DUMP_PATH}" \
  --output "${EXTRA_DATA_PATH}/extra_data_manifest.json"

printf 'Training complete. Models: %s\n' "${MODEL_DUMP_PATH}"
printf 'Generated candidate data: %s\n' "${EXTRA_DATA_PATH}"
