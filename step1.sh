#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="${GGA_ENV_PATH:-${ROOT_DIR}/.solution_venv}"

usage() {
  printf 'Usage: bash step1.sh [--env-path PATH]\n'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-path)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      ENV_PATH="$2"
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

if command -v python3.13 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v python3.13)"
else
  BOOTSTRAP_PYTHON="$(command -v python3 || true)"
fi
[[ -n "${BOOTSTRAP_PYTHON}" ]] || {
  printf 'Python 3.13.5 is required but no Python interpreter was found.\n' >&2
  exit 1
}

ACTUAL_VERSION="$(${BOOTSTRAP_PYTHON} -c 'import platform; print(platform.python_version())')"
[[ "${ACTUAL_VERSION}" == "3.13.5" ]] || {
  printf 'Python 3.13.5 is required; found %s at %s.\n' \
    "${ACTUAL_VERSION}" "${BOOTSTRAP_PYTHON}" >&2
  exit 1
}

"${BOOTSTRAP_PYTHON}" -m venv "${ENV_PATH}"
"${ENV_PATH}/bin/python" -m pip install \
  --require-hashes \
  --requirement "${ROOT_DIR}/requirements.lock"
PYTHONNOUSERSITE=1 "${ENV_PATH}/bin/python" \
  "${ROOT_DIR}/scripts/verify_environment.py" \
  --lock "${ROOT_DIR}/requirements.lock"

printf 'Environment ready: %s\n' "${ENV_PATH}"
printf 'step2.sh and step3.sh will use this interpreter automatically.\n'
