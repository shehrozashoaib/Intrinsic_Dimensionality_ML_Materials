#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DIMENET_ENV_PREFIX="${DIMENET_ENV_PREFIX:-${PROJECT_DIR}/envs/pydimnet}"
export DIMENET_PYTHON="${DIMENET_PYTHON:-${DIMENET_ENV_PREFIX}/bin/python}"

if [[ ! -x "$DIMENET_PYTHON" ]]; then
  echo "[DimeNet++] ERROR: Python not found at ${DIMENET_PYTHON}" >&2
  echo "[DimeNet++] Run scripts/setup_dimenetpp_env.sh first, or set DIMENET_ENV_PREFIX." >&2
  return 1 2>/dev/null || exit 1
fi

NVIDIA_LIBS="$("$DIMENET_PYTHON" - <<'PY'
import pathlib
import site

paths = []
for base in site.getsitepackages():
    nvidia = pathlib.Path(base) / "nvidia"
    if not nvidia.exists():
        continue
    for lib in sorted(nvidia.glob("*/lib")):
        if lib.is_dir():
            paths.append(str(lib))
print(":".join(paths))
PY
)"

if [[ -n "$NVIDIA_LIBS" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${DIMENET_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${DIMENET_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="${PROJECT_DIR}/dimenetpp_code_only:${PYTHONPATH:-}"
