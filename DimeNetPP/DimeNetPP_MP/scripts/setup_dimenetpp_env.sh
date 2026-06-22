#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/ibex/user/${USER}/miniconda3}"
ENV_PREFIX="${DIMENET_ENV_PREFIX:-${PROJECT_DIR}/envs/pydimnet}"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "[DimeNet++] ERROR: conda.sh not found under ${CONDA_BASE}" >&2
  exit 1
fi

mkdir -p "$(dirname "$ENV_PREFIX")"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.12 pip
fi

"${ENV_PREFIX}/bin/python" -m pip install --upgrade pip
"${ENV_PREFIX}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements_dimenetpp.txt"
# kgcnn 4.x is the Keras-3 compatible line the v3 run scripts target (input_node_embedding /
# input_tensor_type / output_tensor_type). 2.1.0 is Keras-2 only and its DimeNet++ layers break
# under Keras 3. dm-tree ('tree') is a runtime import kgcnn 4.x needs but does not pin.
"${ENV_PREFIX}/bin/python" -m pip install kgcnn==4.0.2 dm-tree

"${ENV_PREFIX}/bin/python" - <<'PY'
import tensorflow as tf
import kgcnn
import jarvis
import pymatgen
# Validate the symbols the data-build and training paths actually import.
from kgcnn.data.crystal import CrystalDataset
from kgcnn.literature.DimeNetPP import make_crystal_model
print("tensorflow", tf.__version__)
print("kgcnn", getattr(kgcnn, "__version__", "2.1.0 (no __version__ attr)"))
print("DimeNet++ env import check OK")
PY
