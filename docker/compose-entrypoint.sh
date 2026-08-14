#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME:-psed}"
ENV_DIR="/opt/conda/envs/${ENV_NAME}"

if [ ! -d "${ENV_DIR}" ]; then
  conda env create -f /workspace/environment.yml
else
  if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo "Removing incomplete conda env at ${ENV_DIR}"
    conda env remove -n "${ENV_NAME}" -y || rm -rf "${ENV_DIR}"
    conda env create -f /workspace/environment.yml
  else
    conda env update -n "${ENV_NAME}" -f /workspace/environment.yml --prune
  fi
fi

cat >/root/.bashrc <<EOF
source /opt/conda/etc/profile.d/conda.sh
conda activate ${ENV_NAME}
cd /workspace
EOF

source /opt/conda/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

python - <<'PY'
import shutil
import subprocess

for binary in ("ffmpeg", "ffprobe"):
    path = shutil.which(binary)
    if path is None:
        raise SystemExit(f"{binary} was not found in PATH")
    version_line = subprocess.check_output([path, "-version"], text=True).splitlines()[0]
    print(f"{binary}: {version_line}")

import torch
print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}, cuda_available: {torch.cuda.is_available()}")

import torchcodec
print(f"torchcodec: {torchcodec.__version__}")
PY

exec "$@"
