#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_CONDA_PY="/home/feilongtang/anaconda3/bin/python"

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi
  if [[ -x "$DEFAULT_CONDA_PY" ]]; then
    printf '%s\n' "$DEFAULT_CONDA_PY"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  command -v python
}

PYTHON_BIN="$(pick_python)"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
FORCE_TORCH_INSTALL="${FORCE_TORCH_INSTALL:-0}"

cd "$ROOT"

echo "[BOOTSTRAP] root=$ROOT"
echo "[BOOTSTRAP] python=$PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys; print(sys.executable); print(sys.version)'

"$PYTHON_BIN" -m pip install --upgrade pip

if [[ "$FORCE_TORCH_INSTALL" == "1" ]]; then
  "$PYTHON_BIN" -m pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"
else
  if ! "$PYTHON_BIN" -c 'import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"
  else
    echo "[BOOTSTRAP] Reusing existing torch installation"
  fi
fi

"$PYTHON_BIN" -m pip install -r requirements-rtdetr-server.txt

"$PYTHON_BIN" - <<'PY'
import cv2
import torch
import ultralytics
import yaml

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_devices", torch.cuda.device_count())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"cuda_name[{idx}]", torch.cuda.get_device_name(idx))
print("ultralytics", ultralytics.__version__)
print("cv2", cv2.__version__)
print("yaml_ok", yaml.__version__)
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[BOOTSTRAP] nvidia-smi summary"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
fi
