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

pick_device() {
  if [[ -n "${DEVICE:-}" ]]; then
    printf '%s\n' "$DEVICE"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | sort -t',' -k2,2n -k3,3n | head -n 1 | cut -d',' -f1 | tr -d ' '
    return 0
  fi
  printf '%s\n' 0
}

PYTHON_BIN="$(pick_python)"
DEVICE="$(pick_device)"
RUN_NAME="${RUN_NAME:-main_rtdetr_100ep_server}"
WORKERS="${WORKERS:-8}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-100}"

cd "$ROOT"

echo "[PHASE3] root=$ROOT"
echo "[PHASE3] python=$PYTHON_BIN"
echo "[PHASE3] device=$DEVICE"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
fi

"$PYTHON_BIN" scripts/data/prepare_rtdetr_data.py --dataset all --overwrite

"$PYTHON_BIN" scripts/train/train_rtdetr.py \
  --project runs/detect \
  --data configs/kvasir_det.yaml \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --name "$RUN_NAME"

"$PYTHON_BIN" scripts/eval/eval_rtdetr.py \
  --weights "runs/detect/${RUN_NAME}/weights/best.pt" \
  --device "$DEVICE" \
  --project runs/detect \
  --workers "$WORKERS" \
  --batch "$BATCH" \
  --imgsz "$IMGSZ" \
  --output results/summary/rtdetr_eval_summary.json
