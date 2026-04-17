# Reproduce The Experiments

Run all commands from the repository root.

## 1. Environment

Install the Python packages:

```bash
pip install -r requirements.txt
```

Install a matching PyTorch build for your CUDA or CPU environment from the official PyTorch selector:

- [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)

For the original server-oriented bootstrap flow, use:

```bash
bash bootstrap_rtdetr_env.sh
```

## 2. Dataset Preparation

Download the raw datasets using [`docs/DATASETS.md`](DATASETS.md), then run:

```bash
python scripts/data/convert_kvasir.py
python scripts/data/convert_cvc.py
python scripts/data/convert_etis.py
python scripts/data/prepare_rtdetr_data.py --dataset all --overwrite
```

## 3. Baseline YOLO Runs

This repository keeps the original helper scripts used during the project:

- `python scripts/train/run_all_missing_experiments.py`
- `python scripts/train/resume_phase3.py`

Public baseline results are summarized in:

- [`results/summary/complete_baseline_results.md`](../results/summary/complete_baseline_results.md)
- [`results/summary/experiment_results_phase1.json`](../results/summary/experiment_results_phase1.json)
- [`results/summary/experiment_results_phase2.json`](../results/summary/experiment_results_phase2.json)
- [`results/summary/experiment_results_final.json`](../results/summary/experiment_results_final.json)

## 4. RT-DETR

Train:

```bash
python scripts/train/train_rtdetr.py --prepare-data --epochs 100 --name main_rtdetr_100ep_server
```

Evaluate:

```bash
python scripts/eval/eval_rtdetr.py --weights runs/detect/main_rtdetr_100ep_server/weights/best.pt
```

One-command server workflow:

```bash
bash run_rtdetr_phase3.sh
```

## 5. RT-DETR + SAM

```bash
python scripts/eval/eval_rtdetr_sam.py
```

Outputs:

- Summary JSON: [`results/summary/rtdetr_sam_summary.json`](../results/summary/rtdetr_sam_summary.json)
- Qualitative masks: [`results/visuals/rtdetr_sam/`](../results/visuals/rtdetr_sam/)

## 6. TTN / TTT

YOLO segmentation:

```bash
python scripts/eval/eval_ttt_yolo.py
```

RT-DETR + SAM:

```bash
python scripts/eval/eval_ttt_rtdetr_sam.py
```

Outputs:

- [`results/summary/ttt_ttn_yolo_results.json`](../results/summary/ttt_ttn_yolo_results.json)
- [`results/summary/ttt_ttn_rtdetr_sam_results.json`](../results/summary/ttt_ttn_rtdetr_sam_results.json)
- [`results/visuals/ttn/`](../results/visuals/ttn/)
- [`results/visuals/ttt/`](../results/visuals/ttt/)

## 7. Figures

Rebuild the publication-style figures:

```bash
python scripts/analysis/visualize_baseline_results.py
```

Generated images are stored in [`results/figures/`](../results/figures/).
