# Experiment Index

This document maps each experiment line to its code and stored artifacts.

## 1. YOLO Segmentation Baselines

Models:

- `YOLOv8n-seg`
- `YOLOv11s-seg`

Primary helper scripts:

- [`scripts/train/run_all_missing_experiments.py`](../scripts/train/run_all_missing_experiments.py)
- [`scripts/train/resume_phase3.py`](../scripts/train/resume_phase3.py)

Public result files:

- [`results/summary/complete_baseline_results.md`](../results/summary/complete_baseline_results.md)
- [`results/summary/experiment_results_phase1.json`](../results/summary/experiment_results_phase1.json)
- [`results/summary/experiment_results_phase2.json`](../results/summary/experiment_results_phase2.json)
- [`results/summary/experiment_results_final.json`](../results/summary/experiment_results_final.json)
- [`results/figures/`](../results/figures/)

## 2. RT-DETR-L

Training and evaluation:

- [`scripts/train/train_rtdetr.py`](../scripts/train/train_rtdetr.py)
- [`scripts/eval/eval_rtdetr.py`](../scripts/eval/eval_rtdetr.py)
- [`scripts/train/run_rtdetr_phase3.sh`](../scripts/train/run_rtdetr_phase3.sh)

Public result files:

- [`results/summary/rtdetr_eval_summary.json`](../results/summary/rtdetr_eval_summary.json)
- [`results/summary/rtdetr_eval_summary_100ep.json`](../results/summary/rtdetr_eval_summary_100ep.json)

## 3. RT-DETR-L + SAM

Evaluation:

- [`scripts/eval/eval_rtdetr_sam.py`](../scripts/eval/eval_rtdetr_sam.py)

Stored artifacts:

- [`results/summary/rtdetr_sam_summary.json`](../results/summary/rtdetr_sam_summary.json)
- [`results/visuals/rtdetr_sam/`](../results/visuals/rtdetr_sam/)

## 4. TTN / TTT On YOLO

Evaluation:

- [`scripts/eval/eval_ttt_yolo.py`](../scripts/eval/eval_ttt_yolo.py)

Stored artifacts:

- [`results/summary/ttt_ttn_yolo_results.json`](../results/summary/ttt_ttn_yolo_results.json)
- [`results/visuals/ttn/`](../results/visuals/ttn/)
- [`results/visuals/ttt/`](../results/visuals/ttt/)

## 5. TTN / TTT On RT-DETR + SAM

Evaluation:

- [`scripts/eval/eval_ttt_rtdetr_sam.py`](../scripts/eval/eval_ttt_rtdetr_sam.py)

Stored artifacts:

- [`results/summary/ttt_ttn_rtdetr_sam_results.json`](../results/summary/ttt_ttn_rtdetr_sam_results.json)
- [`results/visuals/ttn/ttn_rtdetr_sam_cvc/`](../results/visuals/ttn/ttn_rtdetr_sam_cvc/)
- [`results/visuals/ttt/ttt_rtdetr_sam_cvc/`](../results/visuals/ttt/ttt_rtdetr_sam_cvc/)

## 6. Curated Reporting Layer

The repository separates the curated report table from the runnable code and visualization folders:

- Curated report table: [`results/summary/complete_baseline_results.md`](../results/summary/complete_baseline_results.md)
- Aligned metric JSONs: [`results/summary/`](../results/summary/)
- Figures: [`results/figures/`](../results/figures/)

This keeps the reporting layer readable without mixing it into the raw `runs/` tree.
