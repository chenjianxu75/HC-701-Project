# HC701 Cross-Domain Polyp Experiments

This repository packages the code, curated summaries, figures, logs, and qualitative outputs for the HC701 polyp detection and segmentation experiments.

The project covers five main experiment lines:

- `YOLOv8n-seg` and `YOLOv11s-seg` baseline training on Kvasir-SEG
- `RT-DETR-L` detection-based cross-domain evaluation
- `RT-DETR-L + SAM` two-stage box-to-mask evaluation
- `TTN` / `TTT` test-time adaptation on both YOLO segmentation and RT-DETR+SAM
- `QP-TTA` query-prototype adaptation for RT-DETR and RT-DETR+SAM

## How To Read The Tables

- Unless noted otherwise, the main benchmark tables use `Kvasir-SEG` as the training source domain.
- `Kvasir mAP50` is the in-domain result. `CVC mAP50` and `ETIS mAP50` are cross-domain results on different target datasets, so lower values there reflect domain shift rather than training failure.
- `YOLOv8n-seg` and `YOLOv11s-seg` report mask `mAP50`.
- `RT-DETR-L` reports box `mAP50`.
- `RT-DETR-L + SAM` converts RT-DETR boxes into masks so it can be compared against the segmentation models on a fairer output format.
- `TTN` and `TTT` are test-time adaptation methods. They are not new training runs from scratch.

## Main Cross-Domain Results

The table below is the main reported comparison for models trained on `Kvasir-SEG`.

| Experiment Line | Kvasir mAP50 (source-domain) | CVC mAP50 (cross-domain) | ETIS mAP50 (cross-domain) |
| :--- | :---: | :---: | :---: |
| YOLOv8n-seg (100 ep) | 95.4% | 43.3% | 39.5% |
| YOLOv11s-seg (100 ep) | 95.8% | 82.4% | 77.8% |
| RT-DETR-L (100 ep) | **96.3%** | 87.7% | 87.9% |
| RT-DETR-L + SAM (100 ep) | 95.9% | 84.6% | 82.7% |
| YOLOv8n-seg + TTN | 95.6% | 72.5% | 67.8% |
| YOLOv8n-seg + TTT | 95.9% | 80.7% | 76.3% |
| YOLOv11s-seg + TTN | 95.9% | 85.7% | 81.2% |
| YOLOv11s-seg + TTT | 96.1% | 91.6% | 87.1% |
| RT-DETR-L + SAM + TTN | 96.1% | 88.2% | 86.1% |
| RT-DETR-L + SAM + TTT | **96.3%** | **93.4%** | **91.8%** |

What this table shows:
- `YOLOv8n-seg` fits the source domain well, but its cross-domain generalization drops sharply.
- `YOLOv11s-seg` is much stronger than `YOLOv8n-seg` under domain shift.
- `RT-DETR-L` is the strongest raw cross-domain detector in the baseline comparison.
- `RT-DETR-L + SAM` sacrifices a small amount of mAP50 to produce mask outputs instead of only boxes.
- `TTT` brings the largest recovery on cross-domain performance, especially for `YOLOv8n-seg`.

## QP-TTA On RT-DETR

This repository also includes a DETR-specific test-time adaptation route based on query prototypes and decoder attention alignment.

Core entry points:
- [`scripts/data/build_qptta_bank.py`](scripts/data/build_qptta_bank.py)
- [`scripts/eval/eval_qptta_rtdetr.py`](scripts/eval/eval_qptta_rtdetr.py)
- [`scripts/eval/eval_qptta_rtdetr_sam.py`](scripts/eval/eval_qptta_rtdetr_sam.py)

What this route is used for:
- build a source prototype bank from `Kvasir-SEG`
- adapt `RT-DETR-L` on target-domain images at test time
- evaluate both raw detector boxes and the final `RT-DETR-L + SAM` mask pipeline

The corresponding comparison block is included in:
- [`results/summary/complete_baseline_results.md`](results/summary/complete_baseline_results.md)

## Ablation: Few-Shot Finetuning vs TTN/TTT

This ablation asks a practical deployment question: is it better to use a few labeled target-domain samples for supervised finetuning, or to use label-free test-time adaptation?

Notation:
- `Baseline`: Kvasir-trained checkpoint with no target adaptation.
- `TTN`: test-time normalization using target-domain test data, no labels, no retraining.
- `TTT`: test-time training on target-domain test data, no labels, no full finetuning.
- `Exp A`: finetune with `50` labeled CVC images for `20` epochs.
- `Exp B`: continue from `Exp A`, then finetune with `20` labeled ETIS images for another `20` epochs.

### CVC Performance

| Model | Baseline | +TTN | +TTT | Exp A | Exp B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| YOLOv8n-seg | 43.3% | 72.5% | 80.7% | 68.5% | 82.3% |
| YOLOv11s-seg | 82.4% | 85.7% | 91.6% | 87.3% | 92.4% |
| RT-DETR-L + SAM | 84.6% | 88.2% | 93.4% | 89.5% | 94.1% |

### ETIS Performance

| Model | Baseline | +TTN | +TTT | Exp A | Exp B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| YOLOv8n-seg | 39.5% | 67.8% | 76.3% | 67.2% | 79.8% |
| YOLOv11s-seg | 77.8% | 81.2% | 87.1% | 80.5% | 88.6% |
| RT-DETR-L + SAM | 82.7% | 86.1% | 91.8% | 85.3% | 92.5% |

What this ablation shows:
- `TTT` beats single-domain few-shot finetuning (`Exp A`) on every model and on both target datasets.
- `Exp B` is only slightly higher than `TTT`, but it requires labeled target data and extra supervised training.
- The practical implication is that `TTT` gets close to supervised target-domain adaptation while avoiding new annotation cost.

## Result Notes

- [`results/summary/complete_baseline_results.md`](results/summary/complete_baseline_results.md) is the curated report table used for project reporting.
- Public result files are organized under [`results/summary/`](results/summary/), [`results/figures/`](results/figures/), [`results/visuals/`](results/visuals/), and [`results/logs/`](results/logs/).
- Large datasets and checkpoints are intentionally not committed. Download instructions live in [`docs/DATASETS.md`](docs/DATASETS.md) and [`checkpoints/README.md`](checkpoints/README.md).

## Repository Layout

```text
.
├─ configs/                  Dataset YAML files
├─ scripts/
│  ├─ data/                  Dataset conversion and RT-DETR data prep
│  ├─ train/                 Training entry points and server bootstrap
│  ├─ eval/                  Evaluation, TTN, TTT, and QP-TTA scripts
│  └─ analysis/              Figure generation and analysis helpers
├─ results/
│  ├─ summary/               Curated summaries and aligned metric JSON files
│  ├─ figures/               Publication-style figures
│  ├─ visuals/               Qualitative outputs for RT-DETR+SAM, TTN, and TTT
│  └─ logs/                  Selected logs
├─ docs/                     Dataset, reproduction, and experiment notes
├─ datasets/README.md        Dataset download instructions
└─ checkpoints/README.md     Checkpoint placement instructions
```

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download datasets and checkpoints using:

- [`docs/DATASETS.md`](docs/DATASETS.md)
- [`checkpoints/README.md`](checkpoints/README.md)

3. Run the main experiment entry points from the repository root:

```bash
python scripts/data/prepare_rtdetr_data.py --dataset all --overwrite
python scripts/train/train_rtdetr.py --prepare-data --epochs 100 --name main_rtdetr_100ep_server
python scripts/eval/eval_rtdetr.py --weights runs/detect/main_rtdetr_100ep_server/weights/best.pt
python scripts/eval/eval_rtdetr_sam.py
python scripts/eval/eval_ttt_yolo.py
python scripts/eval/eval_ttt_rtdetr_sam.py
python scripts/data/build_qptta_bank.py
python scripts/eval/eval_qptta_rtdetr.py --datasets cvc etis
python scripts/eval/eval_qptta_rtdetr_sam.py --datasets cvc etis
```

4. Rebuild the figures if needed:

```bash
python scripts/analysis/visualize_baseline_results.py
```

## Where To Look First

- Report table: [`results/summary/complete_baseline_results.md`](results/summary/complete_baseline_results.md)
- Metric JSON files: [`results/summary/`](results/summary/)
- Figures: [`results/figures/`](results/figures/)
- Qualitative outputs: [`results/visuals/`](results/visuals/)
- Reproduction guide: [`docs/REPRODUCE.md`](docs/REPRODUCE.md)
