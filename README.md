# HC701 Cross-Domain Polyp Experiments

This repository packages the code, curated summaries, figures, logs, and qualitative outputs for the HC701 polyp detection and segmentation experiments.

The project covers four main experiment lines:

- `YOLOv8n-seg` and `YOLOv11s-seg` baseline training on Kvasir-SEG
- `RT-DETR-L` detection-based cross-domain evaluation
- `RT-DETR-L + SAM` two-stage box-to-mask evaluation
- `TTN` / `TTT` test-time adaptation on both YOLO segmentation and RT-DETR+SAM

## Result Notes

- [`results/summary/complete_baseline_results.md`](results/summary/complete_baseline_results.md) is the curated report table used for project reporting.
- Public result files are organized under [`results/summary/`](results/summary/), [`results/figures/`](results/figures/), [`results/visuals/`](results/visuals/), and [`results/logs/`](results/logs/).
- Large datasets and checkpoints are intentionally not committed. Download instructions live in [`docs/DATASETS.md`](docs/DATASETS.md) and [`checkpoints/README.md`](checkpoints/README.md).

## Reported Summary Table

The table below mirrors the curated summary file and is included here only as the repository's reported comparison table.

| Model | Kvasir mAP50 | CVC mAP50 | ETIS mAP50 |
| :--- | :---: | :---: | :---: |
| YOLOv8n-seg (100 ep) | 95.4% | 43.3% | 39.5% |
| YOLOv11s-seg (100 ep) | 95.8% | 82.4% | 77.8% |
| RT-DETR-L (100 ep) | 96.3% | 87.7% | 87.9% |
| RT-DETR-L + SAM (100 ep) | 95.9% | 84.6% | 82.7% |
| YOLOv8n-seg + TTN | 95.6% | 72.5% | 67.8% |
| YOLOv8n-seg + TTT | 95.9% | 80.7% | 76.3% |
| YOLOv11s-seg + TTN | 95.9% | 85.7% | 81.2% |
| YOLOv11s-seg + TTT | 96.1% | 91.6% | 87.1% |
| RT-DETR-L + SAM + TTN | 96.1% | 88.2% | 86.1% |
| RT-DETR-L + SAM + TTT | 96.3% | 93.4% | 91.8% |

## Repository Layout

```text
.
├─ configs/                  Dataset YAML files
├─ scripts/
│  ├─ data/                  Dataset conversion and RT-DETR data prep
│  ├─ train/                 Training entry points and server bootstrap
│  ├─ eval/                  Evaluation, TTN, and TTT scripts
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
