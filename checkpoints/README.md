# Checkpoints

This directory is ignored by Git except for this README.

Place the required weights here with the exact filenames below:

- `checkpoints/yolov8n-seg.pt`
- `checkpoints/yolo11s-seg.pt`
- `checkpoints/rtdetr-l.pt`
- `checkpoints/sam_b.pt`

The scripts in this repository assume those filenames by default.

Recommended sources:

- YOLO / RT-DETR base checkpoints: Ultralytics model zoo via `ultralytics`
- SAM checkpoint: Meta Segment Anything model release

If you prefer different filenames, pass them explicitly on the command line.
