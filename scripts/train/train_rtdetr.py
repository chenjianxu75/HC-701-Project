from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR on the prepared detection dataset.")
    parser.add_argument("--model", default="checkpoints/rtdetr-l.pt", help="Base RT-DETR checkpoint.")
    parser.add_argument("--data", default="configs/kvasir_det.yaml", help="Training data yaml.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="", help="CUDA device id(s), for example 0 or 0,1.")
    parser.add_argument("--name", default="main_rtdetr_100ep_server")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-data", action="store_true", help="Regenerate *_det datasets before training.")
    parser.add_argument("--overwrite-data", action="store_true", help="Delete existing *_det datasets before regeneration.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable AMP (off by default — RT-DETR + AMP causes NaN in validation).")
    return parser.parse_args()


def ensure_runtime() -> None:
    try:
        import ultralytics  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing runtime dependency 'ultralytics'. Run scripts/bootstrap_rtdetr_env.sh on the server first."
        ) from exc


def maybe_prepare_data(args: argparse.Namespace) -> None:
    if not args.prepare_data:
        return
    from scripts.data.prepare_rtdetr_data import main as prepare_main
    import sys

    argv = ["prepare_rtdetr_data.py", "--dataset", "all"]
    if args.overwrite_data:
        argv.append("--overwrite")
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        prepare_main()
    finally:
        sys.argv = old_argv


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    ensure_runtime()
    maybe_prepare_data(args)

    from ultralytics import YOLO

    model = YOLO(args.model)
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = ROOT / project_path

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "name": args.name,
        "project": str(project_path),
        "exist_ok": args.exist_ok,
        "patience": args.patience,
        "seed": args.seed,
        "deterministic": True,
        "pretrained": True,
        "plots": True,
        "val": True,
        "split": "val",
        "amp": args.amp,
    }
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", project_path / args.name))
    best_weight = save_dir / "weights" / "best.pt"
    last_weight = save_dir / "weights" / "last.pt"
    print(f"[TRAIN] save_dir={save_dir}")
    print(f"[TRAIN] best={best_weight}")
    print(f"[TRAIN] last={last_weight}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
