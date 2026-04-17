from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALS = (
    ("kvasir_val", "configs/kvasir_det.yaml", "val"),
    ("cvc_test", "configs/cvc_clinicdb_det.yaml", "test"),
    ("etis_test", "configs/etis_larib_det.yaml", "test"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR on Kvasir/CVC/ETIS detection splits.")
    parser.add_argument("--weights", required=True, help="Path to RT-DETR checkpoint.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="", help="CUDA device id(s).")
    parser.add_argument("--prepare-data", action="store_true", help="Regenerate *_det datasets before evaluation.")
    parser.add_argument("--overwrite-data", action="store_true", help="Delete existing *_det datasets before regeneration.")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="rtdetr_eval")
    parser.add_argument("--output", default="results/summary/rtdetr_eval_summary.json")
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


def extract_box_metrics(metrics) -> dict[str, float]:
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    ensure_runtime()
    maybe_prepare_data(args)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    summary: dict[str, dict[str, float | str]] = {}
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = ROOT / project_path

    for dataset_name, data_yaml, split in DEFAULT_EVALS:
        val_kwargs = {
            "data": data_yaml,
            "split": split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "plots": True,
            "save_json": True,
            "project": str(project_path),
            "name": f"{args.name}_{dataset_name}",
            "exist_ok": True,
        }
        if args.device:
            val_kwargs["device"] = args.device

        metrics = model.val(**val_kwargs)
        payload = extract_box_metrics(metrics)
        payload["data"] = data_yaml
        payload["split"] = split
        summary[dataset_name] = payload
        print(
            f"[EVAL] {dataset_name}: "
            f"precision={payload['precision']:.4f} "
            f"recall={payload['recall']:.4f} "
            f"mAP50={payload['map50']:.4f} "
            f"mAP50-95={payload['map50_95']:.4f}"
        )

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[EVAL] summary={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
