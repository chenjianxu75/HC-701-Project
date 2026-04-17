from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    src_root: Path
    dst_root: Path
    splits: tuple[str, ...]


DATASETS = {
    "kvasir": DatasetSpec(
        name="kvasir",
        src_root=ROOT / "datasets" / "kvasir_seg",
        dst_root=ROOT / "datasets" / "kvasir_det",
        splits=("train", "val"),
    ),
    "cvc": DatasetSpec(
        name="cvc",
        src_root=ROOT / "datasets" / "cvc_clinicdb",
        dst_root=ROOT / "datasets" / "cvc_clinicdb_det",
        splits=("test",),
    ),
    "etis": DatasetSpec(
        name="etis",
        src_root=ROOT / "datasets" / "etis_larib",
        dst_root=ROOT / "datasets" / "etis_larib_det",
        splits=("test",),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert existing YOLO polygon labels into detection box labels for RT-DETR."
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "kvasir", "cvc", "etis"),
        default="all",
        help="Dataset to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing *_det outputs before regenerating them.",
    )
    return parser.parse_args()


def polygon_line_to_box(line: str) -> str | None:
    parts = line.strip().split()
    if not parts:
        return None

    cls_id = parts[0]
    coords = [float(value) for value in parts[1:]]
    if len(coords) == 4:
        return f"{cls_id} " + " ".join(f"{value:.6f}" for value in coords)
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None

    xs = coords[0::2]
    ys = coords[1::2]
    x_min = max(0.0, min(xs))
    x_max = min(1.0, max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(1.0, max(ys))
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def label_to_boxes(src_label: Path) -> list[str]:
    boxes: list[str] = []
    for line in src_label.read_text(encoding="utf-8").splitlines():
        box_line = polygon_line_to_box(line)
        if box_line is not None:
            boxes.append(box_line)
    return boxes


def find_matching_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def convert_split(spec: DatasetSpec, split: str) -> tuple[int, int]:
    src_image_dir = spec.src_root / "images" / split
    src_label_dir = spec.src_root / "labels" / split
    dst_image_dir = spec.dst_root / "images" / split
    dst_label_dir = spec.dst_root / "labels" / split

    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    for src_label in sorted(src_label_dir.glob("*.txt")):
        image_path = find_matching_image(src_image_dir, src_label.stem)
        if image_path is None:
            skipped += 1
            continue

        shutil.copy2(image_path, dst_image_dir / image_path.name)
        boxes = label_to_boxes(src_label)
        (dst_label_dir / src_label.name).write_text("\n".join(boxes), encoding="utf-8")
        converted += 1

    return converted, skipped


def prepare_dataset(spec: DatasetSpec, overwrite: bool) -> None:
    if not spec.src_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {spec.src_root}")

    if overwrite and spec.dst_root.exists():
        shutil.rmtree(spec.dst_root)

    print(f"[DATA] Preparing {spec.name}: {spec.src_root} -> {spec.dst_root}")
    total_converted = 0
    total_skipped = 0
    for split in spec.splits:
        converted, skipped = convert_split(spec, split)
        total_converted += converted
        total_skipped += skipped
        print(f"[DATA]   split={split} converted={converted} skipped={skipped}")
    print(f"[DATA] Done {spec.name}: converted={total_converted} skipped={total_skipped}")


def main() -> int:
    args = parse_args()
    selected = DATASETS.values() if args.dataset == "all" else (DATASETS[args.dataset],)
    for spec in selected:
        prepare_dataset(spec, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
