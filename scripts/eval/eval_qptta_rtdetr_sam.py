"""
QP-TTA + SAM Final Evaluation for RT-DETR+SAM Pipeline
=======================================================
Runs QP-TTA on RT-DETR, then feeds adapted detections to SAM for mask output.

Execution per target domain:
  1. Load fresh RT-DETR from source checkpoint
  2. Load source prototype bank
  3. QP-TTA adaptation on target domain
  4. Box evaluation via ultralytics model.val() (same as eval_rtdetr.py)
  5. Feed adapted boxes -> SAM -> mask evaluation

Output:
  results/summary/qptta_rtdetr_box_results.json
  results/summary/qptta_rtdetr_sam_results.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

RTDETR_WEIGHTS = "runs/detect/main_rtdetr_100ep_server3/weights/best.pt"
BANK_PATH = "results/artifacts/qptta/source_bank_kvasir_train.pt"
SAM_CHECKPOINT = "checkpoints/sam_b.pt"

DATASETS = {
    "kvasir": {
        "img_dir": "datasets/kvasir_seg/images/val",
        "gt_dir":  "kvasir-seg/masks",
    },
    "cvc": {
        "img_dir": "datasets/cvc_clinicdb/images/test",
        "gt_dir":  "archive/TIF/Ground Truth",
    },
    "etis": {
        "img_dir": "datasets/etis_larib/images/test",
        "gt_dir":  "datasets/ETIS-Larib/masks",
    },
}

# Detection YAML configs — same as eval_rtdetr.py
EVAL_CONFIGS = {
    "kvasir": ("configs/kvasir_det.yaml",       "val"),
    "cvc":    ("configs/cvc_clinicdb_det.yaml",  "test"),
    "etis":   ("configs/etis_larib_det.yaml",    "test"),
}


# ─── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QP-TTA + SAM evaluation for RT-DETR+SAM pipeline.")
    p.add_argument("--weights",    default=RTDETR_WEIGHTS)
    p.add_argument("--bank",       default=BANK_PATH)
    p.add_argument("--sam-checkpoint", default=SAM_CHECKPOINT)
    p.add_argument("--datasets",   nargs="+", default=["cvc", "etis"],
                   choices=list(DATASETS.keys()))
    p.add_argument("--batch",      type=int, default=8)
    p.add_argument("--adapt-steps", type=int, default=3)
    p.add_argument("--lr",         type=float, default=1e-5)
    p.add_argument("--query-lr",   type=float, default=5e-3)
    p.add_argument("--k",          type=int, default=8)
    p.add_argument("--n-neg",      type=int, default=32)
    p.add_argument("--target-conf-thres", type=float, default=0.50)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--imgsz",      type=int, default=640)
    p.add_argument("--device",     default="")
    p.add_argument("--box-output", default="results/summary/qptta_rtdetr_box_results.json")
    p.add_argument("--mask-output", default="results/summary/qptta_rtdetr_sam_results.json")
    p.add_argument("--vis-dir",    default="results/visuals/qptta")
    p.add_argument("--log-dir",    default="results/logs")
    return p.parse_args()


# ─── Mask metric helpers (from eval_ttt_rtdetr_sam.py) ────────────────

def calculate_iou(mask1, mask2):
    m1, m2 = np.squeeze(mask1), np.squeeze(mask2)
    if m1.ndim != 2 or m2.ndim != 2:
        return 0.0
    if m1.shape != m2.shape:
        m1 = cv2.resize(m1.astype(np.uint8),
                        (m2.shape[1], m2.shape[0]),
                        interpolation=cv2.INTER_NEAREST) > 0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 0.0


def load_gt_masks(gt_dir, image_names):
    gts, total = {}, 0
    gt_dir = Path(gt_dir)
    for name in image_names:
        base = os.path.splitext(name)[0]
        mask_path = None
        for ext in [".png", ".jpg", ".tif", ".bmp", ".jpeg"]:
            p = gt_dir / (base + ext)
            if p.exists():
                mask_path = p
                break
        masks_list = []
        if mask_path:
            g = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                gb = g > 127
                contours, _ = cv2.findContours(
                    (gb * 255).astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) > 10:
                        m = np.zeros_like(gb, dtype=np.uint8)
                        cv2.drawContours(m, [c], -1, 1, -1)
                        masks_list.append(m > 0)
        gts[name] = {"masks": masks_list, "matched": [False] * len(masks_list)}
        total += len(masks_list)
    return gts, total


def compute_mask_map(predictions, gts, total_gts):
    """Compute mask mAP@0.5 and mAP@0.5:0.95."""
    predictions.sort(key=lambda x: x["score"], reverse=True)
    map_scores = []
    p50, r50 = 0.0, 0.0

    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        for v in gts.values():
            v["matched"] = [False] * len(v["masks"])
        tps = np.zeros(len(predictions))
        fps = np.zeros(len(predictions))

        for i, pred in enumerate(predictions):
            gt_data = gts.get(pred["img_name"], {"masks": [], "matched": []})
            best_iou, best_j = 0.0, -1
            pm = pred["mask"]
            for j, gm in enumerate(gt_data["masks"]):
                if pm.shape != gm.shape:
                    pm2 = cv2.resize(pm.astype(np.uint8),
                                     (gm.shape[1], gm.shape[0]),
                                     interpolation=cv2.INTER_NEAREST) > 0
                else:
                    pm2 = pm
                iou = calculate_iou(pm2, gm)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_thresh and best_j >= 0:
                if not gt_data["matched"][best_j]:
                    tps[i] = 1
                    gt_data["matched"][best_j] = True
                else:
                    fps[i] = 1
            else:
                fps[i] = 1

        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        rec = tp_cum / total_gts if total_gts > 0 else np.zeros_like(tp_cum)
        pre = tp_cum / (tp_cum + fp_cum + 1e-16)

        if iou_thresh == 0.5 and len(pre) > 0:
            f1 = 2 * (pre * rec) / (pre + rec + 1e-16)
            idx = np.argmax(f1)
            p50, r50 = float(pre[idx]), float(rec[idx])

        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([1.0], pre, [0.0]))
        for ii in range(mpre.size - 1, 0, -1):
            mpre[ii - 1] = max(mpre[ii - 1], mpre[ii])
        idx2 = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx2 + 1] - mrec[idx2]) * mpre[idx2 + 1])
        map_scores.append(ap)

    return {
        "precision": p50,
        "recall": r50,
        "map50": float(map_scores[0]) if map_scores else 0.0,
        "map50_95": float(np.mean(map_scores)) if map_scores else 0.0,
    }


# ─── Box evaluation (identical to eval_rtdetr.py baseline) ────────────

def evaluate_box(model, data_yaml: str, split: str,
                 project: str, name: str) -> dict[str, float]:
    """Box mAP via ultralytics val() — same as baseline."""
    r = model.val(
        data=data_yaml,
        split=split,
        imgsz=640,
        batch=16,
        workers=0,
        plots=True,
        save_json=True,
        project=project,
        name=name,
        exist_ok=True,
    )
    return {
        "precision":  float(r.box.mp),
        "recall":     float(r.box.mr),
        "map50":      float(r.box.map50),
        "map50_95":   float(r.box.map),
    }


# ─── Two-stage: adapted RT-DETR → SAM ───────────────────────────────

def run_rtdetr_sam_eval(det_model, sam_model, img_dir, gt_dir, out_dir,
                        imgsz=640):
    """Run detection with adapted RT-DETR, then segment with SAM."""
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".png", ".tif", ".jpeg", ".bmp"}
    )
    gts, total_gts = load_gt_masks(gt_dir, [f.name for f in image_files])
    predictions = []
    save_viz = 3

    for img_path in tqdm(image_files, desc="  RT-DETR+SAM Inference"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        det = det_model(img, imgsz=imgsz, verbose=False)[0]
        boxes = det.boxes.xyxy.cpu().numpy()
        confs = det.boxes.conf.cpu().numpy()

        if len(boxes) > 0:
            sam_res = sam_model(img, bboxes=boxes.tolist(), verbose=False)[0]
            if sam_res.masks is not None:
                masks_raw = sam_res.masks.data.cpu().numpy()
                n = min(len(boxes), masks_raw.shape[0])
                for k in range(n):
                    pm = np.squeeze(masks_raw[k])
                    if pm.ndim != 2:
                        continue
                    predictions.append({
                        "img_name": img_path.name,
                        "score": float(confs[k]),
                        "mask": pm > 0.5,
                    })

                if save_viz > 0:
                    viz = img.copy()
                    for k in range(n):
                        pm = np.squeeze(masks_raw[k])
                        if pm.ndim != 2:
                            continue
                        c = np.random.randint(0, 255, (3,), dtype=np.uint8)
                        if pm.shape[:2] != viz.shape[:2]:
                            pm = cv2.resize(
                                pm.astype(np.uint8),
                                (viz.shape[1], viz.shape[0]),
                                interpolation=cv2.INTER_NEAREST) > 0
                        viz[pm] = (viz[pm] * 0.5 + c * 0.5).astype(np.uint8)
                    cv2.imwrite(str(out_dir / f"{img_path.stem}_qptta_sam.jpg"),
                                viz)
                    save_viz -= 1

    mask_metrics = compute_mask_map(predictions, gts, total_gts)
    return mask_metrics


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.chdir(ROOT)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  QP-TTA + SAM — RT-DETR+SAM Pipeline Evaluation")
    print("=" * 60)

    # Load bank
    bank_path = ROOT / args.bank
    print(f"\n[QP-TTA+SAM] Loading bank from {bank_path}")
    bank = torch.load(str(bank_path), map_location="cpu", weights_only=False)
    print(f"[QP-TTA+SAM] Bank: {bank['queries'].shape[0]} prototypes")

    # Load SAM (shared across all datasets)
    from ultralytics import RTDETR, SAM
    sam_model = SAM(str(ROOT / args.sam_checkpoint))

    # Import QP-TTA engine and image loader from eval_qptta_rtdetr
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_qptta_rtdetr",
        str(ROOT / "scripts" / "eval" / "eval_qptta_rtdetr.py"))
    qptta_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qptta_mod)
    QPTTAEngine = qptta_mod.QPTTAEngine
    load_images_tensor = qptta_mod.load_images_tensor

    all_box_results = {}
    all_mask_results = {}

    for ds in args.datasets:
        cfg = DATASETS[ds]
        eval_yaml, eval_split = EVAL_CONFIGS[ds]

        print(f"\n{'=' * 60}")
        print(f"  Target domain: {ds.upper()}")
        print(f"{'=' * 60}")

        # ── Step 1: Fresh RT-DETR ──
        print(f"  Loading fresh RT-DETR from {args.weights}")
        det_model = RTDETR(args.weights)
        det_model.model.to(device)

        # ── Step 2–3: QP-TTA adaptation ──
        imgs = load_images_tensor(cfg["img_dir"], args.imgsz)
        print(f"  Loaded {len(imgs)} images for adaptation")

        engine = QPTTAEngine(det_model.model, bank, args)
        engine.adapt(imgs)

        # ── Step 4: Box evaluation via ultralytics val() ──
        print(f"  [BOX] Evaluating with {eval_yaml} ({eval_split}) ...")
        box_metrics = evaluate_box(
            det_model, eval_yaml, eval_split,
            project=str(ROOT / "runs" / "detect"),
            name=f"qptta_rtdetr_sam_box_{ds}",
        )
        print(f"  [BOX]  mAP50={box_metrics['map50'] * 100:.2f}%  "
              f"P={box_metrics['precision'] * 100:.2f}%  "
              f"R={box_metrics['recall'] * 100:.2f}%")
        all_box_results[ds] = {"qptta_box": box_metrics}

        # ── Step 5: RT-DETR+SAM mask evaluation ──
        vis_dir = ROOT / args.vis_dir / ds
        print(f"  [MASK] Running RT-DETR+SAM inference ...")
        mask_metrics = run_rtdetr_sam_eval(
            det_model, sam_model, cfg["img_dir"], cfg["gt_dir"],
            vis_dir, args.imgsz)
        print(f"  [MASK] mAP50={mask_metrics['map50'] * 100:.2f}%  "
              f"P={mask_metrics['precision'] * 100:.2f}%  "
              f"R={mask_metrics['recall'] * 100:.2f}%")
        all_mask_results[ds] = {"qptta_mask": mask_metrics}

        # Log
        log_dir = ROOT / args.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"qptta_{ds}.jsonl", "a") as f:
            f.write(json.dumps({
                "dataset": ds,
                "box_metrics": box_metrics,
                "mask_metrics": mask_metrics,
            }) + "\n")

        del det_model
        torch.cuda.empty_cache()

    # Save summaries
    box_out = ROOT / args.box_output
    box_out.parent.mkdir(parents=True, exist_ok=True)
    box_out.write_text(json.dumps(all_box_results, indent=2), encoding="utf-8")
    print(f"\n[QP-TTA+SAM] Box summary  → {box_out}")

    mask_out = ROOT / args.mask_output
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.write_text(json.dumps(all_mask_results, indent=2), encoding="utf-8")
    print(f"[QP-TTA+SAM] Mask summary → {mask_out}")


if __name__ == "__main__":
    main()
