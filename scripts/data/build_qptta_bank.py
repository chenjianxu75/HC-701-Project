"""
QP-TTA Source Prototype Bank Builder
=====================================
Runs pretrained RT-DETR on Kvasir-SEG training set, hooks into the last
decoder layer to capture queries / attention weights / contexts, then
filters by GT match quality and caches to disk.

Uses ultralytics predict API for box postprocessing (avoids manual
cxcywh→xyxy conversion), and matches predictions back to the 300 query
indices via exact score matching on the raw decoder output.

Output: results/artifacts/qptta/source_bank_kvasir_train.pt
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

# ─── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build QP-TTA source prototype bank.")
    p.add_argument("--weights", default="runs/detect/main_rtdetr_100ep_server3/weights/best.pt",
                   help="RT-DETR checkpoint")
    p.add_argument("--data",    default="configs/kvasir_det.yaml",
                   help="Dataset YAML (for locating images)")
    p.add_argument("--split",   default="train", help="Dataset split to use")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--conf-thres",  type=float, default=0.70)
    p.add_argument("--iou-thres",   type=float, default=0.75)
    p.add_argument("--max-per-image", type=int, default=3)
    p.add_argument("--output",  default="results/artifacts/qptta/source_bank_kvasir_train.pt")
    p.add_argument("--device",  default="")
    return p.parse_args()


# ─── GT loading ───────────────────────────────────────────────────────

def load_yolo_labels(label_dir: Path) -> dict[str, np.ndarray]:
    """Load YOLO-format labels. Returns {stem: array of [cls,cx,cy,w,h]}."""
    labels = {}
    if not label_dir.exists():
        return labels
    for f in sorted(label_dir.iterdir()):
        if f.suffix != ".txt":
            continue
        rows = []
        for line in f.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = [float(x) for x in parts[1:5]]
                rows.append([cls, cx, cy, w, h])
        labels[f.stem] = np.array(rows) if rows else np.zeros((0, 5))
    return labels


def yolo_to_xyxy(box_cxcywh: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert normalised YOLO [cls,cx,cy,w,h] to pixel [x1,y1,x2,y2]."""
    cx, cy, w, h = box_cxcywh[1], box_cxcywh[2], box_cxcywh[3], box_cxcywh[4]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.array([x1, y1, x2, y2])


def compute_iou_box(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ─── Decoder hook infrastructure ─────────────────────────────────────

class DecoderHookCapture:
    """
    Patches the last decoder layer's MSDeformAttn.forward to capture:
      - q_in:              [B, 300, 256]  (query input to cross-attn)
      - attention_weights: [B, 300, 8, 3, 4]
      - sampling_locations:[B, 300, 8, 3, 4, 2]
      - context:           [B, 300, 256]  (cross-attn output)

    Also patches RTDETRDecoder.forward to capture:
      - raw_output:        [B, 300, 4+nc] (full 300-query output for index matching)
    """

    def __init__(self, decoder_head):
        self.head = decoder_head
        self.decoder = decoder_head.decoder
        self.last_layer = self.decoder.layers[-1]
        self.cross_attn = self.last_layer.cross_attn
        self.captured: dict[str, torch.Tensor] = {}
        self._orig_cross_fwd = self.cross_attn.forward
        self._orig_head_fwd = self.head.forward
        self._patch()

    def _patch(self):
        ca = self.cross_attn
        cap = self.captured
        head = self.head
        orig_head_fwd = self._orig_head_fwd

        # ── Patch cross-attn to capture internals ──
        def hooked_cross_fwd(query, refer_bbox, value,
                             value_shapes, value_mask=None):
            bs, len_q = query.shape[:2]
            len_v = value.shape[1]

            cap["q_in"] = query.detach()

            vp = ca.value_proj(value)
            if value_mask is not None:
                vp = vp.masked_fill(value_mask[..., None], 0.0)
            vp = vp.view(bs, len_v, ca.n_heads, ca.d_model // ca.n_heads)

            so = ca.sampling_offsets(query).view(
                bs, len_q, ca.n_heads, ca.n_levels, ca.n_points, 2)
            aw = ca.attention_weights(query).view(
                bs, len_q, ca.n_heads, ca.n_levels * ca.n_points)
            aw = F.softmax(aw, -1).view(
                bs, len_q, ca.n_heads, ca.n_levels, ca.n_points)

            cap["attention_weights"] = aw.detach()

            np_ = refer_bbox.shape[-1]
            if np_ == 2:
                on = torch.as_tensor(
                    value_shapes, dtype=query.dtype,
                    device=query.device).flip(-1)
                sl = refer_bbox[:, :, None, :, None, :] + \
                     so / on[None, None, None, :, None, :]
            elif np_ == 4:
                sl = refer_bbox[:, :, None, :, None, :2] + \
                     so / ca.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            else:
                raise ValueError(f"ref dim must be 2 or 4, got {np_}")

            cap["sampling_locations"] = sl.detach()

            from ultralytics.nn.modules.utils import multi_scale_deformable_attn_pytorch
            out = multi_scale_deformable_attn_pytorch(vp, value_shapes, sl, aw)
            result = ca.output_proj(out)
            cap["context"] = result.detach()
            return result

        ca.forward = hooked_cross_fwd

        # ── Patch RTDETRDecoder head to capture raw output ──
        def hooked_head_fwd(x, batch=None):
            result = orig_head_fwd(x, batch)
            if not head.training and isinstance(result, tuple):
                y = result[0]  # [bs, 300, 4+nc]
                cap["raw_output"] = y.detach()
            return result

        head.forward = hooked_head_fwd

    def unpatch(self):
        self.cross_attn.forward = self._orig_cross_fwd
        self.head.forward = self._orig_head_fwd

    def get(self) -> dict[str, torch.Tensor]:
        return dict(self.captured)

    def clear(self):
        self.captured.clear()


# ─── Dataset resolution ──────────────────────────────────────────────

def resolve_dirs(data_yaml: str, split: str) -> tuple[Path, Path]:
    """Parse ultralytics YAML to find image and label directories."""
    import yaml
    yaml_path = ROOT / data_yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    ds_path = cfg.get("path", "")
    ds_root = Path(ds_path) if ds_path else yaml_path.parent
    if not ds_root.is_absolute():
        ds_root = ROOT / ds_root

    img_dir = ds_root / "images" / split
    lbl_dir = ds_root / "labels" / split

    if not img_dir.exists():
        alt = cfg.get(split, "")
        if alt:
            alt_path = Path(alt)
            if not alt_path.is_absolute():
                alt_path = ds_root / alt_path
            img_dir = alt_path
            lbl_dir = alt_path.parent.parent / "labels" / alt_path.name

    return img_dir, lbl_dir


# ─── Bank builder core ───────────────────────────────────────────────

def build_bank(args: argparse.Namespace) -> dict[str, Any]:
    """
    Build the source prototype bank.

    Strategy:
      1. Use ultralytics predict API for detection (correct postprocessing)
      2. Hook captures decoder internals (q_in, attn_weights, context) + raw output
      3. Match each filtered prediction back to its 300-query index via score matching
      4. Filter by confidence + box IoU against GT
      5. Store matched query triplets
    """
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(ROOT)

    from ultralytics import RTDETR

    print(f"[Bank] Loading RT-DETR from {args.weights}")
    model = RTDETR(args.weights)
    model.model.to(device).eval()

    head = model.model.model[-1]  # RTDETRDecoder
    hook = DecoderHookCapture(head)

    img_dir, lbl_dir = resolve_dirs(args.data, args.split)
    print(f"[Bank] Images: {img_dir}")
    print(f"[Bank] Labels: {lbl_dir}")

    gt_labels = load_yolo_labels(lbl_dir)
    image_files = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".png", ".jpeg", ".tif", ".bmp"}
    )
    print(f"[Bank] Found {len(image_files)} images, {len(gt_labels)} label files")

    # Accumulators
    all_queries, all_contexts = [], []
    all_attn_weights, all_sampling_locs = [], []
    all_scores, all_boxes = [], []
    all_image_ids, all_gt_ids = [], []

    for img_path in tqdm(image_files, desc="[Bank] Extracting"):
        stem = img_path.stem
        gt = gt_labels.get(stem)
        if gt is None or len(gt) == 0:
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h_orig, w_orig = img_bgr.shape[:2]

        # ── Run ultralytics predict (correct postprocessing) ──
        hook.clear()
        with torch.no_grad():
            results = model(img_bgr, imgsz=args.imgsz, verbose=False)

        det = results[0]
        pred_boxes_xyxy = det.boxes.xyxy.cpu().numpy()  # [N, 4] pixel coords
        pred_confs = det.boxes.conf.cpu().numpy()        # [N]

        if len(pred_boxes_xyxy) == 0:
            continue

        captured = hook.get()
        if "q_in" not in captured or "raw_output" not in captured:
            continue

        q_in   = captured["q_in"]               # [1, 300, 256]
        attn_w = captured["attention_weights"]   # [1, 300, 8, 3, 4]
        samp_l = captured["sampling_locations"]  # [1, 300, 8, 3, 4, 2]
        ctx    = captured["context"]             # [1, 300, 256]
        raw_y  = captured["raw_output"]          # [1, 300, 5]

        # All 300 query scores from raw output (last dim for nc=1)
        raw_scores = raw_y[0, :, -1].cpu()  # [300]

        # ── Match predictions to queries, then to GT ──
        matched_q = set()
        gt_matched = [False] * len(gt)
        kept = 0

        # Process predictions in confidence order (already sorted by ultralytics)
        for pred_idx in range(len(pred_boxes_xyxy)):
            conf = float(pred_confs[pred_idx])
            if conf < args.conf_thres:
                continue

            # ── Find query index via exact score matching ──
            diff = (raw_scores - conf).abs()
            # Mask already-matched queries
            for mq in matched_q:
                diff[mq] = float("inf")
            q_idx = diff.argmin().item()
            if diff[q_idx] > 1e-3:
                continue  # no reliable match
            matched_q.add(q_idx)

            # ── Match prediction box against GT (both in original pixel coords) ──
            pred_box = pred_boxes_xyxy[pred_idx]
            best_iou, best_gt = 0.0, -1

            for gi in range(len(gt)):
                if gt_matched[gi]:
                    continue
                gt_box = yolo_to_xyxy(gt[gi], w_orig, h_orig)
                iou = compute_iou_box(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gi

            if best_iou >= args.iou_thres and best_gt >= 0:
                gt_matched[best_gt] = True

                # Store this query's decoder internals (exact index)
                all_queries.append(q_in[0, q_idx].half().cpu())
                all_contexts.append(ctx[0, q_idx].half().cpu())
                all_attn_weights.append(attn_w[0, q_idx].half().cpu())
                all_sampling_locs.append(samp_l[0, q_idx].half().cpu())
                all_scores.append(conf)
                all_boxes.append(pred_box.tolist())
                all_image_ids.append(stem)
                all_gt_ids.append(int(best_gt))
                kept += 1

                if kept >= args.max_per_image:
                    break

    # ── Assemble bank ──
    bank = {
        "queries":            torch.stack(all_queries) if all_queries else torch.zeros(0, 256),
        "contexts":           torch.stack(all_contexts) if all_contexts else torch.zeros(0, 256),
        "attn_weights":       (torch.stack(all_attn_weights)
                               if all_attn_weights else torch.zeros(0, 8, 3, 4)),
        "sampling_locations": (torch.stack(all_sampling_locs)
                               if all_sampling_locs else torch.zeros(0, 8, 3, 4, 2)),
        "scores":             torch.tensor(all_scores, dtype=torch.float16),
        "boxes":              all_boxes,
        "image_ids":          all_image_ids,
        "gt_ids":             all_gt_ids,
        "conf_thres":         args.conf_thres,
        "iou_thres":          args.iou_thres,
    }

    hook.unpatch()
    return bank


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.chdir(ROOT)

    print("=" * 60)
    print("  QP-TTA Source Prototype Bank Builder")
    print("=" * 60)

    bank = build_bank(args)
    n_proto = len(bank["scores"])
    print(f"\n[Bank] Collected {n_proto} prototypes "
          f"(conf>={args.conf_thres}, IoU>={args.iou_thres})")

    # Fallback if too few prototypes
    if n_proto < 1000:
        print(f"[Bank] ⚠ Only {n_proto} prototypes "
              "— falling back to relaxed thresholds")
        args_relaxed = copy.deepcopy(args)
        args_relaxed.conf_thres = 0.60
        args_relaxed.iou_thres = 0.60
        bank = build_bank(args_relaxed)
        n_proto = len(bank["scores"])
        print(f"[Bank] After fallback: {n_proto} prototypes")

    # Save
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, str(out_path))
    print(f"\n[Bank] ✅ Saved → {out_path}  "
          f"({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"[Bank] Keys:          {list(bank.keys())}")
    print(f"[Bank] queries shape: {bank['queries'].shape}")
    print(f"[Bank] unique images: {len(set(bank['image_ids']))}")
    if n_proto > 0:
        print(f"[Bank] score range:   "
              f"[{bank['scores'].min():.3f}, {bank['scores'].max():.3f}]")


if __name__ == "__main__":
    main()
