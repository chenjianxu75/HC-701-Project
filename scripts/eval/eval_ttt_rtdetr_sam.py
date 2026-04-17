"""
TTT / TTN Test-Time Adaptation for RT-DETR-L + SAM Pipeline
===========================================================
Adapts RT-DETR BN layers at test time, then feeds adapted detections to SAM.

TTN = BN running stats adaptation
TTT = BN affine params entropy minimization
"""

import os, json, copy
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

RTDETR_WEIGHTS = 'runs/detect/main_rtdetr_100ep_server3/weights/best.pt'

DATASETS = {
    'kvasir': {
        'img_dir': 'datasets/kvasir_seg/images/val',
        'gt_dir':  'kvasir-seg/masks',
    },
    'cvc': {
        'img_dir': 'datasets/cvc_clinicdb/images/test',
        'gt_dir':  'archive/TIF/Ground Truth',
    },
    'etis': {
        'img_dir': 'datasets/etis_larib/images/test',
        'gt_dir':  'datasets/ETIS-Larib/masks',
    },
}

# ─── Image Loading ────────────────────────────────────────────────────
def load_images_tensor(img_dir, imgsz=640):
    img_dir = Path(img_dir)
    files = sorted(f for f in img_dir.iterdir()
                   if f.suffix.lower() in {'.jpg', '.png', '.tif', '.jpeg', '.bmp'})
    tensors = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.resize(img, (imgsz, imgsz))
        img = img[:, :, ::-1].copy()
        img = img.transpose(2, 0, 1)
        tensors.append(torch.from_numpy(img.astype(np.float32) / 255.0))
    return torch.stack(tensors)

# ─── BN helpers ───────────────────────────────────────────────────────
def _is_bn(m):
    return isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm))

# ─── TTN ──────────────────────────────────────────────────────────────
def apply_ttn(torch_model, images_tensor, device, batch_size=8):
    torch_model.eval()
    for m in torch_model.modules():
        if _is_bn(m):
            m.momentum = 0.1
            m.train()
    with torch.no_grad():
        for _ in range(3):  # multiple passes to stabilise
            for i in range(0, len(images_tensor), batch_size):
                batch = images_tensor[i:i+batch_size].to(device)
                torch_model(batch)
    torch_model.eval()

# ─── TTT ──────────────────────────────────────────────────────────────
def apply_ttt(torch_model, images_tensor, device,
               batch_size=4, lr=1e-3, steps=1):
    torch_model.eval()
    for p in torch_model.parameters():
        p.requires_grad_(False)

    bn_params = []
    for m in torch_model.modules():
        if _is_bn(m):
            m.train()
            if m.affine:
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
                bn_params.extend([m.weight, m.bias])

    if not bn_params:
        print("    ⚠ No BN affine params – skipping TTT")
        return

    optimizer = torch.optim.Adam(bn_params, lr=lr)
    for step in range(steps):
        total_loss, n = 0.0, 0
        idx = torch.randperm(len(images_tensor))
        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[idx[i:i+batch_size]].to(device)
            outputs = torch_model(batch)
            preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            if isinstance(preds, dict):
                # RT-DETR returns dict with 'pred_logits' and 'pred_boxes'
                logits = preds.get('pred_logits', preds.get('scores', None))
                if logits is None:
                    for v in preds.values():
                        if isinstance(v, torch.Tensor) and v.ndim >= 2:
                            logits = v; break
                if logits is not None:
                    probs = torch.sigmoid(logits)
                else:
                    continue
            elif preds.ndim == 3:
                logits = preds[:, 4:5, :]
                probs = torch.sigmoid(logits)
            else:
                probs = torch.sigmoid(preds)

            eps = 1e-7
            ent = -(probs * (probs + eps).log()
                    + (1 - probs) * (1 - probs + eps).log())
            loss = ent.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n += 1

        print(f"    TTT step {step+1}/{steps}  avg_entropy={total_loss/max(n,1):.4f}")
    torch_model.eval()

# ─── Mask metric helpers (reused from eval_rtdetr_sam.py) ─────────────
def calculate_iou(mask1, mask2):
    m1 = np.squeeze(mask1); m2 = np.squeeze(mask2)
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
        for ext in ['.png', '.jpg', '.tif', '.bmp', '.jpeg']:
            p = gt_dir / (base + ext)
            if p.exists():
                mask_path = p; break
        masks_list = []
        if mask_path:
            g = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                gb = g > 127
                contours, _ = cv2.findContours(
                    (gb*255).astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) > 10:
                        m = np.zeros_like(gb, dtype=np.uint8)
                        cv2.drawContours(m, [c], -1, 1, -1)
                        masks_list.append(m > 0)
        gts[name] = {'masks': masks_list, 'matched': [False]*len(masks_list)}
        total += len(masks_list)
    return gts, total

def compute_map(predictions, gts, total_gts):
    predictions.sort(key=lambda x: x['score'], reverse=True)
    map_scores = []
    p50, r50 = 0.0, 0.0
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        for v in gts.values():
            v['matched'] = [False]*len(v['masks'])
        tps = np.zeros(len(predictions))
        fps = np.zeros(len(predictions))
        for i, pred in enumerate(predictions):
            gt_data = gts.get(pred['img_name'], {'masks':[], 'matched':[]})
            best_iou, best_j = 0.0, -1
            pm = pred['mask']
            for j, gm in enumerate(gt_data['masks']):
                if pm.shape != gm.shape:
                    pm2 = cv2.resize(pm.astype(np.uint8),
                                     (gm.shape[1], gm.shape[0]),
                                     interpolation=cv2.INTER_NEAREST) > 0
                else:
                    pm2 = pm
                iou = calculate_iou(pm2, gm)
                if iou > best_iou:
                    best_iou = iou; best_j = j
            if best_iou >= iou_thresh and best_j >= 0:
                if not gt_data['matched'][best_j]:
                    tps[i] = 1; gt_data['matched'][best_j] = True
                else:
                    fps[i] = 1
            else:
                fps[i] = 1
        tp_cum = np.cumsum(tps); fp_cum = np.cumsum(fps)
        rec = tp_cum / total_gts if total_gts > 0 else np.zeros_like(tp_cum)
        pre = tp_cum / (tp_cum + fp_cum + 1e-16)
        if iou_thresh == 0.5 and len(pre) > 0:
            f1 = 2*(pre*rec)/(pre+rec+1e-16)
            idx = np.argmax(f1)
            p50, r50 = float(pre[idx]), float(rec[idx])
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([1.], pre, [0.]))
        for ii in range(mpre.size-1, 0, -1):
            mpre[ii-1] = max(mpre[ii-1], mpre[ii])
        idx2 = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx2+1]-mrec[idx2]) * mpre[idx2+1])
        map_scores.append(ap)
    return map_scores[0], np.mean(map_scores), p50, r50

# ─── Two-stage inference: RT-DETR → SAM ──────────────────────────────
def run_rtdetr_sam_eval(det_model, sam_model, img_dir, gt_dir, out_dir):
    img_dir = Path(img_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(f for f in img_dir.iterdir()
                         if f.suffix.lower() in {'.jpg','.png','.tif','.jpeg'})
    gts, total_gts = load_gt_masks(gt_dir, [f.name for f in image_files])
    predictions = []
    save_viz = 3

    for img_path in tqdm(image_files, desc="  Inference"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        det = det_model(img, verbose=False)[0]
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
                        'img_name': img_path.name,
                        'score': float(confs[k]),
                        'mask': pm > 0.5,
                    })
                if save_viz > 0:
                    viz = img.copy()
                    for k in range(n):
                        pm = np.squeeze(masks_raw[k])
                        if pm.ndim != 2: continue
                        c = np.random.randint(0,255,(3,),dtype=np.uint8)
                        if pm.shape[:2] != viz.shape[:2]:
                            pm = cv2.resize(pm.astype(np.uint8),(viz.shape[1],viz.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)>0
                        viz[pm] = (viz[pm]*0.5+c*0.5).astype(np.uint8)
                    cv2.imwrite(str(out_dir/f"{img_path.stem}_sam.jpg"), viz)
                    save_viz -= 1

    map50, map50_95, p50, r50 = compute_map(predictions, gts, total_gts)
    return {'precision': p50, 'recall': r50, 'map50': map50, 'map50_95': map50_95}

# ─── Main ─────────────────────────────────────────────────────────────
def main():
    os.chdir(ROOT)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from ultralytics import RTDETR, SAM

    sam_model = SAM(str(ROOT / 'checkpoints' / 'sam_b.pt'))
    ttn_project = ROOT / 'results' / 'visuals' / 'ttn'
    ttt_project = ROOT / 'results' / 'visuals' / 'ttt'
    summary_path = ROOT / 'results' / 'summary' / 'ttt_ttn_rtdetr_sam_results.json'
    all_results = {}

    for ds, cfg in DATASETS.items():
        print(f"\n{'='*60}\n  RT-DETR-L + SAM  →  {ds.upper()}\n{'='*60}")
        imgs = load_images_tensor(cfg['img_dir'])
        print(f"  Loaded {len(imgs)} images for adaptation")

        # ── TTN ──
        print("  [TTN] adapting BN stats …")
        det = RTDETR(RTDETR_WEIGHTS)
        det.model.to(device)
        apply_ttn(det.model, imgs, device)
        out_dir = ttn_project / f"ttn_rtdetr_sam_{ds}"
        ttn = run_rtdetr_sam_eval(det, sam_model, cfg['img_dir'], cfg['gt_dir'], out_dir)
        print(f"  [TTN] mAP50={ttn['map50']*100:.2f}%  P={ttn['precision']*100:.2f}%  R={ttn['recall']*100:.2f}%")
        del det; torch.cuda.empty_cache()

        # ── TTT ──
        print("  [TTT] entropy minimisation …")
        det = RTDETR(RTDETR_WEIGHTS)
        det.model.to(device)
        apply_ttt(det.model, imgs, device, batch_size=4, lr=1e-3, steps=1)
        out_dir = ttt_project / f"ttt_rtdetr_sam_{ds}"
        ttt = run_rtdetr_sam_eval(det, sam_model, cfg['img_dir'], cfg['gt_dir'], out_dir)
        print(f"  [TTT] mAP50={ttt['map50']*100:.2f}%  P={ttt['precision']*100:.2f}%  R={ttt['recall']*100:.2f}%")
        del det; torch.cuda.empty_cache()

        all_results[ds] = {'ttn': ttn, 'ttt': ttt}

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2), encoding='utf-8')
    print(f"\nSaved -> {summary_path}")

if __name__ == '__main__':
    main()
