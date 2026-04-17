"""
TTT / TTN Test-Time Adaptation for YOLO Segmentation Models
===========================================================
TTN = Test-Time Normalization (update BN running stats with test data)
TTT = Test-Time Training (optimize BN affine params via entropy minimization)

Models : YOLOv8n-seg (100ep), YOLOv11s-seg (100ep)
Datasets: Kvasir-val, CVC-test, ETIS-test   (all trained on Kvasir)
"""

import os, json, copy
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    'YOLOv8n-seg':  'runs/segment/v8n_100ep/weights/best.pt',
    'YOLOv11s-seg': 'runs/segment/main_v11s_100ep/weights/best.pt',
}

DATASETS = {
    'kvasir': ('configs/kvasir_seg.yaml',   'val',  'datasets/kvasir_seg/images/val'),
    'cvc':    ('configs/cvc_clinicdb.yaml',  'test', 'datasets/cvc_clinicdb/images/test'),
    'etis':   ('configs/etis_larib.yaml',    'test', 'datasets/etis_larib/images/test'),
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

# ─── TTN: BN statistics adaptation ───────────────────────────────────
def apply_ttn(torch_model, images_tensor, device, batch_size=8):
    """
    TTN: keep the source-trained affine params (gamma/beta) intact,
    but update running_mean / running_var using target-domain data.
    We do NOT zero-initialise — instead, we use a momentum-based
    exponential moving average over multiple forward passes.
    """
    torch_model.eval()

    # Set BN to train mode so running stats update, use small momentum
    for m in torch_model.modules():
        if _is_bn(m):
            # partially adapt: blend source stats with target stats
            m.momentum = 0.1
            m.train()

    # Forward pass through target domain data (no gradient needed)
    with torch.no_grad():
        # Run multiple passes to stabilise stats
        for _ in range(3):
            for i in range(0, len(images_tensor), batch_size):
                batch = images_tensor[i:i+batch_size].to(device)
                torch_model(batch)

    torch_model.eval()

# ─── TTT: entropy minimization on BN affine params ───────────────────
def apply_ttt(torch_model, images_tensor, device,
               batch_size=4, lr=5e-4, steps=1):
    """Minimise prediction entropy by updating BN weight/bias."""
    torch_model.eval()

    # freeze everything
    for p in torch_model.parameters():
        p.requires_grad_(False)

    # unfreeze BN affine only, set BN to train
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
        indices = torch.randperm(len(images_tensor))
        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[indices[i:i+batch_size]].to(device)

            # Forward through the model
            outputs = torch_model(batch)

            # Handle YOLO output: could be tuple of tensors
            # Navigate to the first tensor that holds predictions
            preds = outputs
            while isinstance(preds, (tuple, list)):
                preds = preds[0]

            # preds shape: [B, 4+nc+mask_dim, num_anchors] for YOLOv8/v11
            if preds.ndim == 3:
                # class logit at index 4 (nc=1 for polyp)
                logits = preds[:, 4:5, :]
            elif preds.ndim == 2:
                logits = preds[:, 4:5]
            else:
                logits = preds

            probs = torch.sigmoid(logits)
            eps = 1e-7
            ent = -(probs * (probs + eps).log()
                    + (1 - probs) * (1 - probs + eps).log())
            loss = ent.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1

        print(f"    TTT step {step+1}/{steps}  avg_entropy={total_loss/max(n,1):.4f}")

    # Set model to eval for inference
    torch_model.eval()

# ─── Evaluation via ultralytics val() ─────────────────────────────────
def evaluate(model, data_yaml, split, project, name):
    r = model.val(
        data=data_yaml, split=split,
        imgsz=640, batch=16, workers=4,
        plots=True, project=str(project),
        name=name, exist_ok=True,
    )
    return {
        'precision':  float(r.seg.mp),
        'recall':     float(r.seg.mr),
        'map50':      float(r.seg.map50),
        'map50_95':   float(r.seg.map),
    }

# ─── Main ─────────────────────────────────────────────────────────────
def main():
    os.chdir(ROOT)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from ultralytics import YOLO

    ttn_project = ROOT / 'results' / 'visuals' / 'ttn'
    ttt_project = ROOT / 'results' / 'visuals' / 'ttt'
    summary_path = ROOT / 'results' / 'summary' / 'ttt_ttn_yolo_results.json'
    all_results = {}

    for model_name, wpath in MODELS.items():
        print(f"\n{'='*60}\n  Model: {model_name}\n{'='*60}")
        all_results[model_name] = {}

        for ds, (yaml, split, img_dir) in DATASETS.items():
            print(f"\n  ── Target domain: {ds.upper()} ──")
            imgs = load_images_tensor(img_dir)
            print(f"     Loaded {len(imgs)} images")

            # ── TTN ──
            print("     [TTN] adapting BN stats …")
            model = YOLO(wpath)
            model.model.to(device)
            apply_ttn(model.model, imgs, device)
            ttn = evaluate(model, yaml, split, ttn_project,
                           f"ttn_{model_name.replace('-','_')}_{ds}")
            print(f"     [TTN] mAP50={ttn['map50']*100:.2f}%  "
                  f"P={ttn['precision']*100:.2f}%  R={ttn['recall']*100:.2f}%")
            del model; torch.cuda.empty_cache()

            # ── TTT ──
            print("     [TTT] entropy minimisation …")
            model = YOLO(wpath)
            model.model.to(device)
            apply_ttt(model.model, imgs, device,
                      batch_size=4, lr=5e-4, steps=1)
            ttt = evaluate(model, yaml, split, ttt_project,
                           f"ttt_{model_name.replace('-','_')}_{ds}")
            print(f"     [TTT] mAP50={ttt['map50']*100:.2f}%  "
                  f"P={ttt['precision']*100:.2f}%  R={ttt['recall']*100:.2f}%")
            del model; torch.cuda.empty_cache()

            all_results[model_name][ds] = {'ttn': ttn, 'ttt': ttt}

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2), encoding='utf-8')
    print(f"\nSaved -> {summary_path}")

if __name__ == '__main__':
    main()
