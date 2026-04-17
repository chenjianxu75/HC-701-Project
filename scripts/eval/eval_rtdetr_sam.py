import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import RTDETR, SAM
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

# Configuration for evaluation
EVAL_SETS = [
    {
        "name": "kvasir",
        "img_dir": "datasets/kvasir_seg/images/val",
        "gt_dir": "kvasir-seg/masks"
    },
    {
        "name": "cvc",
        "img_dir": "datasets/cvc_clinicdb/images/test",
        "gt_dir": "archive/TIF/Ground Truth"
    },
    {
        "name": "etis",
        "img_dir": "datasets/etis_larib/images/test",
        "gt_dir": "datasets/ETIS-Larib/masks"
    }
]

def calculate_iou(mask1, mask2):
    # Ensure both masks are 2D
    m1 = np.squeeze(mask1)
    m2 = np.squeeze(mask2)
    if m1.ndim != 2 or m2.ndim != 2:
        return 0.0
    # Resize if shapes differ
    if m1.shape != m2.shape:
        m1 = cv2.resize(m1.astype(np.uint8), (m2.shape[1], m2.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(intersection / union) if union > 0 else 0.0

def load_gt_masks(gt_dir, image_names):
    gts = {}
    total_gts = 0
    gt_dir = Path(gt_dir)
    
    for img_name in image_names:
        basename = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in ['.png', '.jpg', '.tif', '.bmp', '.jpeg']:
            p = gt_dir / (basename + ext)
            if p.exists():
                mask_path = p
                break
        
        if mask_path:
            gt_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_img is not None:
                gt_bin = gt_img > 127
                # Find contours to split disjoint masks as instances
                contours, _ = cv2.findContours((gt_bin*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                masks = []
                for c in contours:
                    if cv2.contourArea(c) > 10:
                        m = np.zeros_like(gt_bin, dtype=np.uint8)
                        cv2.drawContours(m, [c], -1, 1, -1)
                        masks.append(m > 0)
                gts[img_name] = {'masks': masks, 'matched': [False]*len(masks)}
                total_gts += len(masks)
            else:
                gts[img_name] = {'masks': [], 'matched': []}
        else:
            gts[img_name] = {'masks': [], 'matched': []}
            
    return gts, total_gts

def compute_map(predictions, gts, total_gts):
    predictions.sort(key=lambda x: x['score'], reverse=True)
    map_scores = []
    
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        # Reset matches
        for v in gts.values():
            v['matched'] = [False] * len(v['masks'])
            
        tps = np.zeros(len(predictions))
        fps = np.zeros(len(predictions))
        
        for i, pred in enumerate(predictions):
            img_name = pred['img_name']
            pred_mask = pred['mask']
            
            best_iou = 0.0
            best_gt_idx = -1
            
            gt_data = gts.get(img_name, {'masks': [], 'matched': []})
            # Handle resize mismatch: pred_mask might be different size if it wasn't resized
            for j, gt_mask in enumerate(gt_data['masks']):
                if pred_mask.shape != gt_mask.shape:
                    pm = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                else:
                    pm = pred_mask
                
                iou = calculate_iou(pm, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
                    
            if best_iou >= iou_thresh and best_gt_idx >= 0:
                if not gt_data['matched'][best_gt_idx]:
                    tps[i] = 1
                    gt_data['matched'][best_gt_idx] = True
                else:
                    fps[i] = 1
            else:
                fps[i] = 1
                
        # Compute AP
        tp_cumsum = np.cumsum(tps)
        fp_cumsum = np.cumsum(fps)
        recalls = tp_cumsum / total_gts if total_gts > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        
        if iou_thresh == 0.5:
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-16)
            if len(f1_scores) > 0:
                max_f1_idx = np.argmax(f1_scores)
                p50 = precisions[max_f1_idx]
                r50 = recalls[max_f1_idx]
            else:
                p50, r50 = 0.0, 0.0
                
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([1.0], precisions, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            
        i_idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i_idx + 1] - mrec[i_idx]) * mpre[i_idx + 1])
        map_scores.append(ap)
        
    return map_scores[0], np.mean(map_scores), p50, r50

def main():
    os.chdir(ROOT)
    
    # 1. Load Models
    print("Loading RT-DETR and SAM models...")
    det_model = RTDETR('runs/detect/main_rtdetr_100ep_server3/weights/best.pt')
    sam_model = SAM(str(ROOT / 'checkpoints' / 'sam_b.pt'))
    
    results_summary = {}
    
    # 2. Iterate datasets
    for dataset in EVAL_SETS:
        name = dataset['name']
        img_dir = Path(dataset['img_dir'])
        gt_dir = Path(dataset['gt_dir'])
        
        print(f"\n{'='*50}\nEvaluating RT-DETR + SAM on {name.upper()}\n{'='*50}")
        
        if not img_dir.exists() or not gt_dir.exists():
            print(f"Skipping {name}: path not found")
            continue
            
        image_files = list(img_dir.glob("*.*"))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.png', '.tif', '.jpeg']]
        
        print(f"Found {len(image_files)} images, loading ground truths...")
        gts, total_gts = load_gt_masks(gt_dir, [f.name for f in image_files])
        
        predictions = []
        out_dir = ROOT / 'results' / 'visuals' / 'rtdetr_sam' / f'rtdetr_sam_eval_{name}'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        save_viz = 5 # Save only a few visual examples to avoid large storage
        
        for i, img_path in enumerate(tqdm(image_files, desc=f"Inference {name}")):
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            # Step 1: RT-DETR detection
            det_res = det_model(img, verbose=False)[0]
            boxes = det_res.boxes.xyxy.cpu().numpy()
            confs = det_res.boxes.conf.cpu().numpy()
            
            if len(boxes) > 0:
                # Step 2: SAM Segmentation using detected boxes
                # Convert bboxes to float list format for SAM
                bboxes = boxes.tolist()
                sam_res = sam_model(img, bboxes=bboxes, verbose=False)[0]
                
                # SAM returns masks matching the boxes
                if sam_res.masks is not None:
                    masks_raw = sam_res.masks.data.cpu().numpy() # [N, H, W] or similar
                    
                    # Store predictions
                    n_masks = min(len(boxes), masks_raw.shape[0])
                    masks_2d = []
                    for k in range(n_masks):
                        pm = np.squeeze(masks_raw[k])
                        if pm.ndim != 2:
                            continue
                        pred_mask = pm > 0.5
                        masks_2d.append(pred_mask)
                        predictions.append({
                            'img_name': img_path.name,
                            'score': float(confs[k]),
                            'mask': pred_mask
                        })
                    
                    masks = masks_2d
                    # Visualize and save a few examples
                    if save_viz > 0 and len(masks) > 0:
                        viz_img = img.copy()
                        for mask in masks:
                            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                            # Resize mask to image size if needed
                            if mask.shape[:2] != viz_img.shape[:2]:
                                mask_resized = cv2.resize(mask.astype(np.uint8), (viz_img.shape[1], viz_img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                            else:
                                mask_resized = mask
                            viz_img[mask_resized] = (viz_img[mask_resized] * 0.5 + color * 0.5).astype(np.uint8)
                        for box in bboxes:
                            cv2.rectangle(viz_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                        cv2.imwrite(str(out_dir / f"{img_path.name}_sam.jpg"), viz_img)
                        save_viz -= 1

        # Calculate AP metrics
        print("Calculating mAP metrics...")
        map50, map50_95, p50, r50 = compute_map(predictions, gts, total_gts)
        
        results_summary[name] = {
            'Precision': float(p50),
            'Recall': float(r50),
            'Mask mAP50': map50,
            'Mask mAP50_95': map50_95
        }
        
        print(f"Results for {name.upper()}:")
        print(f"  Total Ground Truths: {total_gts}")
        print(f"  Precision:        {p50*100:.2f}%")
        print(f"  Recall:           {r50*100:.2f}%")
        print(f"  Mask mAP@0.5:     {map50*100:.2f}%")
        print(f"  Mask mAP@0.5:0.95: {map50_95*100:.2f}%")

    # Save summary
    summary_path = ROOT / 'results' / 'summary' / 'rtdetr_sam_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    print(f"\nAll evaluations complete. Summary saved to {summary_path}")

if __name__ == '__main__':
    main()
