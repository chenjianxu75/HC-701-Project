"""
Calculate Medical Metrics: Dice Score and IoU
For evaluating polyp segmentation results
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def dice_score(pred, gt):
    """Calculate Dice coefficient"""
    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0  # Both empty
    return 2.0 * intersection / union


def iou_score(pred, gt):
    """Calculate IoU (Jaccard Index)"""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return intersection / union


def evaluate_predictions(model_path, data_yaml, split="val"):
    """
    Evaluate model predictions and calculate Dice/IoU
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset yaml config
        split: 'val' or 'test'
    """
    model = YOLO(model_path)
    
    # Run validation to get predictions
    results = model.val(data=data_yaml, split=split, save=True, plots=True)
    
    print("\n" + "="*50)
    print("YOLO Standard Metrics:")
    print(f"  Box mAP50: {results.box.map50:.4f}")
    print(f"  Box mAP50-95: {results.box.map:.4f}")
    print(f"  Mask mAP50: {results.seg.map50:.4f}")
    print(f"  Mask mAP50-95: {results.seg.map:.4f}")
    print("="*50)
    
    return results


def calculate_dice_from_masks(pred_dir, gt_dir):
    """
    Calculate Dice and IoU from saved prediction masks
    
    Args:
        pred_dir: Directory containing predicted masks
        gt_dir: Directory containing ground truth masks
    """
    dice_scores = []
    iou_scores = []
    
    pred_files = list(Path(pred_dir).glob("*.png"))
    
    for pred_path in pred_files:
        gt_path = Path(gt_dir) / pred_path.name
        
        if not gt_path.exists():
            continue
        
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        
        if pred is None or gt is None:
            continue
        
        # Resize if needed
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        
        # Binarize
        pred_bin = pred > 127
        gt_bin = gt > 127
        
        dice_scores.append(dice_score(pred_bin, gt_bin))
        iou_scores.append(iou_score(pred_bin, gt_bin))
    
    print("\n" + "="*50)
    print("Medical Metrics (Pixel-level):")
    print(f"  Mean Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"  Mean IoU Score:  {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"  Samples evaluated: {len(dice_scores)}")
    print("="*50)
    
    return dice_scores, iou_scores


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate Dice and IoU metrics")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, default="configs/kvasir_seg.yaml", help="Dataset yaml")
    parser.add_argument("--split", type=str, default="val", help="Data split (val/test)")
    
    args = parser.parse_args()
    
    print(f"Evaluating: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Split: {args.split}")
    
    evaluate_predictions(args.model, args.data, args.split)
