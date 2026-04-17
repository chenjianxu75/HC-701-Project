"""
Kvasir-SEG Mask to YOLO Segmentation Format Converter
Converts binary mask images to YOLO polygon format
Auto-splits into train/val (80/20)
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import random

# Configuration
ROOT = Path(__file__).resolve().parents[2]
KVASIR_ROOT = ROOT / "kvasir-seg"
OUTPUT_ROOT = ROOT / "datasets" / "kvasir_seg"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

def mask_to_yolo_polygon(mask_path, img_width, img_height):
    """Convert binary mask to YOLO polygon format"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    # Threshold to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    yolo_lines = []
    for contour in contours:
        # Simplify contour to reduce points
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:  # Need at least 3 points for polygon
            continue
        
        # Normalize coordinates
        points = approx.reshape(-1, 2)
        normalized = []
        for x, y in points:
            nx = x / img_width
            ny = y / img_height
            normalized.extend([nx, ny])
        
        # Format: class_id x1 y1 x2 y2 ...
        coords = " ".join([f"{c:.6f}" for c in normalized])
        yolo_lines.append(f"0 {coords}")
    
    return yolo_lines


def main():
    random.seed(RANDOM_SEED)
    
    # Get all image files
    images_dir = KVASIR_ROOT / "images"
    masks_dir = KVASIR_ROOT / "masks"
    
    if not images_dir.exists():
        print(f"Error: {images_dir} not found")
        return
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * TRAIN_RATIO)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create output directories
    for split in ["train", "val"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Process files
    for split, files in [("train", train_files), ("val", val_files)]:
        success_count = 0
        for img_path in files:
            # Find corresponding mask
            mask_name = img_path.stem + ".jpg"  # Kvasir masks are jpg
            mask_path = masks_dir / mask_name
            
            if not mask_path.exists():
                mask_name = img_path.stem + ".png"
                mask_path = masks_dir / mask_name
            
            if not mask_path.exists():
                print(f"Warning: Mask not found for {img_path.name}")
                continue
            
            # Get image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # Convert mask to YOLO format
            yolo_lines = mask_to_yolo_polygon(mask_path, w, h)
            
            if not yolo_lines:
                print(f"Warning: No valid contours for {img_path.name}")
                continue
            
            # Copy image
            shutil.copy(img_path, OUTPUT_ROOT / "images" / split / img_path.name)
            
            # Write label
            label_path = OUTPUT_ROOT / "labels" / split / (img_path.stem + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
            
            success_count += 1
        
        print(f"{split}: {success_count} files processed successfully")
    
    print("\nDone! Dataset ready at:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
