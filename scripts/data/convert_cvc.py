"""
CVC-ClinicDB to YOLO Segmentation Format Converter
Handles TIF format, outputs as test set only
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil

# Configuration
ROOT = Path(__file__).resolve().parents[2]
CVC_ROOT = ROOT / "archive" / "TIF"
OUTPUT_ROOT = ROOT / "datasets" / "cvc_clinicdb"

def find_mask_by_id(mask_folder, img_id):
    """Find mask file by ID, regardless of extension"""
    for ext in [".png", ".jpg", ".tif", ".bmp"]:
        mask_path = mask_folder / (img_id + ext)
        if mask_path.exists():
            return mask_path
    return None


def mask_to_yolo_polygon(mask_path, img_width, img_height):
    """Convert binary mask to YOLO polygon format"""
    # Use IMREAD_UNCHANGED for TIF support
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    
    # Convert to grayscale if needed
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Normalize to 0-255 if needed
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    # Threshold to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    yolo_lines = []
    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            continue
        
        points = approx.reshape(-1, 2)
        normalized = []
        for x, y in points:
            nx = x / img_width
            ny = y / img_height
            normalized.extend([nx, ny])
        
        coords = " ".join([f"{c:.6f}" for c in normalized])
        yolo_lines.append(f"0 {coords}")
    
    return yolo_lines


def main():
    original_dir = CVC_ROOT / "Original"
    gt_dir = CVC_ROOT / "Ground Truth"
    
    if not original_dir.exists():
        print(f"Error: {original_dir} not found")
        return
    
    # Create output directories
    (OUTPUT_ROOT / "images" / "test").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / "test").mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(original_dir.glob("*.tif")) + list(original_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")
    
    success_count = 0
    for img_path in image_files:
        # Extract ID using splitext
        img_id = os.path.splitext(img_path.name)[0]
        
        # Find corresponding mask
        mask_path = find_mask_by_id(gt_dir, img_id)
        
        if mask_path is None:
            print(f"Warning: Mask not found for {img_path.name}")
            continue
        
        # Read image (UNCHANGED for TIF)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Convert mask to YOLO format
        yolo_lines = mask_to_yolo_polygon(mask_path, w, h)
        
        if not yolo_lines:
            print(f"Warning: No valid contours for {img_path.name}")
            continue
        
        # Convert and save as PNG (for compatibility)
        out_name = img_id + ".png"
        cv2.imwrite(str(OUTPUT_ROOT / "images" / "test" / out_name), img)
        
        # Write label
        label_path = OUTPUT_ROOT / "labels" / "test" / (img_id + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
        
        success_count += 1
    
    print(f"\nProcessed {success_count} files successfully")
    print("Dataset ready at:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
