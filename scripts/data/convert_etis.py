"""
ETIS-Larib to YOLO Format Converter
Converts 196 PNG binary masks to YOLO polygon labels (test set only)
"""

import os
import cv2
import numpy as np
from pathlib import Path


# Configuration
ROOT = Path(__file__).resolve().parents[2]
ETIS_ROOT = ROOT / "datasets" / "ETIS-Larib"
OUTPUT_ROOT = ROOT / "datasets" / "etis_larib"


def mask_to_yolo_polygon(mask_path, img_width, img_height):
    """Convert binary mask to YOLO polygon format"""
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
    images_dir = ETIS_ROOT / "images"
    masks_dir = ETIS_ROOT / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        print(f"Error: ETIS-Larib directory structure not found at {ETIS_ROOT}")
        return

    # Create output directories
    (OUTPUT_ROOT / "images" / "test").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / "test").mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = sorted(images_dir.glob("*.png"), key=lambda p: int(p.stem))
    print(f"Found {len(image_files)} images in {images_dir}")

    success_count = 0
    skip_count = 0

    for img_path in image_files:
        img_id = img_path.stem  # e.g. "1", "2", ..., "196"

        # Find corresponding mask
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            print(f"Warning: Mask not found for {img_path.name}")
            skip_count += 1
            continue

        # Read image to get dimensions
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Cannot read image {img_path.name}")
            skip_count += 1
            continue

        h, w = img.shape[:2]

        # Convert mask to YOLO format
        yolo_lines = mask_to_yolo_polygon(mask_path, w, h)

        if not yolo_lines:
            print(f"Warning: No valid contours for {img_path.name}")
            skip_count += 1
            continue

        # Copy image to output
        out_img = OUTPUT_ROOT / "images" / "test" / img_path.name
        cv2.imwrite(str(out_img), img)

        # Write label
        label_path = OUTPUT_ROOT / "labels" / "test" / (img_id + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        success_count += 1

    print(f"\n{'='*50}")
    print(f"ETIS-Larib Conversion Complete!")
    print(f"  Successfully converted: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Output directory: {OUTPUT_ROOT}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
