"""
Compute real LAB reference colors from the labeled training dataset.

For each class (1-12), collects all pixels of that class across all
training images, converts to CIE LAB, then saves the per-class median
to band_extraction/lab_references.json.

Usage:
    python calibrate_colors.py
    python calibrate_colors.py --data_dir ../datasets/segmentation
"""

import os
import argparse
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

ID_TO_COLOR_NAME = {
    1: "black",  2: "blue",   3: "brown",  4: "gold",
    5: "green",  6: "grey",   7: "orange", 8: "violet",
    9: "red",    10: "silver", 11: "white", 12: "yellow",
}


def collect_lab_pixels(data_dir: str):
    """
    Walk train (and val) splits, collect LAB pixels per class.
    Returns dict: class_id → (N, 3) float32 array of LAB pixels.
    """
    buckets = {c: [] for c in ID_TO_COLOR_NAME}

    for split in ("train", "val"):
        img_dir  = os.path.join(data_dir, split, "images")
        mask_dir = os.path.join(data_dir, split, "masks")
        if not os.path.isdir(img_dir):
            continue

        fnames = sorted(f for f in os.listdir(img_dir)
                        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"})

        for fname in tqdm(fnames, desc=split):
            img_path = os.path.join(img_dir, fname)
            stem = os.path.splitext(fname)[0]
            mask_path = None
            for ext in (".png", ".jpg", ".jpeg"):
                p = os.path.join(mask_dir, stem + ext)
                if os.path.exists(p):
                    mask_path = p
                    break
            if mask_path is None:
                continue

            try:
                img  = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
                mask = np.array(Image.open(mask_path).convert("L"),   dtype=np.uint8)
            except Exception:
                continue

            if img.shape[:2] != mask.shape:
                continue

            # Convert to float32 LAB
            img_f32 = img.astype(np.float32) / 255.0
            lab = cv2.cvtColor(img_f32[:, :, ::-1], cv2.COLOR_BGR2LAB)

            for cls in ID_TO_COLOR_NAME:
                ys, xs = np.where(mask == cls)
                if len(ys) == 0:
                    continue
                pixels = lab[ys, xs]           # (N, 3)
                buckets[cls].append(pixels)

    return {cls: np.concatenate(arrs, axis=0) if arrs else None
            for cls, arrs in buckets.items()}


def compute_references(buckets):
    refs = {}
    for cls, pixels in buckets.items():
        name = ID_TO_COLOR_NAME[cls]
        if pixels is None or len(pixels) < 10:
            print(f"  WARNING: {name} has < 10 pixels — skipping")
            continue
        median = np.median(pixels, axis=0)
        refs[name] = median.tolist()
        print(f"  {name:8s}: L={median[0]:.1f}  A={median[1]:.1f}  B={median[2]:.1f}"
              f"  (n={len(pixels):,})")
    return refs


def main(args):
    print(f"Collecting LAB pixels from: {args.data_dir}\n")
    buckets = collect_lab_pixels(args.data_dir)
    print("\nMedian LAB per class:")
    refs = compute_references(buckets)

    out_path = os.path.join(os.path.dirname(__file__), "lab_references.json")
    with open(out_path, "w") as f:
        json.dump(refs, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=os.path.join("..", "datasets", "segmentation"))
    main(parser.parse_args())
