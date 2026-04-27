"""
Per-class IoU evaluation.

Usage:
    python eval_per_class_iou.py
    python eval_per_class_iou.py --encoder efficientnet-b2 --checkpoint checkpoints/efficientnet-b2_best.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import ResistorSegDataset, get_val_transform, NUM_CLASSES

CLASS_NAMES = [
    "background", "black", "blue", "brown", "gold",
    "green", "grey", "orange", "violet", "red",
    "silver", "white", "yellow",
]

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "segmentation"
)


def build_model(checkpoint_path: str, encoder: str, device: torch.device):
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model


def compute_per_class_iou(preds: np.ndarray, targets: np.ndarray, num_classes: int):
    """Returns IoU per class (None if class absent from both pred and target)."""
    ious = []
    for cls in range(num_classes):
        p = preds == cls
        t = targets == cls
        inter = (p & t).sum()
        union = (p | t).sum()
        ious.append(inter / union if union > 0 else None)
    return ious


def evaluate(model, loader, device):
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(masks.numpy())
    preds   = np.concatenate([p.flatten() for p in all_preds])
    targets = np.concatenate([t.flatten() for t in all_targets])
    return compute_per_class_iou(preds, targets, NUM_CLASSES)


def print_results(label: str, ious: list):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  {'Class':<12} {'IoU':>8}")
    print(f"  {'-'*22}")
    valid = []
    for cls, iou in enumerate(ious):
        if iou is not None:
            print(f"  {CLASS_NAMES[cls]:<12} {iou:>8.4f}")
            valid.append(iou)
        else:
            print(f"  {CLASS_NAMES[cls]:<12} {'N/A':>8}  (absent)")
    # mIoU matching train.py: skip background (class 0)
    non_bg = [iou for cls, iou in enumerate(ious) if cls != 0 and iou is not None]
    print(f"  {'-'*22}")
    print(f"  {'mIoU (no bg)':<12} {np.mean(non_bg):>8.4f}")
    print(f"  {'mIoU (all)':<12} {np.mean(valid):>8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",    default="efficientnet-b0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir",   default=DATA_DIR)
    args = parser.parse_args()

    ckpt = args.checkpoint or os.path.join(
        os.path.dirname(__file__), "checkpoints", f"{args.encoder}_best.pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Encoder: {args.encoder}")
    print(f"Checkpoint: {ckpt}")

    val_ds = ResistorSegDataset(
        os.path.join(args.data_dir, "val", "images"),
        os.path.join(args.data_dir, "val", "masks"),
        transform=get_val_transform(),
        size=512,
    )
    loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    print(f"Val set: {len(val_ds)} images")

    model = build_model(ckpt, args.encoder, device)
    ious  = evaluate(model, loader, device)
    print_results(args.encoder, ious)


if __name__ == "__main__":
    main()
