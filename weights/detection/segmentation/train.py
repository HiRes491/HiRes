"""
Resistor band segmentation trainer.

Usage:
    python train.py
    python train.py --encoder efficientnet-b0
    python train.py --data_dir ../datasets/segmentation --scale_normalize
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset import ResistorSegDataset, get_train_transform, get_val_transform, NUM_CLASSES


def build_model(encoder: str = 'efficientnet-b2'):
    return smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )


def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = NUM_CLASSES) -> float:
    """Mean IoU across non-background classes."""
    preds   = preds.flatten().numpy()
    targets = targets.flatten().numpy()
    ious = []
    for cls in range(1, num_classes):
        p = preds   == cls
        t = targets == cls
        inter = (p & t).sum()
        union = (p | t).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # AMP (fp16) causes NaN overflow with EfficientNet on Turing GPUs (GTX 1650 Ti).
    # bfloat16 requires Ampere+. Train in fp32 with batch_size=2 instead.
    use_amp = False
    print(f'Device: {device}  |  AMP: {use_amp}  |  Encoder: {args.encoder}')
    print(f'scale_normalize={args.scale_normalize}  |  balanced_sampling={args.balanced_sampling}')

    # ---- Datasets ----------------------------------------------------------
    train_ds = ResistorSegDataset(
        os.path.join(args.data_dir, 'train', 'images'),
        os.path.join(args.data_dir, 'train', 'masks'),
        transform=get_train_transform(args.size),
        size=args.size,
        scale_normalize=args.scale_normalize,
    )
    val_ds = ResistorSegDataset(
        os.path.join(args.data_dir, 'val', 'images'),
        os.path.join(args.data_dir, 'val', 'masks'),
        transform=get_val_transform(),
        size=args.size,
        scale_normalize=args.scale_normalize,
    )
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

    if args.balanced_sampling:
        weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.workers, pin_memory=use_amp, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=use_amp, drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=use_amp,
    )

    # ---- Model -------------------------------------------------------------
    model = build_model(args.encoder).to(device)

    # ---- Loss --------------------------------------------------------------
    # Balanced sampling already handles class imbalance; focal gamma=2 downweights
    # easy background pixels. No need for manual class_weights on top.
    dice_loss  = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    focal_loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.0, normalized=False)

    def criterion(logits, targets):
        return 0.5 * dice_loss(logits, targets) + 0.5 * focal_loss(logits, targets)

    # ---- Optimizer & Scheduler ---------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    scaler = GradScaler('cuda', enabled=use_amp)

    # ---- Checkpoint setup --------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt = os.path.join(args.checkpoint_dir, f'{args.encoder}_best.pt')

    best_miou     = 0.0
    patience_left = args.patience

    # ---- Training loop -----------------------------------------------------
    for epoch in range(1, args.epochs + 1):

        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f'Ep {epoch:03d}/{args.epochs} train', leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Ep {epoch:03d}/{args.epochs} val  ', leave=False):
                images, masks = images.to(device), masks.to(device)
                with autocast('cuda', enabled=use_amp):
                    logits = model(images)
                    loss   = criterion(logits, masks)
                val_loss += loss.item()
                all_preds.append(logits.argmax(dim=1).cpu())
                all_targets.append(masks.cpu())
        val_loss /= len(val_loader)

        miou = compute_miou(torch.cat(all_preds), torch.cat(all_targets))
        print(f'Ep {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | mIoU={miou:.4f}'
              f' | lr={scheduler.get_last_lr()[0]:.2e}')

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), best_ckpt)
            print(f'         -> best mIoU={best_miou:.4f}  saved to {best_ckpt}')
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f'Early stopping at epoch {epoch}.')
                break

    print(f'\nFinished. Best val mIoU: {best_miou:.4f}')
    print(f'Checkpoint: {best_ckpt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resistor band segmentation trainer')

    parser.add_argument('--data_dir',       default=os.path.join('..', 'datasets', 'segmentation'))
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--encoder',    default='efficientnet-b2',
                        help='SMP encoder name, e.g. efficientnet-b0, efficientnet-b2')
    parser.add_argument('--size',       type=int,   default=512)
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--batch_size', type=int,   default=2,
                        help='fp32 on GTX 1650 Ti — keep at 2 to stay within 4 GB VRAM')
    parser.add_argument('--lr',         type=float, default=2e-4)
    parser.add_argument('--patience',   type=int,   default=30)
    parser.add_argument('--workers',    type=int,   default=0,
                        help='Keep at 0 on Windows to avoid multiprocessing issues')
    parser.add_argument('--scale_normalize',  action='store_true',
                        help='Crop to mask bounding box before resizing (normalises resistor scale)')
    parser.add_argument('--balanced_sampling', action='store_true',
                        help='WeightedRandomSampler: upweight images with rare band classes')

    main(parser.parse_args())
