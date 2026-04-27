import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

NUM_CLASSES = 13  # 0=background + 12 band colours (black→yellow, bronze removed)
IMG_SIZE = 512
MIN_SHORT_SIDE = 50

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def resize_pad(image: np.ndarray, mask: np.ndarray | None, size: int = IMG_SIZE):
    """
    Resize image (and optionally mask) to fit inside a square canvas,
    preserving aspect ratio. Pads with black (zeros).

    Returns:
        canvas_img  : (size, size, 3) uint8
        canvas_mask : (size, size)    uint8  -- only if mask is not None
        offsets     : (pad_y, pad_x, new_h, new_w) -- only if mask is None (for unpadding at inference)
    """
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    pil_img = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    canvas_img = np.zeros((size, size, 3), dtype=np.uint8)
    pad_y = (size - new_h) // 2
    pad_x = (size - new_w) // 2
    canvas_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.array(pil_img)

    if mask is not None:
        pil_mask = Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST)
        canvas_mask = np.zeros((size, size), dtype=np.uint8)
        canvas_mask[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.array(pil_mask)
        return canvas_img, canvas_mask

    return canvas_img, (pad_y, pad_x, new_h, new_w)


def crop_to_content(image: np.ndarray, mask: np.ndarray, size: int = IMG_SIZE,
                    pad_frac: float = 0.20):
    """
    Crop to the bounding box of non-background mask pixels + padding,
    then resize to size x size. Normalises the resistor scale so it fills
    a consistent fraction of the canvas regardless of source crop tightness.
    Falls back to resize_pad when the mask is empty.
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return resize_pad(image, mask, size)

    rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])

    h, w = image.shape[:2]
    bh, bw = max(1, rmax - rmin), max(1, cmax - cmin)
    py = max(8, int(bh * pad_frac))
    px = max(8, int(bw * pad_frac))

    rmin = max(0, rmin - py);  rmax = min(h, rmax + py)
    cmin = max(0, cmin - px);  cmax = min(w, cmax + px)

    return resize_pad(image[rmin:rmax, cmin:cmax], mask[rmin:rmax, cmin:cmax], size)


def _find_mask(mask_dir: str, stem: str) -> str | None:
    """Find mask file matching image stem, tolerating .png/.jpg extension differences."""
    for ext in ('.png', '.jpg', '.jpeg'):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


class ResistorSegDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, transform=None,
                 size: int = IMG_SIZE, min_short_side: int = MIN_SHORT_SIDE,
                 scale_normalize: bool = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.size = size
        self.scale_normalize = scale_normalize
        self.pairs = []

        n_dropped = 0
        for fname in sorted(os.listdir(image_dir)):
            img_path = os.path.join(image_dir, fname)
            stem = os.path.splitext(fname)[0]
            mask_path = _find_mask(mask_dir, stem)
            if mask_path is None:
                continue
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if min(w, h) >= min_short_side:
                        self.pairs.append((img_path, mask_path))
                    else:
                        n_dropped += 1
            except Exception:
                n_dropped += 1

        print(f'[Dataset] {os.path.basename(image_dir)}: '
              f'{len(self.pairs)} kept, {n_dropped} dropped (short side < {min_short_side}px)')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        mask  = np.array(Image.open(mask_path).convert('L'))
        if self.scale_normalize:
            image, mask = crop_to_content(image, mask, self.size)
        else:
            image, mask = resize_pad(image, mask, self.size)

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out['image'], out['mask']

        return image, mask.long()

    def get_sample_weights(self) -> list[float]:
        """
        Per-image sampling weight for WeightedRandomSampler.
        Images containing rare band classes get higher weight so every
        class combination is seen roughly equally often.
        """
        from collections import Counter
        class_counts: Counter = Counter()
        image_classes: list[set] = []

        for _, mask_path in self.pairs:
            mask_arr = np.array(Image.open(mask_path).convert('L'))
            classes = set(int(c) for c in np.unique(mask_arr) if c != 0)
            image_classes.append(classes)
            class_counts.update(classes)

        n = len(self.pairs)
        # inverse-frequency weight per class (floor at 1 to avoid div-by-zero)
        inv_freq = {c: n / max(cnt, 1) for c, cnt in class_counts.items()}

        weights = []
        for classes in image_classes:
            w = max((inv_freq.get(c, 1.0) for c in classes), default=1.0)
            weights.append(w)

        # print a summary so the user can see what's being upweighted
        print('[Sampler] class frequencies in training set:')
        for c in sorted(class_counts):
            print(f'  class {c:2d}: {class_counts[c]:4d} images  weight={inv_freq[c]:.2f}')

        return weights


def get_train_transform(size: int = IMG_SIZE):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # Reduced from 180° — resistors in YOLO crops are roughly upright.
        # Full 360 rotation hurts more than it helps once scale is normalised.
        A.Rotate(limit=45, border_mode=0, p=0.7),
        # Small scale + translation jitter only; wide scale range is no longer needed
        # because crop_to_content already normalises the resistor size.
        A.Affine(scale=(0.85, 1.10), translate_percent=(-0.08, 0.08),
                 fill=0, fill_mask=0, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        # Tight HSV jitter only — large hue shifts destroy class identity
        # (red→green, brown→orange, etc.) and teach the model colour is unreliable.
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        A.RandomGamma(gamma_limit=(60, 140), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32),
                        hole_width_range=(8, 32), fill=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
