"""
Inference script — takes a cropped resistor image, outputs segmentation results.

Outputs (per image):
  <name>_mask.png    — grayscale class-ID mask (pixel value = class 0-12)
                        this is the format Final_Stage expects
  <name>_visual.png  — black background + colored band overlay (for human review)

Usage:
    # Single image
    python infer.py path/to/crop.jpg --checkpoint checkpoints/phase2_best.pt

    # Directory of images
    python infer.py path/to/crops/ --checkpoint checkpoints/phase2_best.pt
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch

import segmentation_models_pytorch as smp
from dataset import resize_pad, get_val_transform, NUM_CLASSES, IMG_SIZE

# ---------------------------------------------------------------------------
# Label info  (matches label_map.json)
# ---------------------------------------------------------------------------

LABEL_NAMES = [
    'background',   # 0
    'black',        # 1
    'blue',         # 2
    'brown',        # 3
    'gold',         # 4
    'green',        # 5
    'grey',         # 6
    'orange',       # 7
    'violet',       # 8
    'red',          # 9
    'silver',       # 10
    'white',        # 11
    'yellow',       # 12
]

# Colors for visualization — black background, realistic band colors
VIS_COLORS = np.array([
    (  0,   0,   0),   # 0  background
    ( 30,  30,  30),   # 1  black
    (  0,  74, 173),   # 2  blue
    ( 94,  56,  49),   # 3  brown
    (235, 191, 124),   # 4  gold
    (  0, 191,  99),   # 5  green
    ( 88,  90,  89),   # 6  grey
    (255, 145,  77),   # 7  orange
    (148,   0, 211),   # 8  violet
    (228,  50,  50),   # 9  red
    (192, 192, 192),   # 10 silver
    (255, 255, 255),   # 11 white
    (244, 203,  36),   # 12 yellow
], dtype=np.uint8)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device):
    from train import build_model
    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f'Loaded checkpoint: {ckpt_path}')
    return model


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _run_pass(model, img_np: np.ndarray, transform, device: torch.device,
              size: int) -> tuple[np.ndarray, float]:
    """Single inference pass on a numpy image. Returns (mask, seg_confidence)."""
    import torch.nn.functional as F
    h, w = img_np.shape[:2]
    canvas, (pad_y, pad_x, new_h, new_w) = resize_pad(img_np, None, size)
    out = transform(image=canvas, mask=np.zeros(canvas.shape[:2], dtype=np.uint8))
    img_tensor = out['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
    probs = F.softmax(logits, dim=1)
    max_probs = probs.max(dim=1)[0][0].cpu().numpy()
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    pred_unpad = pred[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    unpad_max_probs = max_probs[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    nonbg = pred_unpad > 0
    seg_confidence = float(unpad_max_probs[nonbg].mean()) if nonbg.any() else float(unpad_max_probs.mean())
    mask = np.array(Image.fromarray(pred_unpad).resize((w, h), Image.NEAREST))
    return mask, seg_confidence


def run_inference(model, img_path: str, transform, device: torch.device, size: int):
    """
    Returns
    -------
    mask : np.ndarray (H, W) uint8 — class IDs at original image resolution
    vis  : np.ndarray (H, W, 3) uint8 — colored visualization
    seg_confidence : float

    Pass 1: run on the full YOLO crop.
    Pass 2: crop to the non-background bbox of the pass-1 mask with 20% padding,
    run again, and place the result back into the full-size mask.
    """
    img_np = np.array(Image.open(img_path).convert('RGB'))
    h, w = img_np.shape[:2]

    mask, seg_confidence = _run_pass(model, img_np, transform, device, size)

    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if rows.any():
        rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
        cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
        bh, bw = max(1, rmax - rmin), max(1, cmax - cmin)
        py = max(8, int(bh * 0.20));  px = max(8, int(bw * 0.20))
        r0 = max(0, rmin - py);  r1 = min(h, rmax + py)
        c0 = max(0, cmin - px);  c1 = min(w, cmax + px)
        crop_np = img_np[r0:r1, c0:c1]
        if min(crop_np.shape[:2]) >= 20:
            crop_mask, seg_confidence = _run_pass(model, crop_np, transform, device, size)
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[r0:r1, c0:c1] = np.array(
                Image.fromarray(crop_mask).resize((c1 - c0, r1 - r0), Image.NEAREST)
            )
            mask = full_mask

    vis = VIS_COLORS[mask]
    return mask, vis, seg_confidence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = load_model(args.checkpoint, device)
    transform = get_val_transform()

    # Collect inputs
    if os.path.isdir(args.input):
        inputs = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        )
    else:
        inputs = [args.input]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Running inference on {len(inputs)} image(s)...\n')

    for img_path in inputs:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask, vis, _ = run_inference(model, img_path, transform, device, args.size)

        mask_out = os.path.join(args.output_dir, f'{stem}_mask.png')
        vis_out  = os.path.join(args.output_dir, f'{stem}_visual.png')

        Image.fromarray(mask).save(mask_out)
        Image.fromarray(vis).save(vis_out)

        bands = [LABEL_NAMES[c] for c in sorted(np.unique(mask)) if c > 0]
        print(f'{os.path.basename(img_path):50s}  bands: {bands}')

    print(f'\nOutputs saved to: {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resistor band segmentation inference')

    parser.add_argument('input',        help='Image file or directory of images')
    parser.add_argument('--checkpoint', required=True,
                        help='Model checkpoint (.pt file)')
    parser.add_argument('--output_dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'inference_output'))
    parser.add_argument('--size',       type=int, default=IMG_SIZE)

    main(parser.parse_args())
