"""
End-to-end resistor value detection pipeline.

Output per image:
  <stem>_resistor<N>.png  — 3-panel composite: detection | seg overlay | band extraction

Output per run:
  results.txt  — detected value for every file

Usage:
    python pipeline.py path/to/image.jpg
    python pipeline.py path/to/images/
    python pipeline.py image.jpg --det_weights weights/detection/best.pt \\
                                 --seg_weights weights/segmentation/efficientnet-b2_best.pt \\
                                 --output_dir results/pipeline_output
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, 'band_extraction'))
sys.path.insert(0, os.path.join(_ROOT, 'segmentation'))

from ultralytics import YOLO
from infer import load_model, run_inference, VIS_COLORS, IMG_EXTS
from dataset import get_val_transform, IMG_SIZE
from resistance_calculator import calculate_resistance_with_axis_info
from color_code_tables import ResistanceResult

PANEL_H = 300
GAP_PX  = 8

_COLOR_HEX = {
    'black':'#000000','blue':'#004aad','brown':'#5e3831','gold':'#ebbf7c',
    'green':'#00bf63','grey':'#585a59','orange':'#ff914d','violet':'#5e17eb',
    'red':'#e43232','silver':'#cdcdcd','white':'#ffffff','yellow':'#f4cb24',
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _rh(arr, h):
    """Resize array to height h maintaining aspect ratio. Returns (resized, scale)."""
    im = Image.fromarray(arr.astype(np.uint8))
    s  = h / im.height
    nw = max(1, int(im.width * s))
    return np.array(im.resize((nw, h), Image.LANCZOS)), s


def _seg_overlay(crop_arr, vis_arr, alpha=0.6):
    nonzero = vis_arr.sum(axis=2) > 0
    out = crop_arr.copy()
    out[nonzero] = (alpha * vis_arr[nonzero] + (1 - alpha) * crop_arr[nonzero]).astype(np.uint8)
    return out


def _draw_p3(arr, bands, axis_info, s3, crop_arr):
    """Draw band markers on panel 3 using OpenCV."""
    out = arr.copy()
    if not (axis_info and bands):
        return out
    h_c, w_c = crop_arr.shape[:2]
    av   = np.array(axis_info['axis_vector'])
    ao   = np.array(axis_info['axis_origin']) * s3
    perp = np.array([-av[1], av[0]])
    tick = int(max(14, int(min(h_c, w_c) * 0.20) * s3))
    H, W = out.shape[:2]
    t    = max(W, H)
    cv2.line(out,
             (int(ao[0] - t*av[0]), int(ao[1] - t*av[1])),
             (int(ao[0] + t*av[0]), int(ao[1] + t*av[1])),
             (180, 180, 180), 1, cv2.LINE_AA)
    for j, band in enumerate(bands):
        cx, cy = int(band.centroid[0] * s3), int(band.centroid[1] * s3)
        hx  = _COLOR_HEX.get(band.color_name.lower(), '#ffffff')
        col = (int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16))
        cv2.line(out,
                 (int(cx - perp[0]*tick), int(cy - perp[1]*tick)),
                 (int(cx + perp[0]*tick), int(cy + perp[1]*tick)),
                 (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 9, col,          -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 9, (255, 255, 255), 1, cv2.LINE_AA)
        tc = (20, 20, 20) if band.color_name in ('white', 'gold', 'yellow', 'silver') else (255, 255, 255)
        cv2.putText(out, str(j + 1), (cx - 4, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, tc, 1, cv2.LINE_AA)
    return out


def _badge(img, x1, y1, text, font_scale=0.5, color=(0, 210, 0)):
    """Draw a filled-background text badge at (x1, y1-pad) on img (in-place, BGR)."""
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = 4
    bx1, by1 = x1, max(0, y1 - th - 2 * pad)
    bx2, by2 = x1 + tw + 2 * pad, y1
    cv2.rectangle(img, (bx1, by1), (bx2, by2), color, -1)
    cv2.putText(img, text, (bx1 + pad, by2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Composite builder
# ---------------------------------------------------------------------------

def _make_composite(img_np, crop_np, vis, bands, axis_info,
                    x1, y1, x2, y2, det_conf, value_str, tol_str):
    """
    3-panel PIL composite (all panels PANEL_H pixels tall):
      (a) full image with detection box + badge
      (b) seg overlay on crop
      (c) band extraction on crop with cv2 markers
    """
    # Panel 1: detection on full image
    det_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.rectangle(det_bgr, (x1, y1), (x2, y2), (0, 210, 0), 2)
    det_rgb = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2RGB)
    p1, s1  = _rh(det_rgb, PANEL_H)
    # Badge above box
    p1_bgr  = cv2.cvtColor(p1, cv2.COLOR_RGB2BGR)
    _badge(p1_bgr, int((x1 + 4) * s1), int(y1 * s1),
           f'resistor {det_conf:.2f}')
    p1 = cv2.cvtColor(p1_bgr, cv2.COLOR_BGR2RGB)

    # Panel 2: seg overlay on crop
    p2, _ = _rh(_seg_overlay(crop_np, vis, 0.6), PANEL_H)

    # Panel 3: band extraction
    p3b, s3 = _rh(_seg_overlay(crop_np, vis, 0.30), PANEL_H)
    p3 = _draw_p3(p3b, bands, axis_info, s3, crop_np)

    # Stitch
    gap    = np.full((PANEL_H, GAP_PX, 3), 245, dtype=np.uint8)
    row    = np.concatenate([p1, gap, p2, gap, p3], axis=1)
    p1_w   = p1.shape[1]
    p3_off = p1_w + GAP_PX + p2.shape[1] + GAP_PX

    # Value label across top of panel 3 — strip Unicode for Hershey font
    label    = f'{value_str}  {tol_str}'.replace('Ω', 'Ohm').replace('±', '+/-')
    row_bgr  = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
    _badge(row_bgr, p3_off + 4, PANEL_H, label, font_scale=0.55, color=(0, 180, 0))
    row = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(row)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def _nms_boxes(boxes_xyxy, confs, iou_thresh=0.5):
    if len(boxes_xyxy) == 0:
        return []
    order = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
    kept, suppressed = [], set()
    for i in order:
        if i in suppressed:
            continue
        kept.append(i)
        x1i, y1i, x2i, y2i = boxes_xyxy[i]
        ai = max(0, x2i - x1i) * max(0, y2i - y1i)
        for j in order:
            if j in suppressed or j == i:
                continue
            x1j, y1j, x2j, y2j = boxes_xyxy[j]
            inter = max(0, min(x2i, x2j) - max(x1i, x1j)) * max(0, min(y2i, y2j) - max(y1i, y1j))
            if inter == 0:
                continue
            aj = max(0, x2j - x1j) * max(0, y2j - y1j)
            if inter / (ai + aj - inter + 1e-6) > iou_thresh:
                suppressed.add(j)
    return kept


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def run_pipeline(image_path, det_model, seg_model, transform, device, args):
    img_bgr = cv2.imread(image_path)
    img_np  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = det_model(image_path, verbose=False, conf=0.01)

    if not results or len(results[0].boxes) == 0:
        return []

    raw_boxes = [box.xyxy[0].tolist() for box in results[0].boxes]
    raw_confs = [float(box.conf[0])   for box in results[0].boxes]
    keep_idx  = _nms_boxes(raw_boxes, raw_confs, iou_thresh=0.5)
    kept_boxes = [results[0].boxes[i] for i in keep_idx]

    if not kept_boxes:
        return []

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    detections = []
    for i, box in enumerate(kept_boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_conf        = float(box.conf[0])
        h, w = img_np.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(0, x1 - int(bw * .10)); cy1 = max(0, y1 - int(bh * .10))
        cx2 = min(w, x2 + int(bw * .10)); cy2 = min(h, y2 + int(bh * .10))
        crop_np = img_np[cy1:cy2, cx1:cx2]
        if min(crop_np.shape[:2]) < 30:
            continue

        tmp = os.path.join(args.output_dir, f'_tmp_{stem}_{i}.png')
        cv2.imwrite(tmp, cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR))
        mask, vis, seg_conf = run_inference(seg_model, tmp, transform, device, args.size)
        os.remove(tmp)

        result, axis_info = calculate_resistance_with_axis_info(mask)

        if isinstance(result, ResistanceResult):
            value_str = result.formatted
            tol_str   = f'±{result.tolerance:.4g}%'
            bands     = result.bands
        else:
            value_str = f'ERROR({result.error_type})'
            tol_str   = ''
            bands     = result.detected_bands

        composite = _make_composite(
            img_np, crop_np, vis, bands, axis_info,
            x1, y1, x2, y2, det_conf, value_str, tol_str,
        )
        composite.save(os.path.join(args.output_dir, f'{stem}_resistor{i}.png'))

        detections.append({
            'crop':      i,
            'det_box':   (x1, y1, x2, y2),
            'det_conf':  det_conf,
            'seg_conf':  seg_conf,
            'value':     value_str,
            'tolerance': tol_str,
        })

    return detections


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Detection weights:    {args.det_weights}')
    print(f'Segmentation weights: {args.seg_weights}\n')

    det_model = YOLO(args.det_weights)
    seg_model = load_model(args.seg_weights, device)
    transform = get_val_transform()

    if os.path.isdir(args.input):
        inputs = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        )
    else:
        inputs = [args.input]

    print(f'Processing {len(inputs)} image(s)...\n')

    all_results = []
    for img_path in inputs:
        fname      = os.path.basename(img_path)
        detections = run_pipeline(img_path, det_model, seg_model, transform, device, args)
        all_results.append((fname, detections))

        if not detections:
            print(f'{fname}  →  [no detection]')
        for d in detections:
            print(f'{fname}  →  resistor {d["crop"]}: {d["value"]} {d["tolerance"]}'
                  f'  (det={d["det_conf"]:.2f}, seg={d["seg_conf"]:.2f})')

    os.makedirs(args.output_dir, exist_ok=True)
    n_det     = sum(len(d) for _, d in all_results)
    n_decoded = sum(1 for _, d in all_results for det in d if not det['value'].startswith('ERROR'))

    txt_path = os.path.join(args.output_dir, 'results.txt')
    with open(txt_path, 'w') as f:
        f.write('Resistor Detection Results\n')
        f.write('==========================\n\n')
        for fname, detections in all_results:
            f.write(f'{fname}\n')
            if not detections:
                f.write('  [no detection]\n')
            for d in detections:
                f.write(f'  resistor {d["crop"]}: {d["value"]} {d["tolerance"]}\n'.rstrip() + '\n')
            f.write('\n')
        f.write(f'Summary: {len(inputs)} images  |  {n_det} resistors detected  |  {n_decoded} values decoded\n')

    print(f'\nOutputs saved to: {args.output_dir}')
    print(f'Results:          {txt_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end resistor value detection')
    parser.add_argument('input',          help='Image file or directory')
    parser.add_argument('--det_weights',  default=os.path.join('weights', 'detection', 'best.pt'))
    parser.add_argument('--seg_weights',  default=os.path.join('weights', 'segmentation', 'efficientnet-b2_best.pt'))
    parser.add_argument('--output_dir',   default=os.path.join('results', 'pipeline_output'))
    parser.add_argument('--size',         type=int, default=IMG_SIZE)
    main(parser.parse_args())
