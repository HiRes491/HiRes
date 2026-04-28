# Resistor Value Detection

End-to-end computer vision pipeline that detects resistors in images and reads their resistance value from color bands — supporting 4-band and 5-band resistors.

---

## Pipeline

```
Input image
    │
    ▼
YOLO Detection          — locates and crops each resistor (F1 = 0.9945)
    │
    ▼
Segmentation            — EfficientNet-B2 UNet++, 13 classes (bg + 12 colors)
Two-pass refinement:      pass 1 on full crop → tight bbox → pass 2 on crop
    │
    ▼
Band Extraction         — connected components, noise filtering, fragment merging
    │
    ▼
Axis Detection          — PCA on band centroids → principal resistor axis
    │
    ▼
Direction Heuristic     — gap heuristic (5-band) or gold/silver position (4-band)
    │
    ▼
Resistance Calculation  — color → digit / multiplier / tolerance lookup
```

---

## Results

| | Images | Correct | Accuracy |
|--|--|--|--|
| 4-band | 52 | 47 | **90.4%** |
| 5-band | 54 | 44 | **81.5%** |
| **Overall** | **106** | **91** | **85.8%** |

YOLO detection rate: **91.5%** (97/106). Of detected resistors, value decode rate: **93.8%** (91/97).

### Per-class Segmentation IoU (val set, 140 images)

| Class | IoU | | Class | IoU |
|--|--|--|--|--|
| background | 0.993 | | orange | 0.893 |
| black | 0.912 | | violet | 0.905 |
| blue | 0.941 | | red | 0.929 |
| brown | 0.896 | | silver | 0.660 |
| gold | 0.905 | | white | 0.835 |
| green | 0.869 | | yellow | 0.826 |
| **grey** | **0.564** | | **mIoU (no bg)** | **0.844** |

---

## Project Structure

```
resistor-value-detection/
├── quickstart.ipynb                 # Interactive demo — start here
├── pipeline.py                      # CLI end-to-end pipeline
│
├── segmentation/
│   ├── train.py                     # Training script
│   ├── infer.py                     # Segmentation inference (two-pass)
│   ├── dataset.py                   # Dataset loader + augmentations
│   └── eval_per_class_iou.py        # Per-class IoU evaluation
│
├── band_extraction/
│   ├── band_extractor.py            # Connected components + noise filtering
│   ├── resistance_calculator.py     # Axis detection + direction + value
│   └── color_code_tables.py         # Color → digit / multiplier / tolerance
│
├── weights/
│   ├── detection/best.pt            # YOLOv8 detection weights
│   └── segmentation/
│       └── efficientnet-b2_best.pt  # Segmentation checkpoint (mIoU = 0.844)
│
└── datasets/
    └── segmentation/                # 800-image labeled dataset (train/val split)
        ├── train/images/
        ├── train/masks/
        ├── val/images/
        └── val/masks/
```

---

## Quickstart

```bash
pip install torch torchvision segmentation-models-pytorch ultralytics \
            pillow numpy matplotlib scikit-learn opencv-python albumentations
```

**Interactive demo** — open `quickstart.ipynb` and run all cells.

**CLI pipeline:**
```bash
python pipeline.py path/to/image.jpg
python pipeline.py path/to/images/
```

Outputs a 3-panel PNG per detected resistor (Detection | Segmentation | Band Extraction) and a `results.txt` summary.

---

## Segmentation Model

| | |
|--|--|
| Architecture | UNet++ with EfficientNet-B2 encoder |
| Input size | 512 × 512 (resize + pad, aspect preserved) |
| Classes | 13 (background + 12 band colors) |
| Loss | Focal loss (γ = 2.0) |
| Epochs | 150 |
| Augmentations | Rotation ±45°, flip, hue/saturation jitter, random gamma, coarse dropout |
| Sampling | WeightedRandomSampler (balanced by class) |

**Training:**
```bash
cd segmentation
python train.py --data_dir ../datasets/segmentation \
                --encoder efficientnet-b2 \
                --balanced_sampling \
                --scale_normalize
```

**Evaluation:**
```bash
python eval_per_class_iou.py --encoder efficientnet-b2 \
                              --checkpoint checkpoints/efficientnet-b2_best.pt
```

---

## Band Extraction

1. PCA on all non-background pixel positions → unit axis vector **â** and centroid origin **o**
2. Project every pixel: `t_i = (p_i − o) · â`
3. Discretize into N = 200 uniform bins; accumulate per-class vote histogram V ∈ ℝ^{200×13}
4. Gaussian-smooth each class curve (σ = 2.5 bins) to fill intra-band gaps
5. Assign dominant class per bin: `ĉ[b] = argmax_{c≥1} Ṽ[b,c]`; background bins identified from raw unsmoothed votes to preserve physical gaps between bands
6. Run-length encode dominant-class signal; keep runs spanning ≥ δ = 4 bins and ≥ 40 pixels
7. *(Optional — when RGB available)* Refine each band's color via weighted LAB distance to calibrated references; drop bands exceeding the confidence threshold

### Post-extraction filtering

- Absolute floor: drop bands below `MIN_BAND_AREA` = 40 px
- Relative floor: drop bands below 30% of median area of top-5 bands
- Hard cap: retain at most 5 bands (largest by area)

### Reading Direction

Priority cascade (runs in order, first match wins):

| Priority | Condition | Result |
|----------|-----------|--------|
| 0a | Black at either edge (position 0 or N-1) | `error_black_edge` — invalid as leading digit and as tolerance |
| 0b | Gold/silver count > 2, or count == 2 not adjacent at one end | `error_multiple_tolerance_bands` |
| 0c | Single gold/silver at strictly interior position (N ≥ 5 only) | `error_interior_tolerance_band` |
| 1a | Last band is gold or silver | `forward` |
| 1b | First band is gold or silver | `reverse` |
| 2 | 5-band: known tolerance color at one end only | `forward` / `reverse` |
| 2a | Gap heuristic: end gap ≥ 1.5 × median interior gap | direction from larger qualifying end |
| 2b | Ambiguous | `forward` (default) |

Post-direction: if the resolved tolerance-position color is not in {brown, red, green, blue, violet, grey, gold, silver} → `INVALID_TOLERANCE_COLOR`.

### E24 Retry

If the decoded value is not an E24 preferred value, the two smallest bands are dropped in turn and re-decoded. Catches ghost bands from over-segmentation.

---

## Resistance Calculation

| Type | Formula |
|------|---------|
| 4-band | `(D1×10 + D2) × Multiplier ± Tolerance%` |
| 5-band | `(D1×100 + D2×10 + D3) × Multiplier ± Tolerance%` |

### Color Code

| Color | Digit | Multiplier | Tolerance |
|-------|-------|------------|-----------|
| Black | 0 | ×1 | — |
| Brown | 1 | ×10 | ±1% |
| Red | 2 | ×100 | ±2% |
| Orange | 3 | ×1 kΩ | — |
| Yellow | 4 | ×10 kΩ | — |
| Green | 5 | ×100 kΩ | ±0.5% |
| Blue | 6 | ×1 MΩ | ±0.25% |
| Violet | 7 | ×10 MΩ | ±0.1% |
| Grey | 8 | ×100 MΩ | — |
| White | 9 | ×1 GΩ | — |
| Gold | — | ×0.1 | ±5% |
| Silver | — | ×0.01 | ±10% |

---

## Limitations

- 6-band resistors not supported
- SMD (surface-mount) resistors not supported
- Grey and silver are hardest to segment (IoU 0.564 / 0.660)
- 1 MΩ 5-band resistors are the most common failure case
- Area and gap thresholds are tuned for the model's output resolution; re-tuning needed for different crop sizes
- No black-edge artifact recovery — returns `error_black_edge` rather than attempting to drop the offending band and re-decode

---

## Error Reference

| Error type | Cause | Recovery |
|------------|-------|----------|
| `INSUFFICIENT_BANDS` | Fewer than 3 bands survive filtering | Improve image quality / lighting |
| `error_black_edge` | Black at position 0 or N-1 (invalid as D1 or tolerance) | Segmentation artifact — retry |
| `error_multiple_tolerance_bands` | >2 gold/silver, or 2 not adjacent at one end | Segmentation artifact — retry |
| `error_interior_tolerance_band` | Single gold/silver at a strictly interior slot | Segmentation artifact — retry |
| `INVALID_TOLERANCE_COLOR` | Resolved tolerance band is orange/yellow/white/etc. | Segmentation artifact — retry |
| `UNKNOWN_DIRECTION` | Direction undetermined after full cascade | Retry with clearer image |
| `INVALID_COLOR` | Color not in lookup table | Check segmentation output |

---

## Tunable Parameters

### Band Extraction

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MIN_BAND_AREA` | 40 px | Absolute pixel floor per band |
| Histogram bins N | 200 | Projection resolution |
| Gaussian σ | 2.5 bins | Smoothing to fill intra-band gaps |
| Min run length δ | 4 bins | Minimum contiguous run to keep |
| Relative area floor | 30% of top-5 median | Drops small spurious fragments |
| Band cap | 5 | Max bands retained |

### Reading Direction

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `GAP_RATIO_THRESHOLD` | 1.5 | End gap must exceed 1.5× median interior gap to qualify |

### Validation

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `DEFAULT_TOLERANCE` | 20% | Assumed tolerance for 3-band resistors |
| E24 mismatch threshold | 1% | Tolerance for E24 value matching |
| Min reasonable value | 0.1 Ω | Sanity lower bound |
| Max reasonable value | 100 MΩ | Sanity upper bound |


