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

1. Segmentation mask → connected components per color class
2. Noise filter: drops regions below pixel floor or below 15% of median band area
3. Fragment merge: nearby same-color components merged into one band
4. PCA on band centroids → principal axis
5. Project centroids onto axis → sort left-to-right
6. **Direction heuristic:**
   - 4-band: gold/silver position determines tolerance end
   - 5-band ambiguous: largest inter-band gap → tolerance band is on that side

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
- 1MΩ 5-band resistors are the most common failure case


