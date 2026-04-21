# Resistor Color Code Reader

An automated system that reads resistor color bands from images and calculates the resistance value.

## Quick Start

### Prerequisites

1. **Python 3.8+** with conda or pip
2. **Required packages:**
   ```bash
   pip install tensorflow numpy scipy matplotlib pillow pillow-heif
   ```

### Usage

1. **Place resistor images** in `data/raw_images/`
   - Supported formats: HEIC, JPG, JPEG, PNG

2. **Run the inference script:**
   ```bash
   python scripts/run_inference.py
   ```

3. **View results** in `data/inference_results/`
   - Each output shows: Original → Segmentation Mask → Overlay with Resistance

---

## Example Output

```
HEIC support enabled!
Loading model from: models/resistor_unet.keras
Model loaded successfully!

Processing 45 images...

[1/45] Processing: 1kohm.heic... Saved! Resistance: 1 kΩ ±5%
[2/45] Processing: 10kohm.heic... Saved! Resistance: 10 kΩ ±5%
[3/45] Processing: 220ohm.heic... Saved! Resistance: 220 Ω ±5%
...

==================================================
Processing complete!
  Successfully calculated resistance: 42/45 images
  Resistance calculation errors: 3
  Results saved to: data/inference_results
```

---

## Project Structure

```
Dataset/
├── data/
│   ├── raw_images/           # Input: Your resistor images
│   ├── inference_results/    # Output: Results with resistance values
│   └── tfrecords/            # Training data (not needed for inference)
├── models/
│   └── resistor_unet.keras   # Pre-trained segmentation model
├── scripts/
│   └── run_inference.py      # Main script to run
├── docs/
│   ├── LIMITATIONS.md        # Known limitations
│   └── DOCUMENTATION.md      # Technical details
└── README.md                 # This file
```

---

## How It Works

1. **Image Input** → Load resistor photo
2. **Segmentation** → U-Net model identifies color bands
3. **Band Detection** → Extract individual bands from mask
4. **Orientation** → Determine resistor axis direction
5. **Reading Order** → Use gold tolerance band to find correct direction
6. **Calculation** → Apply color code rules to get resistance

---

## Supported Resistors

| Type | Bands | Example |
|------|-------|---------|
| Standard | 4-band | Brown-Black-Red-Gold = 1kΩ ±5% |
| Precision | 5-band | Brown-Black-Black-Brown-Brown = 1kΩ ±1% |

---

## Supported Colors

| Color | Digit | Multiplier |
|-------|-------|------------|
| Black | 0 | ×1 |
| Brown | 1 | ×10 |
| Red | 2 | ×100 |
| Orange | 3 | ×1k |
| Yellow | 4 | ×10k |
| Green | 5 | ×100k |
| Blue | 6 | ×1M |
| Violet | 7 | ×10M |
| Grey | 8 | ×100M |
| White | 9 | ×1G |
| **Gold** | — | Tolerance ±5% |

---

## Tips for Best Results

1. **Lighting**: Use even lighting, avoid shadows on bands
2. **Focus**: Ensure color bands are clearly visible
3. **Orientation**: Any angle works (horizontal, vertical, diagonal)
4. **Background**: Plain background helps segmentation
5. **Tolerance Band**: Gold band must be visible for reading direction

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| "No gold tolerance band found" | Gold band not detected | Ensure gold band is visible in image |
| "Need at least 3 bands" | Poor segmentation | Improve lighting/image quality |
| "pillow-heif not installed" | Missing package | Run `pip install pillow-heif` |

---

## Known Limitations

- **Silver tolerance** not yet supported (only gold)
- Requires **gold band visible** for reading direction
- See `docs/LIMITATIONS.md` for full details

---

## License

ECEN 491 Senior Design Project - Texas A&M University at Qatar
