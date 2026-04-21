# Technical Documentation

This document explains how the resistor color code reader works, including the model architecture and each script's functionality.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Model Architecture](#model-architecture)
3. [Scripts Reference](#scripts-reference)
4. [Data Flow](#data-flow)
5. [Code Architecture](#code-architecture)
6. [Algorithm Details](#algorithm-details)

---

## System Overview

The system uses a deep learning segmentation model combined with computer vision algorithms to:

1. **Segment** resistor images into color band regions
2. **Extract** individual bands using connected component analysis
3. **Order** bands using PCA-based axis detection
4. **Calculate** resistance using standard color code rules

### Pipeline Diagram

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│ Input Image │───▶│ U-Net Model  │───▶│ Segmentation  │
│  (256×256)  │    │ (13 classes) │    │     Mask      │
└─────────────┘    └──────────────┘    └───────┬───────┘
                                               │
                                               ▼
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│ Resistance  │◀───│  Band Order  │◀───│ Connected     │
│   Value     │    │   (PCA)      │    │  Components   │
└─────────────┘    └──────────────┘    └───────────────┘
```

---

## Model Architecture

### File: `models/resistor_unet.keras`

The segmentation model is a **U-Net** architecture trained to identify 13 classes:

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | Background | Image background |
| 1 | Gold | Tolerance band (±5%) |
| 2 | Orange | Digit 3 / ×1000 |
| 3 | Green | Digit 5 / ×100k |
| 4 | Brown | Digit 1 / ×10 |
| 5 | background | Resistor body (ignored) |
| 6 | Blue | Digit 6 / ×1M |
| 7 | Yellow | Digit 4 / ×10k |
| 8 | Black | Digit 0 / ×1 |
| 9 | White | Digit 9 / ×1G |
| 10 | Grey | Digit 8 / ×100M |
| 11 | Red | Digit 2 / ×100 |
| 12 | Violet | Digit 7 / ×10M |

### Model Specifications

- **Input Shape**: (256, 256, 3) - RGB image
- **Output Shape**: (256, 256, 13) - Per-pixel class probabilities
- **Architecture**: U-Net (encoder-decoder with skip connections)
- **File Size**: ~90 MB

### Visualization Note

Black bands (class 8) are rendered as **pink** (RGB: 255, 105, 180) in the visualization to make them visible against the black background. The calculation logic correctly interprets these as "black".

---

## Scripts Reference

### 1. `run_inference.py`

**Purpose**: Main entry point for processing resistor images.

**Functions**:

| Function | Description |
|----------|-------------|
| `load_model(path)` | Loads the Keras model from disk |
| `preprocess_image(path)` | Resizes image to 256×256, normalizes to [0,1] |
| `ids_to_color(mask)` | Converts class IDs to RGB visualization |
| `predict_and_visualize(...)` | Full pipeline: predict, calculate, save |
| `main()` | Processes all images in `data/raw_images/` |

**Usage**:
```python
python scripts/run_inference.py
```

**Output**: Saves results to `data/inference_results/` with format:
- Original image | Segmentation mask | Overlay with resistance value

---

### 2. `color_code_tables.py`

**Purpose**: Contains all lookup tables and data structures.

**Constants**:

```python
# Class ID to color name
ID_TO_COLOR_NAME = {
    0: "background", 1: "gold", 2: "orange", ...
}

# Color to digit value (0-9)
COLOR_TO_DIGIT = {
    "black": 0, "brown": 1, "red": 2, ...
}

# Color to multiplier
COLOR_TO_MULTIPLIER = {
    "black": 1, "brown": 10, "red": 100, ...
}

# Color to tolerance percentage
COLOR_TO_TOLERANCE = {
    "gold": 5.0, "brown": 1.0, ...
}
```

**Data Classes**:

```python
@dataclass
class BandInfo:
    class_id: int                    # Segmentation class (0-12)
    color_name: str                  # Human-readable name
    centroid: Tuple[float, float]    # (x, y) center position
    area: int                        # Pixel count
    bounding_box: Tuple[int, int, int, int]  # x_min, y_min, x_max, y_max

@dataclass
class ResistanceResult:
    value: float        # Resistance in Ohms
    tolerance: float    # Tolerance percentage
    band_count: int     # 4 or 5 bands
    bands: List[BandInfo]  # Detected bands in order

@dataclass
class CalculationError:
    error_type: str     # Error category
    message: str        # Human-readable description
    detected_bands: List[BandInfo]  # What was found
```

**Utility Functions**:

| Function | Description |
|----------|-------------|
| `format_resistance(value)` | Formats with units (Ω, kΩ, MΩ) |
| `get_color_name(class_id)` | Looks up color from class ID |
| `is_tolerance_color(name)` | Checks if color is a tolerance band |

---

### 3. `band_extractor.py`

**Purpose**: Extracts individual color bands from segmentation masks.

**Algorithm**:

1. For each color class (excluding backgrounds):
   - Create binary mask for that class
   - Apply connected component labeling (`scipy.ndimage.label`)
   - Filter components by minimum area (50 pixels)
   - Calculate centroid and bounding box for each component

2. Merge nearby components of same color (handles fragmented segmentation)

**Key Functions**:

```python
def extract_color_bands(mask: np.ndarray) -> List[BandInfo]:
    """
    Extract bands using connected component analysis.

    Args:
        mask: 2D array (256, 256) with class IDs

    Returns:
        List of BandInfo objects for each detected band
    """

def merge_nearby_components(bands: List[BandInfo],
                            threshold: float = 30.0) -> List[BandInfo]:
    """
    Merge fragments of the same color that are close together.
    Uses weighted centroid averaging.
    """
```

**Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_BAND_AREA` | 50 | Minimum pixels to be considered a band |
| `MERGE_DISTANCE_THRESHOLD` | 30.0 | Max distance to merge same-color components |

---

### 4. `axis_detector.py`

**Purpose**: Determines resistor orientation and sorts bands along its axis.

**Algorithm** (PCA-based):

1. Collect centroids of all detected bands
2. Compute covariance matrix of centroid positions
3. Find eigenvector with largest eigenvalue (principal axis)
4. Project all band centroids onto this axis
5. Sort bands by their projection values

**Why PCA?**
- Handles any orientation (horizontal, vertical, diagonal)
- Robust to outliers
- Works with as few as 2 bands

**Key Functions**:

```python
def compute_principal_axis(centroids: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use PCA to find the resistor's main axis.

    Returns:
        axis_vector: Unit vector along resistor
        axis_origin: Mean point of centroids
    """

def sort_bands_by_position(bands: List[BandInfo]) -> List[BandInfo]:
    """
    Main entry point: Sort bands along the resistor axis.
    """
```

**Example**:

```
Input centroids: [(50,100), (150,100), (100,100), (200,100)]
                 (unordered)

PCA axis: [1.0, 0.0] (horizontal)

Projections: [-50, 50, 0, 100]

Sorted output: [(50,100), (100,100), (150,100), (200,100)]
               Band 1    Band 2     Band 3     Band 4
```

---

### 5. `resistance_calculator.py`

**Purpose**: Orchestrates the full calculation pipeline.

**Main Function**:

```python
def calculate_resistance(mask: np.ndarray) -> Union[ResistanceResult, CalculationError]:
    """
    Convert segmentation mask to resistance value.

    Pipeline:
    1. Extract bands (band_extractor)
    2. Sort by position (axis_detector)
    3. Determine reading direction (gold band location)
    4. Calculate resistance (4-band or 5-band formula)
    """
```

**Reading Direction Logic**:

```python
def determine_reading_direction(sorted_bands: List[BandInfo]) -> str:
    """
    Determine correct reading direction using a gated heuristic cascade.

    Returns:
        "forward"                 - Bands are in correct order
        "reverse"                 - Need to reverse the order
        "error_unknown_direction" - Direction could not be determined
        "error_black_edge"        - Black band at a boundary (segmentation artifact)
    """
```

Decision tree:
```
Gold/silver at last end only?  ──Yes──▶ "forward" ✓
        │
        No
        ▼
Gold/silver at first end only? ─Yes──▶ "reverse" (flip order)
        │
        No
        ▼
Gold/silver at both ends?      ─Yes──▶ "error_both_ends_tolerance"
        │                               (physically impossible —
        │                               flagged as segmentation artifact)
        No (no tolerance color at an edge, or both ends non-tolerance)
        ▼
Gap heuristic (requires ≥4 bands):
end-gap ≥ 1.5 × median interior gap?
        │
    Yes (one end qualifies) ──────────▶ direction from larger end-gap
        │
    No (ambiguous, or <4 bands)
        ▼
Apply secondary heuristics (Table 4 orders 2–3):
  black-edge → "error_black_edge" (short-circuits)
  width-ratio → "reverse" if first <70% of last
  else → "forward"
```

**Secondary Heuristics** (when gap heuristic is ambiguous):
1. Black-edge check: If black appears at either end of the sorted band sequence, return the `BLACK_BOUNDARY_BAND` error — black is invalid as the first significant digit (leading zero) and is not a valid tolerance color (valid tolerance colors: gold, silver, brown, red, green, blue, violet, grey). Reversing does not fix either case, so black at a boundary is treated as a segmentation artifact.
2. Band-width ratio: If the first band is significantly thinner than the last (<70% of last's width), reverse the sequence — tolerance bands tend to be thinner than digit bands.

**Calculation Formulas**:

```python
# 4-band resistor: [D1][D2][Mult][Tol]
resistance = (D1 × 10 + D2) × Multiplier

# 5-band resistor: [D1][D2][D3][Mult][Tol]
resistance = (D1 × 100 + D2 × 10 + D3) × Multiplier
```

**Example**:
```
Bands: Brown → Black → Red → Gold
       D1=1    D2=0    M=100  T=5%

Calculation: (1×10 + 0) × 100 = 1000 Ω = 1 kΩ ±5%
```

---

## Data Flow

### Complete Pipeline

```
1. Image Loading
   ┌─────────────────────────────────────────────┐
   │ PIL.Image.open() → resize(256,256) → /255.0 │
   └─────────────────────────────────────────────┘
                          │
                          ▼
2. Model Inference
   ┌─────────────────────────────────────────────┐
   │ model.predict() → (256,256,13) → argmax()   │
   │ Output: 256×256 array of class IDs (0-12)   │
   └─────────────────────────────────────────────┘
                          │
                          ▼
3. Band Extraction
   ┌─────────────────────────────────────────────┐
   │ For each class: scipy.ndimage.label()       │
   │ Filter by area > 50 pixels                  │
   │ Output: List of BandInfo objects            │
   └─────────────────────────────────────────────┘
                          │
                          ▼
4. Axis Detection & Sorting
   ┌─────────────────────────────────────────────┐
   │ PCA on centroids → principal axis           │
   │ Project bands onto axis → sort              │
   │ Output: Ordered list of bands               │
   └─────────────────────────────────────────────┘
                          │
                          ▼
5. Reading Direction
   ┌─────────────────────────────────────────────┐
   │ Find gold band position                     │
   │ If gold at start: reverse order             │
   │ Output: Correctly ordered bands             │
   └─────────────────────────────────────────────┘
                          │
                          ▼
6. Resistance Calculation
   ┌─────────────────────────────────────────────┐
   │ Look up digit values from colors            │
   │ Apply formula: (digits) × multiplier        │
   │ Output: ResistanceResult                    │
   └─────────────────────────────────────────────┘
```

---

## Code Architecture

This section details the complete call hierarchy and shows exactly where each function is defined and called.

### Entry Point

When you run `python scripts/run_inference.py`, execution begins at:

```python
# run_inference.py, line 242-243
if __name__ == "__main__":
    main()
```

### Complete Call Hierarchy

```
python scripts/run_inference.py
│
└── main()                                          # run_inference.py:152
    │
    ├── load_model(model_path)                      # run_inference.py:166
    │   └── tf.keras.models.load_model()            # Load U-Net from disk
    │
    └── for each image file:
        │
        └── predict_and_visualize(model, image_path)  # run_inference.py:192
            │
            ├── preprocess_image(image_path)          # run_inference.py:94
            │   ├── PIL.Image.open()                  # Load image file
            │   ├── img.resize((256, 256))            # Resize to model input
            │   └── np.array() / 255.0                # Normalize to [0,1]
            │
            ├── model.predict(img_array)              # run_inference.py:97
            │   └── U-Net forward pass                # Returns (256,256,13)
            │
            ├── np.argmax(pred, axis=-1)              # run_inference.py:98
            │   └── Convert probabilities → class IDs # Returns (256,256)
            │
            ├── ids_to_color(pred_ids)                # run_inference.py:101
            │   └── Map class IDs → RGB colors        # For visualization
            │
            ├── calculate_resistance(pred_ids)        # run_inference.py:110
            │   │                                     # ↓ resistance_calculator.py:25
            │   │
            │   ├── extract_color_bands(mask)         # resistance_calculator.py:42
            │   │   │                                 # ↓ band_extractor.py:27
            │   │   │
            │   │   ├── for each class_id in BAND_CLASS_IDS:
            │   │   │   ├── binary_mask = (mask == class_id)
            │   │   │   └── ndimage.label(binary_mask)  # Connected components
            │   │   │
            │   │   └── merge_nearby_components(bands)  # band_extractor.py:89
            │   │       └── Merge fragments of same color
            │   │
            │   ├── sort_bands_by_position(bands)     # resistance_calculator.py:57
            │   │   │                                 # ↓ axis_detector.py:114
            │   │   │
            │   │   ├── centroids = [b.centroid for b in bands]
            │   │   │
            │   │   ├── compute_principal_axis(centroids)  # axis_detector.py:133
            │   │   │   │                                  # ↓ axis_detector.py:14
            │   │   │   ├── np.cov(centered.T)             # Covariance matrix
            │   │   │   └── np.linalg.eig(cov_matrix)      # Eigendecomposition
            │   │   │
            │   │   └── project_and_sort_bands()      # axis_detector.py:136
            │   │       │                             # ↓ axis_detector.py:84
            │   │       ├── projection = dot(centroid, axis_vector)
            │   │       └── sort by projection value
            │   │
            │   ├── determine_reading_direction(sorted_bands)  # resistance_calculator.py:60
            │   │   │                                          # ↓ resistance_calculator.py:87
            │   │   ├── Check if last band is gold → "forward"
            │   │   ├── Check if first band is gold → "reverse"
            │   │   └── No gold found → "error"
            │   │
            │   └── calculate_4_band_resistance(bands)  # resistance_calculator.py:76
            │       │                                   # ↓ resistance_calculator.py:161
            │       ├── digit1 = COLOR_TO_DIGIT[color1]
            │       ├── digit2 = COLOR_TO_DIGIT[color2]
            │       ├── multiplier = COLOR_TO_MULTIPLIER[color3]
            │       ├── resistance = (digit1*10 + digit2) * multiplier
            │       └── return ResistanceResult(value, tolerance, ...)
            │
            ├── plt.subplots(1, 3)                    # run_inference.py:121
            │   └── Create 3-panel figure
            │
            ├── plt.savefig(output_path)              # run_inference.py:142
            │   └── Save visualization to PNG
            │
            └── return rgb_mask, pred_ids, colors, resistance_result
```

### Function Location Reference

| Function | File | Line | Called By |
|----------|------|------|-----------|
| `main()` | run_inference.py | 152 | `__main__` |
| `load_model()` | run_inference.py | 62 | `main()` |
| `preprocess_image()` | run_inference.py | 69 | `predict_and_visualize()` |
| `ids_to_color()` | run_inference.py | 77 | `predict_and_visualize()` |
| `predict_and_visualize()` | run_inference.py | 89 | `main()` |
| `calculate_resistance()` | resistance_calculator.py | 25 | `predict_and_visualize()` |
| `extract_color_bands()` | band_extractor.py | 27 | `calculate_resistance()` |
| `merge_nearby_components()` | band_extractor.py | 89 | `extract_color_bands()` |
| `sort_bands_by_position()` | axis_detector.py | 114 | `calculate_resistance()` |
| `compute_principal_axis()` | axis_detector.py | 14 | `sort_bands_by_position()` |
| `project_and_sort_bands()` | axis_detector.py | 84 | `sort_bands_by_position()` |
| `determine_reading_direction()` | resistance_calculator.py | 87 | `calculate_resistance()` |
| `apply_secondary_heuristics()` | resistance_calculator.py | 131 | `determine_reading_direction()` |
| `calculate_4_band_resistance()` | resistance_calculator.py | 161 | `calculate_resistance()` |
| `calculate_5_band_resistance()` | resistance_calculator.py | 225 | `calculate_resistance()` |
| `calculate_3_band_resistance()` | resistance_calculator.py | 281 | `calculate_resistance()` |

### Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ run_inference.py                                                            │
│                                                                             │
│  Imports:                                                                   │
│  ├── tensorflow (model loading & inference)                                │
│  ├── PIL.Image (image loading)                                             │
│  ├── matplotlib.pyplot (visualization)                                     │
│  ├── resistance_calculator (calculate_resistance)                          │
│  └── validate_results (validation functions)                               │
│                                                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ resistance_calculator.py                                                    │
│                                                                             │
│  Imports:                                                                   │
│  ├── color_code_tables (lookup tables, data classes)                       │
│  ├── band_extractor (extract_color_bands)                                  │
│  └── axis_detector (sort_bands_by_position)                                │
│                                                                             │
└───────────────────┬─────────────────────────────┬───────────────────────────┘
                    │                             │
                    ▼                             ▼
┌───────────────────────────────┐ ┌───────────────────────────────────────────┐
│ band_extractor.py             │ │ axis_detector.py                          │
│                               │ │                                           │
│  Imports:                     │ │  Imports:                                 │
│  ├── numpy                    │ │  ├── numpy                                │
│  ├── scipy.ndimage            │ │  └── color_code_tables (BandInfo)        │
│  └── color_code_tables        │ │                                           │
│                               │ │                                           │
└───────────────────────────────┘ └───────────────────────────────────────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ color_code_tables.py                                                        │
│                                                                             │
│  Contains:                                                                  │
│  ├── ID_TO_COLOR_NAME (class ID → color name mapping)                      │
│  ├── COLOR_TO_DIGIT (color → digit value)                                  │
│  ├── COLOR_TO_MULTIPLIER (color → multiplier)                              │
│  ├── COLOR_TO_TOLERANCE (color → tolerance %)                              │
│  ├── BandInfo (dataclass)                                                  │
│  ├── ResistanceResult (dataclass)                                          │
│  └── CalculationError (dataclass)                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Code Sections

#### 1. Preprocessing (run_inference.py:69-75)

```python
def preprocess_image(image_path, img_size=(256, 256)):
    """Load and preprocess an image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, 0)  # Add batch dimension
    return img_array, img
```

#### 2. Model Inference (run_inference.py:97-98)

```python
pred = model.predict(img_array, verbose=0)[0]  # Shape: (256, 256, 13)
pred_ids = np.argmax(pred, axis=-1)             # Shape: (256, 256)
```

#### 3. Band Extraction (band_extractor.py:50-51)

```python
# Find connected components for each color class
labeled_array, num_features = ndimage.label(binary_mask)
```

#### 4. PCA Axis Detection (axis_detector.py:38-52)

```python
# Compute covariance matrix
cov_matrix = np.cov(centered.T)

# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Principal axis is eigenvector with largest eigenvalue
principal_idx = np.argmax(eigenvalues)
axis_vector = eigenvectors[:, principal_idx].real
```

#### 5. Reading Direction (resistance_calculator.py:108-114)

```python
first_is_gold = first_band.color_name.lower() == "gold"
last_is_gold = last_band.color_name.lower() == "gold"

if last_is_gold and not first_is_gold:
    return "forward"
elif first_is_gold and not last_is_gold:
    return "reverse"
```

#### 6. Resistance Calculation (resistance_calculator.py:213-222)

```python
# Calculate resistance value
base_value = digit1 * 10 + digit2
resistance = base_value * multiplier

return ResistanceResult(
    value=resistance,
    tolerance=tolerance,
    band_count=4,
    bands=bands[:4]
)
```

### Example Walkthrough

**Input:** `1kohm.heic` (Brown-Black-Red-Gold resistor)

| Step | Function | Action | Result |
|------|----------|--------|--------|
| 1 | `preprocess_image()` | Load & normalize | (1, 256, 256, 3) tensor |
| 2 | `model.predict()` | U-Net inference | (256, 256, 13) probabilities |
| 3 | `np.argmax()` | Get class IDs | Mask with values 4, 8, 11, 1 |
| 4 | `extract_color_bands()` | Connected components | 4 BandInfo objects |
| 5 | `sort_bands_by_position()` | PCA + projection | Ordered: brown→black→red→gold |
| 6 | `determine_reading_direction()` | Check gold position | "forward" (gold at end) |
| 7 | `calculate_4_band_resistance()` | Apply formula | (1×10 + 0) × 100 = **1000 Ω** |
| 8 | Return | Format result | **1 kΩ ±5%** |

---

## Algorithm Details

### Connected Component Labeling

Uses `scipy.ndimage.label()` which implements 8-connectivity:

```
Neighbors checked:
  ┌───┬───┬───┐
  │ X │ X │ X │
  ├───┼───┼───┤
  │ X │ P │ X │
  ├───┼───┼───┤
  │ X │ X │ X │
  └───┴───┴───┘
```

Each connected region gets a unique label. Properties (centroid, area, bbox) are computed from labeled regions.

### PCA for Orientation

Given band centroids as points in 2D:

1. **Center the data**: Subtract mean
2. **Covariance matrix**:
   ```
   Σ = (1/n) × Xᵀ × X
   ```
3. **Eigendecomposition**: Find eigenvectors
4. **Principal axis**: Eigenvector with largest eigenvalue

This gives the direction of maximum variance, which aligns with the resistor axis.

### Projection Sorting

To sort bands along the axis:

```python
projection = dot(band_centroid - axis_origin, axis_vector)
```

The projection value is a scalar representing position along the axis. Sorting by this value orders bands from one end to the other.

---

## Error Handling

### Error Types

| Error | Cause | Recovery |
|-------|-------|----------|
| `INSUFFICIENT_BANDS` | < 3 bands detected | Improve image quality |
| `UNKNOWN_DIRECTION` | No gold/silver at an edge and gap heuristic was ambiguous | Retry with a clearer image; ensure the tolerance band is visible |
| `BLACK_BOUNDARY_BAND` | Black band at first or last position (invalid as leading digit and as tolerance color) | Segmentation artifact — retry with better image |
| `BOTH_ENDS_TOLERANCE` | Tolerance colors (gold/silver) at both ends — physically impossible on a real resistor | Segmentation artifact — retry with better image |
| `INVALID_COLOR` | Color not in lookup table | Check segmentation output |

### Validation

The `validate_result()` function checks:
1. Value is in E24 standard series
2. Value is within typical range (0.1Ω - 100MΩ)

---

## Performance Considerations

- **Model inference**: ~0.5-1s per image (CPU)
- **Band extraction**: ~10ms per image
- **Total pipeline**: ~1s per image

### Memory Usage

- Model: ~90 MB
- Per image: ~2 MB (256×256×3 float32)

---

## Extending the System

### Adding Silver Tolerance Support

1. Update `TOLERANCE_CLASS_IDS` in `color_code_tables.py`:
   ```python
   TOLERANCE_CLASS_IDS = {1, 10}  # Gold and Grey (as silver)
   ```

2. Update `determine_reading_direction()` in `resistance_calculator.py`:
   ```python
   tolerance_colors = {"gold", "grey"}  # Add grey as silver
   ```

### Adding New Color Classes

1. Retrain the segmentation model with new class
2. Update `ID_TO_COLOR_NAME` mapping
3. Add values to `COLOR_TO_DIGIT`, `COLOR_TO_MULTIPLIER`

---

## References

- [Resistor Color Code Standard](https://en.wikipedia.org/wiki/Electronic_color_code)
- [U-Net Architecture](https://arxiv.org/abs/1505.04597)
- [PCA for Orientation Detection](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Connected Component Labeling](https://en.wikipedia.org/wiki/Connected-component_labeling)
