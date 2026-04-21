# Resistor Color Code Reader - Limitations

This document outlines the current limitations of the resistor color code reading system.

---

## Tolerance Band Support

| Status | Tolerance Band | Tolerance Value |
|--------|----------------|-----------------|
| Supported | Gold | ±5% |
| Not Supported | Silver | ±10% |

**Note:** The system requires a **gold tolerance band** to determine the correct reading direction. Resistors with silver tolerance bands are not currently supported and will return an error.

---

## Resistor Types

| Status | Type | Description |
|--------|------|-------------|
| Supported | 4-band | Standard resistors (2 digits + multiplier + tolerance) |
| Supported | 5-band | Precision resistors (3 digits + multiplier + tolerance) |
| Not Supported | 6-band | Temperature coefficient resistors |
| Not Supported | SMD | Surface-mount device codes |

---

## Segmentation Model

### Black Band Visualization
- The segmentation model outputs masks on a **black background**
- Black color bands (digit 0) are rendered as **pink/purple** (RGB: 255, 105, 180) for visibility
- The resistance calculation correctly maps these back to "black"

### Model Accuracy
- Model performance depends on:
  - Image quality and resolution
  - Lighting conditions
  - Resistor orientation in frame
  - Color accuracy of the camera

### Supported Colors
| Class ID | Color | Digit Value | As Multiplier |
|----------|-------|-------------|---------------|
| 8 | Black | 0 | ×1 |
| 4 | Brown | 1 | ×10 |
| 11 | Red | 2 | ×100 |
| 2 | Orange | 3 | ×1,000 |
| 7 | Yellow | 4 | ×10,000 |
| 3 | Green | 5 | ×100,000 |
| 6 | Blue | 6 | ×1,000,000 |
| 12 | Violet | 7 | ×10,000,000 |
| 10 | Grey | 8 | ×100,000,000 |
| 9 | White | 9 | ×1,000,000,000 |
| 1 | Gold | N/A | ×0.1 (tolerance only) |

---

## Image Requirements

### Orientation
- Resistors can be captured at any angle (horizontal, vertical, diagonal)
- The PCA-based axis detection handles arbitrary orientations

### Quality Requirements
- Clear visibility of all color bands
- Minimal glare or reflections
- Sufficient contrast between bands and resistor body
- Recommended minimum resolution: 256×256 pixels

---

## Known Issues

1. **Fragmented Segmentation**: Some bands may be segmented into multiple components
   - Mitigation: Nearby same-color components are automatically merged

2. **Color Confusion**: Similar colors may be misclassified
   - Brown vs. Red
   - Orange vs. Yellow
   - Gold vs. Yellow

3. **Missing Bands**: Poor lighting or image quality may cause bands to be missed
   - System requires minimum 3 bands for calculation

---

## Future Improvements

- [ ] Add silver tolerance band support
- [ ] Support 6-band resistors with temperature coefficient
- [ ] Improve color differentiation for similar colors
- [ ] Add confidence scores for band detection
- [ ] Support multiple resistors in single image
