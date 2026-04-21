# Band Extraction and Resistance Calculation — Reference Tables

This document collects the reference tables for the post-segmentation pipeline:
error-mitigation techniques, tunable parameters, the reading-direction decision
chain, the secondary heuristics, current limitations, input requirements, and
planned improvements.

---

## Table 1 — Post-Segmentation Error-Mitigation Techniques

| # | Technique | Artifact Addressed | How It Works | Stage |
|---|-----------|--------------------|--------------|-------|
| 1 | Two-tier area filtering | Pixel-level noise | Discards regions below a strict pixel-count threshold; uses a lenient threshold during initial extraction to preserve fragments that may merge later | Extraction |
| 2 | Same-color fragment merging | Split bands (highlights, shadows) | Merges same-color components whose centroids are within a distance threshold using area-weighted centroid averaging | Extraction |
| 3 | Axis-based clustering | Cross-color bleed at band boundaries | Projects all region centroids onto the resistor axis and groups regions whose projections fall within a gap threshold | Extraction |
| 4 | Majority voting | Color confusion (brown/red, orange/gold) | Within each axis cluster, the color with the largest total pixel area wins; minority-color fragments are discarded | Extraction |
| 5 | Dual axis estimation | Noisy centroid positions | First estimates the axis from band mask orientations via image moments, then refines it with PCA on resolved band centroids | Extraction |
| 6 | Band count capping | Excessive false detections | Retains only the largest bands by area when more than six regions are detected | Calculation |
| 7 | Multi-level direction heuristics | Tolerance-band localization without relying on color alone | Cascading decision chain (Table 3): gold/silver at an edge → color-based direction; else → tolerance-gap heuristic (Table 4, order 1); if still ambiguous → secondary heuristics (black-edge error, band-width ratio). Both-ends-tolerance and black-edge cases are flagged as segmentation artifacts. | Calculation |
| 8 | Band count adaptation | Missing or extra bands | Routes to 3-, 4-, or 5-band formulas based on detected count; assumes default tolerance for 3-band case | Calculation |
| 9 | E24 series validation | Catch-all sanity check | Normalizes the result and compares against the E24 standard series; flags values outside the typical resistor range | Calculation |

---

## Table 2 — Tunable Parameters

### Band Extraction

| # | Parameter | Value | Module | Purpose |
|---|-----------|-------|--------|---------|
| 1 | `MIN_BAND_AREA` | 40 px | `band_extractor` | Minimum pixels to accept a band |
| 2 | `MIN_BAND_AREA / 2` | 20 px | `band_extractor` | Lenient threshold for initial raw extraction |
| 3 | `MERGE_DISTANCE_THRESHOLD` | 10.0 px | `band_extractor` | Max centroid distance to merge same-color fragments |
| 4 | `BAND_AXIS_THRESHOLD` | 15.0 px | `band_extractor` | Max gap along axis to group regions into same band |

### Band Filtering

| # | Parameter | Value | Module | Purpose |
|---|-----------|-------|--------|---------|
| 5 | `max_bands` (overflow) | 5 | `resistance_calculator` | Cap on bands kept when > 6 detected |
| 6 | `max_bands` (default) | 6 | `band_extractor` | Default cap in area-based filter |

### Reading Direction

| # | Parameter | Value | Module | Purpose |
|---|-----------|-------|--------|---------|
| 7 | `GAP_RATIO_THRESHOLD` | 1.5 | `resistance_calculator` | Multiple of median interior gap required to flag an end as the tolerance side |

### Validation

| # | Parameter | Value | Module | Purpose |
|---|-----------|-------|--------|---------|
| 8 | `DEFAULT_TOLERANCE` | 20.0% | `color_code_tables` | Assumed tolerance when no tolerance band visible |
| 9 | E24 mismatch threshold | 0.1 (10%) | `resistance_calculator` | Relative distance from nearest E24 value to flag warning |
| 10 | Min reasonable value | 0.1 Ω | `resistance_calculator` | Lower bound for sanity check |
| 11 | Max reasonable value | 100 MΩ | `resistance_calculator` | Upper bound for sanity check |

### Preprocessing

| # | Parameter | Value | Module | Purpose |
|---|-----------|-------|--------|---------|
| 12 | `img_size` | 256×256 | `run_inference` | Input image resolution to model |

---

## Table 3 — Reading Direction Decision Chain

| Priority | Condition | Result |
|----------|-----------|--------|
| 0  | Gold/silver count > 2, OR count == 2 but not adjacent at one end (i.e. not at positions [0,1] nor at [N-2, N-1]) | `error_multiple_tolerance_bands` (flagged as segmentation artifact) |
| 1a | Last band is gold or silver | `forward` |
| 1b | First band is gold or silver | `reverse` |
| 2  | No gold/silver at either edge (and count ≤ 1 by Priority 0) | → proceed to gap heuristic (2a) |
| 2a | → gap heuristic (Table 4, order 1) non-ambiguous | use returned direction |
| 2b | → gap heuristic ambiguous | → proceed to secondary heuristics (Table 4, order 2 + default) |
| 3  | All heuristics exhausted (default branch) | `forward` (best-effort default) |

**Note on Priority 0:** Gold and silver are dual-purpose colors — both valid tolerance colors AND valid multipliers (gold ×0.1, silver ×0.01). A sub-ohm resistor legally has gold/silver at BOTH the multiplier position (N-2) and the tolerance position (N-1): e.g. `[brown, grey, gold(×0.1), gold(±5%)]` = 1.8 Ω ±5%. Priority 0 therefore permits `count == 2` when the two tolerance-colored bands are the last two positions (or the first two, if the sequence is reversed), and only flags other arrangements as impossible.

---

## Table 4 — Secondary Direction Heuristics

Order 1 is the primary heuristic, triggered by Priority 2 in Table 3 (no gold/silver at an edge). Order 2 is the fallback, triggered by Priority 2b when the gap heuristic is ambiguous. **The orders are evaluated in sequence; each order that returns a direction or error short-circuits the chain.**

| Order | Heuristic | Rationale |
|-------|-----------|-----------|
| 1 | **Tolerance-gap:** Compare each end gap against `GAP_RATIO_THRESHOLD` × median interior gap (using PCA projections). If the last gap qualifies and the first does not, return `forward`; if the first qualifies and the last does not, return `reverse`; if neither qualifies, return `ambiguous` and fall to order 2. **Requires ≥4 bands** (i.e. ≥3 inter-band gaps, so that at least one interior gap exists to compute a median). With fewer than 4 bands this heuristic returns `ambiguous` unconditionally. | On physical resistors the tolerance band is deliberately spaced further from the nearest digit band than digit bands are from each other. This geometric cue is independent of color and band width, making it more robust than purely color-based heuristics. |
| 2 | **Black-edge check:** If black appears at either end of the sequence, return the `BLACK_BOUNDARY_BAND` error. | Black encodes digit 0 (invalid as the first significant digit) and is not a valid tolerance color (cannot be the last band). Reversing does not resolve either case, so black at a boundary is flagged as a segmentation artifact rather than a direction signal. |
| — | *Default* | If no order triggers, return `forward`. |

**Removed:** A prior *band-width ratio* heuristic (return `reverse` if the first band's bounding-box width was < 70 % of the last band's) was dropped because tolerance bands are not reliably thinner than digit bands in real-world segmentations, and the heuristic produced wrong directions more often than right ones.

---

## Table 5 — Current Limitations

| # | Limitation | Description | Impact |
|---|------------|-------------|--------|
| 1 | Fixed axis-clustering threshold | The gap threshold for grouping regions into band axes is a single global constant (15 px at 256×256). | Resistors with unusually narrow or wide band spacing may be over-merged or under-merged. |
| 2 | No 6-band resistor support | The calculator handles 3-, 4-, and 5-band resistors; 6-band resistors (with a temperature coefficient band) are not decoded. | The sixth band is either discarded by the cap or causes the pipeline to truncate to five bands. |
| 3 | Single-resistor assumption | The extraction stage assumes all detected band regions belong to a single resistor. | If the mask contains bands from a second resistor or band-like background clutter, spurious bands contaminate the sequence. |
| 4 | No confidence propagation | The extraction stage does not propagate per-pixel softmax confidence from the segmentation model. | Low-confidence pixels are treated identically to high-confidence ones during majority voting and area counting. |
| 5 | Resolution dependence | Area and distance thresholds are tuned for 256×256 input crops and do not scale with image size. | Running on higher- or lower-resolution crops without re-tuning parameters may degrade extraction accuracy. |
| 6 | No recovery from black-edge artifacts | When black is detected at either boundary of the sequence, the pipeline returns `BLACK_BOUNDARY_BAND` (Table 4, order 2) because neither reading direction is valid. | Images where segmentation places a spurious black region at the resistor's edge produce no reading. A future improvement could attempt to drop the offending edge region and re-decode rather than failing outright. |
| 7 | No SMD resistor support | The pipeline is designed for through-hole axial resistors with color bands. Surface-mount resistors use numeric codes, not colors. | SMD components in the input image cannot be decoded. |

---

## Input Requirements

| Requirement | Expectation |
|-------------|-------------|
| Orientation | Any angle (horizontal, vertical, diagonal); PCA handles arbitrary orientations |
| Visibility | All color bands clearly visible, minimal glare/reflections |
| Contrast | Sufficient contrast between bands and resistor body |
| Resolution | Minimum 256×256 px for the resistor crop |
| Scene | One resistor per crop (single-resistor assumption — see Table 5, item 4) |

---

## Future Improvements

- 6-band resistor support (temperature coefficient band)
- Per-pixel confidence propagation from segmentation into band-level voting
- Black-edge artifact recovery (drop offending region and re-decode instead of returning an error)
- Adaptive thresholds that scale with input resolution
