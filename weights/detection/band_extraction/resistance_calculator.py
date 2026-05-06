"""
Resistance Calculator Module

This module orchestrates the full pipeline to convert a segmentation
mask into a resistance value.
"""

import numpy as np
from statistics import median
from typing import List, Union, Optional

from color_code_tables import (
    BandInfo,
    ResistanceResult,
    CalculationError,
    COLOR_TO_DIGIT,
    COLOR_TO_MULTIPLIER,
    COLOR_TO_TOLERANCE,
    DEFAULT_TOLERANCE,
    format_resistance,
)
from band_extractor import extract_color_bands, filter_bands_by_area, extract_color_bands_with_visualization, extract_bands_by_projection, MIN_BAND_AREA
from axis_detector import sort_bands_by_position


def _compute_axis_confidence(bands: List[BandInfo]) -> float:
    """Confidence that the PCA axis is well-defined (0-1)."""
    if len(bands) < 2:
        return 0.0
    points = np.array([b.centroid for b in bands])
    spread = np.std(points, axis=0)
    return float(max(spread) / (np.sum(spread) + 1e-6))


def _compute_area_confidence(mask: np.ndarray, bands: List[BandInfo]) -> float:
    """Ratio of detected band pixels to total non-background pixels (0-1)."""
    non_bg = np.sum(mask > 0)
    if non_bg == 0:
        return 0.0
    return float(min(sum(b.area for b in bands) / non_bg, 1.0))


def _inject_confidence(result, axis_conf: float, area_conf: float):
    """Attach confidence scores to a ResistanceResult (no-op on errors)."""
    if isinstance(result, ResistanceResult):
        result.axis_confidence = axis_conf
        result.area_confidence = area_conf
    return result


# E24 preferred-value series (mantissas 1.0–9.1)
_E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4,
        2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2,
        6.8, 7.5, 8.2, 9.1]


def _is_e24(value: float, tol: float = 0.01) -> bool:
    """Return True if value normalises to within tol of any E24 mantissa."""
    if value <= 0:
        return False
    m = value
    while m >= 10.0:
        m /= 10.0
    while m < 1.0:
        m *= 10.0
    return any(abs(m - e) / e < tol for e in _E24)


def _decode_ordered_bands(bands: List[BandInfo]) -> Union[ResistanceResult, CalculationError]:
    """
    Apply direction detection, leading/trailing strip, and resistance
    calculation to a position-ordered band list.  Returns ResistanceResult
    or CalculationError.  Does NOT mutate the input list.
    """
    bands = list(bands)

    direction = determine_reading_direction(bands)
    _error_map = {
        "error":                          ("NO_TOLERANCE_BAND",           "Could not determine reading direction — no recognisable tolerance band at either end."),
        "error_black_edge":               ("BLACK_BOUNDARY_BAND",         "Black band at a boundary position — invalid as leading digit and as tolerance color."),
        "error_multiple_tolerance_bands": ("MULTIPLE_TOLERANCE_BANDS",    "Impossible gold/silver arrangement — likely a segmentation artifact."),
        "error_interior_tolerance_band":  ("INTERIOR_TOLERANCE_BAND",     "Gold/silver in a strictly interior position — not valid on any real resistor."),
    }
    if direction in _error_map:
        error_type, message = _error_map[direction]
        return CalculationError(error_type=error_type, message=message, detected_bands=bands)
    if direction == "reverse":
        bands = list(reversed(bands))

    _4tol = {'gold', 'silver'}
    while len(bands) > 3 and bands[0].color_name.lower() in _4tol:
        bands = bands[1:]
    while (len(bands) > 4
           and bands[-1].color_name.lower() in _4tol
           and bands[-2].color_name.lower() in _4tol):
        bands = bands[:-1]

    n = len(bands)
    if n == 4:
        return calculate_4_band_resistance(bands)
    elif n == 5:
        return calculate_5_band_resistance(bands)
    elif n == 3:
        return calculate_3_band_resistance(bands)
    else:
        return calculate_5_band_resistance(bands[:5])


def calculate_resistance(mask: np.ndarray, image: np.ndarray = None) -> Union[ResistanceResult, CalculationError]:
    """
    Main entry point: Calculate resistance value from a segmentation mask.

    Pipeline:
    1. Extract color bands via 1D projection
    2. Filter noise (area floor + relative floor)
    3. Decode (direction → strip → calculate)
    4. E24 retry: if result is non-E24, try dropping the smallest band and
       re-decoding — catches hallucinated ghost bands that inflate band count.

    Args:
        mask: 2D numpy array (H, W) with class IDs (0-12)

    Returns:
        ResistanceResult on success, CalculationError on failure
    """
    import statistics

    bands, _ = extract_bands_by_projection(mask, image_rgb=None)

    # Filter noise: absolute floor + relative floor (≥30% of top-5 median)
    if bands:
        top_areas = sorted((b.area for b in bands), reverse=True)[:5]
        ref_area = statistics.median(top_areas)
        bands = [b for b in bands if b.area >= MIN_BAND_AREA and b.area >= 0.30 * ref_area]

    bands = [b for b in bands if b.color_name != "unknown"]

    if len(bands) < 3:
        return CalculationError(
            error_type="INSUFFICIENT_BANDS",
            message=f"Need at least 3 bands, found {len(bands)}",
            detected_bands=bands,
        )

    # Hard cap: keep at most 5 bands (largest by area)
    if len(bands) > 5:
        bands = filter_bands_by_area(bands, max_bands=5)

    sorted_bands = bands  # already position-ordered by projection

    # Primary decode
    result = _decode_ordered_bands(sorted_bands)

    # E24 retry — if the primary result is not a preferred value, a ghost band
    # is likely inflating the count.  Try dropping the smallest band (ghost
    # bands are almost always the smallest) and re-decode.
    if isinstance(result, ResistanceResult) and not _is_e24(result.value) and len(sorted_bands) > 3:
        area_asc = sorted(range(len(sorted_bands)), key=lambda i: sorted_bands[i].area)
        for drop_i in area_asc[:2]:   # try dropping the 1st or 2nd smallest
            candidate = [b for j, b in enumerate(sorted_bands) if j != drop_i]
            alt = _decode_ordered_bands(candidate)
            if isinstance(alt, ResistanceResult) and _is_e24(alt.value):
                result = alt
                sorted_bands = candidate
                break

    return _inject_confidence(result, _compute_axis_confidence(sorted_bands), _compute_area_confidence(mask, bands))


def calculate_resistance_with_axis_info(mask: np.ndarray, image: np.ndarray = None) -> tuple:
    """
    Calculate resistance and return axis visualization info.

    Same as calculate_resistance but also returns axis data for visualization.

    Args:
        mask: 2D numpy array (H, W) with class IDs (0-12)

    Returns:
        Tuple of (result, axis_info) where:
        - result: ResistanceResult or CalculationError
        - axis_info: dict with axis visualization data, or None
    """
    import statistics

    bands, axis_info = extract_bands_by_projection(mask, image_rgb=None)

    if bands:
        median_area = statistics.median(b.area for b in bands)
        bands = [b for b in bands if b.area >= MIN_BAND_AREA and b.area >= 0.20 * median_area]

    bands = [b for b in bands if b.color_name != "unknown"]

    if len(bands) < 3:
        return CalculationError(
            error_type="INSUFFICIENT_BANDS",
            message=f"Need at least 3 bands, found {len(bands)}",
            detected_bands=bands,
        ), axis_info

    if len(bands) > 5:
        bands = filter_bands_by_area(bands, max_bands=5)

    sorted_bands = bands

    result = _decode_ordered_bands(sorted_bands)

    if isinstance(result, ResistanceResult) and not _is_e24(result.value) and len(sorted_bands) > 3:
        area_asc = sorted(range(len(sorted_bands)), key=lambda i: sorted_bands[i].area)
        for drop_i in area_asc[:2]:
            candidate = [b for j, b in enumerate(sorted_bands) if j != drop_i]
            alt = _decode_ordered_bands(candidate)
            if isinstance(alt, ResistanceResult) and _is_e24(alt.value):
                result = alt
                sorted_bands = candidate
                break

    _inject_confidence(result, _compute_axis_confidence(sorted_bands), _compute_area_confidence(mask, bands))
    return result, axis_info


_GAP_RATIO_THRESHOLD = 1.5


def _detect_tolerance_side_by_gap(
    sorted_bands: List[BandInfo],
    ratio_threshold: float = _GAP_RATIO_THRESHOLD,
) -> str:
    """
    Detect reading direction by inter-band spacing.

    The tolerance band is printed with a visibly wider gap from the nearest
    digit/multiplier band than the inter-digit gaps.  Compares each end gap
    to the median of interior gaps; only acts if the end gap exceeds
    ratio_threshold × median — returns "ambiguous" otherwise.

    Returns: "forward", "reverse", or "ambiguous"
    """
    centroids = np.array([b.centroid for b in sorted_bands])
    gaps = [float(np.linalg.norm(centroids[i + 1] - centroids[i]))
            for i in range(len(sorted_bands) - 1)]

    if len(gaps) < 3:
        return "ambiguous"

    first_gap = gaps[0]
    last_gap  = gaps[-1]
    interior  = gaps[1:-1]
    med_interior = median(interior)

    if med_interior == 0:
        return "ambiguous"

    first_qualifies = first_gap >= ratio_threshold * med_interior
    last_qualifies  = last_gap  >= ratio_threshold * med_interior

    if last_qualifies and not first_qualifies:
        return "forward"
    if first_qualifies and not last_qualifies:
        return "reverse"
    if last_qualifies and first_qualifies:
        return "forward" if last_gap >= first_gap else "reverse"
    return "ambiguous"


def determine_reading_direction(sorted_bands: List[BandInfo]) -> str:
    """
    Determine correct reading direction by locating the tolerance band at one end.

    4-band resistors: tolerance is gold or silver — unambiguous.
    5-band resistors: tolerance can also be brown(1%), red(2%), green(0.5%),
                      blue(0.25%), violet(0.1%), grey(0.05%).

    Priority 0 sanity checks run first (direction-independent):
        0a. Black at either edge — invalid as leading digit and as tolerance.
        0b. >2 gold/silver, or 2 gold/silver not adjacent at one end — artifact.
        0c. Single gold/silver strictly interior (not edge/near-edge) — artifact.

    Returns:
        "forward"                        - bands are in correct order
        "reverse"                        - bands need to be reversed
        "error"                          - direction cannot be determined
        "error_black_edge"               - black band at a boundary position
        "error_multiple_tolerance_bands" - impossible gold/silver arrangement
        "error_interior_tolerance_band"  - single gold/silver strictly interior
    """
    if not sorted_bands:
        return "error"

    n = len(sorted_bands)
    first_color = sorted_bands[0].color_name.lower()
    last_color  = sorted_bands[-1].color_name.lower()

    # Priority 0a: black at either edge is invalid regardless of direction.
    if first_color == "black" or last_color == "black":
        return "error_black_edge"

    # Priority 0b: validate gold/silver count and arrangement.
    tol_indices = [i for i, b in enumerate(sorted_bands)
                   if b.color_name.lower() in {"gold", "silver"}]
    if len(tol_indices) > 2:
        return "error_multiple_tolerance_bands"
    if len(tol_indices) == 2:
        valid_last_pair  = tol_indices == [n - 2, n - 1]
        valid_first_pair = tol_indices == [0, 1]
        if not (valid_last_pair or valid_first_pair):
            return "error_multiple_tolerance_bands"

    # Priority 0c: single gold/silver must be at an edge or near-edge slot.
    if len(tol_indices) == 1:
        pos = tol_indices[0]
        if pos not in (0, 1, n - 2, n - 1):
            return "error_interior_tolerance_band"

    # Gold/silver at an end is unambiguous (4-band indicator).
    first_is_4tol = first_color in {'gold', 'silver'}
    last_is_4tol  = last_color  in {'gold', 'silver'}

    if last_is_4tol and not first_is_4tol:
        return "forward"
    if first_is_4tol and not last_is_4tol:
        return "reverse"
    if first_is_4tol and last_is_4tol:
        return "forward"

    # Gold/silver somewhere in the middle but not at ends (near-edge slot).
    if tol_indices:
        return apply_secondary_heuristics(sorted_bands)

    # No gold/silver anywhere.  For 5-band resistors check whether a
    # 5-band tolerance color (brown, red, green, blue, violet, grey)
    # sits at one end — that end is the tolerance end.
    if n == 5:
        _5band_tol = {c for c in COLOR_TO_TOLERANCE if c not in {'gold', 'silver'}}
        first_is_5tol = first_color in _5band_tol
        last_is_5tol  = last_color  in _5band_tol

        if last_is_5tol and not first_is_5tol:
            return "forward"
        if first_is_5tol and not last_is_5tol:
            return "reverse"

        # Both ends are valid tolerance colors — use ratio gap heuristic.
        gap_result = _detect_tolerance_side_by_gap(sorted_bands)
        if gap_result != "ambiguous":
            return gap_result
        # Gap ambiguous — fall through to secondary heuristics.

    # Fall back to secondary heuristics (always returns forward/reverse).
    return apply_secondary_heuristics(sorted_bands)


def apply_secondary_heuristics(bands: List[BandInfo]) -> str:
    """
    Apply secondary heuristics when gold position is ambiguous.

    Heuristics:
    1. Black is rarely the first digit (would mean leading zero)
    2. Compare band widths (tolerance bands tend to be thinner)

    Args:
        bands: Sorted list of bands

    Returns:
        "forward" or "reverse"
    """
    first_band = bands[0]
    last_band = bands[-1]

    # Heuristic 1: Black is unlikely to be first digit
    if first_band.color_name.lower() == "black" and last_band.color_name.lower() != "black":
        return "reverse"

    # Heuristic 2: Compare band widths
    first_width = first_band.bounding_box[2] - first_band.bounding_box[0]
    last_width = last_band.bounding_box[2] - last_band.bounding_box[0]

    if first_width < last_width * 0.7:
        # First band significantly thinner - might be tolerance
        return "reverse"

    # Default to forward
    return "forward"


VALID_TOLERANCE_COLORS = {
    "brown", "red", "green", "blue", "violet", "grey", "gold", "silver"
}


def calculate_4_band_resistance(bands: List[BandInfo]) -> Union[ResistanceResult, CalculationError]:
    """
    Calculate resistance from 4-band resistor.

    Band layout: [Digit1] [Digit2] [Multiplier] [Tolerance]
    Formula: (10*D1 + D2) * Multiplier ± Tolerance%

    Args:
        bands: List of 4 BandInfo objects in reading order

    Returns:
        ResistanceResult or CalculationError
    """
    if len(bands) < 4:
        return CalculationError(
            error_type="INSUFFICIENT_BANDS",
            message=f"4-band calculation requires 4 bands, got {len(bands)}",
            detected_bands=bands
        )

    # Get color names
    color1 = bands[0].color_name.lower()
    color2 = bands[1].color_name.lower()
    color_mult = bands[2].color_name.lower()
    color_tol = bands[3].color_name.lower()

    # Look up values
    digit1 = COLOR_TO_DIGIT.get(color1)
    digit2 = COLOR_TO_DIGIT.get(color2)
    multiplier = COLOR_TO_MULTIPLIER.get(color_mult)
    tolerance = COLOR_TO_TOLERANCE.get(color_tol, DEFAULT_TOLERANCE)

    # Validate digits
    if digit1 is None:
        return CalculationError(
            error_type="INVALID_COLOR",
            message=f"'{color1}' is not a valid digit color for band 1",
            detected_bands=bands
        )
    if digit2 is None:
        return CalculationError(
            error_type="INVALID_COLOR",
            message=f"'{color2}' is not a valid digit color for band 2",
            detected_bands=bands
        )
    if multiplier is None:
        return CalculationError(
            error_type="INVALID_COLOR",
            message=f"'{color_mult}' is not a valid multiplier color for band 3",
            detected_bands=bands
        )
    if color_tol not in VALID_TOLERANCE_COLORS:
        return CalculationError(
            error_type="INVALID_TOLERANCE_COLOR",
            message=f"'{color_tol}' is not a valid tolerance color — likely a segmentation artifact.",
            detected_bands=bands
        )

    # Calculate resistance
    base_value = digit1 * 10 + digit2
    resistance = base_value * multiplier

    return ResistanceResult(
        value=resistance,
        tolerance=tolerance,
        band_count=4,
        bands=bands[:4]
    )


def calculate_5_band_resistance(bands: List[BandInfo]) -> Union[ResistanceResult, CalculationError]:
    """
    Calculate resistance from 5-band precision resistor.

    Band layout: [Digit1] [Digit2] [Digit3] [Multiplier] [Tolerance]
    Formula: (100*D1 + 10*D2 + D3) * Multiplier ± Tolerance%

    Args:
        bands: List of 5 BandInfo objects in reading order

    Returns:
        ResistanceResult or CalculationError
    """
    if len(bands) < 5:
        return CalculationError(
            error_type="INSUFFICIENT_BANDS",
            message=f"5-band calculation requires 5 bands, got {len(bands)}",
            detected_bands=bands
        )

    # Get color names
    color1 = bands[0].color_name.lower()
    color2 = bands[1].color_name.lower()
    color3 = bands[2].color_name.lower()
    color_mult = bands[3].color_name.lower()
    color_tol = bands[4].color_name.lower()

    # Look up values
    digit1 = COLOR_TO_DIGIT.get(color1)
    digit2 = COLOR_TO_DIGIT.get(color2)
    digit3 = COLOR_TO_DIGIT.get(color3)
    multiplier = COLOR_TO_MULTIPLIER.get(color_mult)
    tolerance = COLOR_TO_TOLERANCE.get(color_tol, DEFAULT_TOLERANCE)

    # Validate digits
    if digit1 is None:
        return CalculationError(
            error_type="INVALID_COLOR",
            message=f"'{color1}' is not a valid digit color for band 1",
            detected_bands=bands
        )
    if digit2 is None:
        return CalculationError(
            error_type="INVALID_COLOR",
            message=f"'{color2}' is not a valid digit color for band 2",
            detected_bands=bands
        )
    if digit3 is None:
        return CalculationError(
            error_type="INVALID_COLOR",
            message=f"'{color3}' is not a valid digit color for band 3",
            detected_bands=bands
        )
    if multiplier is None:
        return CalculationError(
            error_type="INVALID_COLOR",
            message=f"'{color_mult}' is not a valid multiplier color for band 4",
            detected_bands=bands
        )
    if color_tol not in VALID_TOLERANCE_COLORS:
        return CalculationError(
            error_type="INVALID_TOLERANCE_COLOR",
            message=f"'{color_tol}' is not a valid tolerance color — likely a segmentation artifact.",
            detected_bands=bands
        )

    # Calculate resistance
    base_value = digit1 * 100 + digit2 * 10 + digit3
    resistance = base_value * multiplier

    return ResistanceResult(
        value=resistance,
        tolerance=tolerance,
        band_count=5,
        bands=bands[:5]
    )


def calculate_3_band_resistance(bands: List[BandInfo]) -> Union[ResistanceResult, CalculationError]:
    """
    Calculate resistance from 3 visible bands (tolerance not visible).

    Assumes 4-band resistor with hidden tolerance band.
    Band layout: [Digit1] [Digit2] [Multiplier]
    Uses default tolerance of 20%.

    Args:
        bands: List of 3 BandInfo objects in reading order

    Returns:
        ResistanceResult or CalculationError
    """
    if len(bands) < 3:
        return CalculationError(
            error_type="INSUFFICIENT_BANDS",
            message=f"3-band calculation requires 3 bands, got {len(bands)}",
            detected_bands=bands
        )

    # Get color names
    color1 = bands[0].color_name.lower()
    color2 = bands[1].color_name.lower()
    color_mult = bands[2].color_name.lower()

    # Look up values
    digit1 = COLOR_TO_DIGIT.get(color1)
    digit2 = COLOR_TO_DIGIT.get(color2)
    multiplier = COLOR_TO_MULTIPLIER.get(color_mult)

    # Validate
    if digit1 is None:
        return CalculationError(
            error_type="INVALID_COLOR",
            message=f"'{color1}' is not a valid digit color for band 1",
            detected_bands=bands
        )
    if digit2 is None:
        return CalculationError(
            error_type="INVALID_COLOR",
            message=f"'{color2}' is not a valid digit color for band 2",
            detected_bands=bands
        )
    if multiplier is None:
        return CalculationError(
            error_type="INVALID_COLOR",
            message=f"'{color_mult}' is not a valid multiplier color for band 3",
            detected_bands=bands
        )

    # Calculate resistance
    base_value = digit1 * 10 + digit2
    resistance = base_value * multiplier

    return ResistanceResult(
        value=resistance,
        tolerance=DEFAULT_TOLERANCE,
        band_count=3,
        bands=bands[:3]
    )


def validate_result(result: ResistanceResult) -> List[str]:
    """
    Validate a resistance result against common values.

    Args:
        result: ResistanceResult to validate

    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []

    # E24 series values (common resistor values)
    E24_SERIES = [
        1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4,
        2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2,
        6.8, 7.5, 8.2, 9.1
    ]

    # Normalize to 1-10 range
    normalized = result.value
    while normalized >= 10:
        normalized /= 10
    while normalized < 1 and normalized > 0:
        normalized *= 10

    # Check if close to E24 value
    if normalized > 0:
        closest_e24 = min(E24_SERIES, key=lambda x: abs(x - normalized))
        if abs(closest_e24 - normalized) / closest_e24 > 0.1:
            warnings.append(
                f"Value {result.formatted} may not be a standard E24 resistor value"
            )

    # Check reasonable range
    if result.value < 0.1:
        warnings.append(f"Value {result.value} Ω is unusually low")
    elif result.value > 100_000_000:
        warnings.append(f"Value {result.formatted} is unusually high")

    return warnings
