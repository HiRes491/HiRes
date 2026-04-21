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
    format_resistance
)
from band_extractor import extract_color_bands, filter_bands_by_area, extract_color_bands_with_visualization, compute_pca_axis, project_perpendicular
from axis_detector import sort_bands_by_position


def calculate_resistance(mask: np.ndarray) -> Union[ResistanceResult, CalculationError]:
    """
    Main entry point: Calculate resistance value from a segmentation mask.

    Pipeline:
    1. Extract color bands from mask
    2. Sort bands along resistor axis
    3. Determine reading direction (gold tolerance at end)
    4. Calculate resistance based on band count

    Args:
        mask: 2D numpy array (H, W) with class IDs (0-12)

    Returns:
        ResistanceResult on success, CalculationError on failure
    """
    # Step 1: Extract color bands
    bands = extract_color_bands(mask)

    # Validate minimum bands
    if len(bands) < 3:
        return CalculationError(
            error_type="INSUFFICIENT_BANDS",
            message=f"Need at least 3 bands, found {len(bands)}",
            detected_bands=bands
        )

    # Filter if too many bands detected
    if len(bands) > 6:
        bands = filter_bands_by_area(bands, max_bands=5)

    # Step 2: Sort bands by position along resistor axis
    sorted_bands = sort_bands_by_position(bands)

    # Step 3: Determine reading direction
    reading_direction = determine_reading_direction(sorted_bands)

    if reading_direction == "error_unknown_direction":
        return CalculationError(
            error_type="UNKNOWN_DIRECTION",
            message="Could not determine reading direction: no gold/silver at an end and gap heuristic was ambiguous.",
            detected_bands=sorted_bands
        )

    if reading_direction == "error_black_edge":
        return CalculationError(
            error_type="BLACK_BOUNDARY_BAND",
            message=(
                "Black band detected at a boundary position. Black is invalid "
                "as the first significant digit (leading zero) and is not a "
                "valid tolerance color. Likely a segmentation artifact."
            ),
            detected_bands=sorted_bands
        )

    if reading_direction == "error_multiple_tolerance_bands":
        return CalculationError(
            error_type="MULTIPLE_TOLERANCE_BANDS",
            message=(
                "More than one tolerance band (gold/silver) detected. A "
                "physical resistor has exactly one tolerance band; multiple "
                "detections indicate a segmentation artifact."
            ),
            detected_bands=sorted_bands
        )

    if reading_direction == "error_interior_tolerance_band":
        return CalculationError(
            error_type="INTERIOR_TOLERANCE_BAND",
            message=(
                "Gold/silver detected in a strictly interior position. On a "
                "real resistor the tolerance band is at an edge (and the "
                "multiplier at most one position inward); this indicates a "
                "segmentation artifact."
            ),
            detected_bands=sorted_bands
        )

    if reading_direction == "reverse":
        sorted_bands = list(reversed(sorted_bands))

    # Step 4: Calculate resistance based on band count
    band_count = len(sorted_bands)

    if band_count == 4:
        return calculate_4_band_resistance(sorted_bands)
    elif band_count == 5:
        return calculate_5_band_resistance(sorted_bands)
    elif band_count == 3:
        # 3 bands: assume no tolerance band visible, treat as 4-band
        return calculate_3_band_resistance(sorted_bands)
    else:
        # 6+ bands: try 5-band calculation with first 5
        return calculate_5_band_resistance(sorted_bands[:5])


def calculate_resistance_with_axis_info(mask: np.ndarray) -> tuple:
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
    # Extract bands with axis info
    bands, axis_info = extract_color_bands_with_visualization(mask)

    # Validate minimum bands
    if len(bands) < 3:
        return CalculationError(
            error_type="INSUFFICIENT_BANDS",
            message=f"Need at least 3 bands, found {len(bands)}",
            detected_bands=bands
        ), axis_info

    # Filter if too many bands detected
    if len(bands) > 6:
        bands = filter_bands_by_area(bands, max_bands=5)

    # Sort bands by position along resistor axis
    sorted_bands = sort_bands_by_position(bands)

    # Determine reading direction
    reading_direction = determine_reading_direction(sorted_bands)

    if reading_direction == "error_unknown_direction":
        return CalculationError(
            error_type="UNKNOWN_DIRECTION",
            message="Could not determine reading direction: no gold/silver at an end and gap heuristic was ambiguous.",
            detected_bands=sorted_bands
        ), axis_info

    if reading_direction == "error_black_edge":
        return CalculationError(
            error_type="BLACK_BOUNDARY_BAND",
            message=(
                "Black band detected at a boundary position. Black is invalid "
                "as the first significant digit (leading zero) and is not a "
                "valid tolerance color. Likely a segmentation artifact."
            ),
            detected_bands=sorted_bands
        ), axis_info

    if reading_direction == "error_multiple_tolerance_bands":
        return CalculationError(
            error_type="MULTIPLE_TOLERANCE_BANDS",
            message=(
                "More than one tolerance band (gold/silver) detected. A "
                "physical resistor has exactly one tolerance band; multiple "
                "detections indicate a segmentation artifact."
            ),
            detected_bands=sorted_bands
        ), axis_info

    if reading_direction == "error_interior_tolerance_band":
        return CalculationError(
            error_type="INTERIOR_TOLERANCE_BAND",
            message=(
                "Gold/silver detected in a strictly interior position. On a "
                "real resistor the tolerance band is at an edge (and the "
                "multiplier at most one position inward); this indicates a "
                "segmentation artifact."
            ),
            detected_bands=sorted_bands
        ), axis_info

    if reading_direction == "reverse":
        sorted_bands = list(reversed(sorted_bands))

    # Calculate resistance based on band count
    band_count = len(sorted_bands)

    if band_count == 4:
        result = calculate_4_band_resistance(sorted_bands)
    elif band_count == 5:
        result = calculate_5_band_resistance(sorted_bands)
    elif band_count == 3:
        result = calculate_3_band_resistance(sorted_bands)
    else:
        result = calculate_5_band_resistance(sorted_bands[:5])

    return result, axis_info


GAP_RATIO_THRESHOLD = 1.5


def detect_tolerance_side_by_gap(
    sorted_bands: List[BandInfo],
    band_projections: List[float],
    ratio_threshold: float = GAP_RATIO_THRESHOLD,
) -> str:
    """
    Detect reading direction by inter-band spacing.

    The tolerance band on a standard resistor is printed with a visibly
    wider gap from the nearest digit/multiplier band than the inter-digit
    gaps. We compare the gap at each end to the median of the interior
    gaps and return the direction that puts the larger end-gap last.

    Returns:
        "forward"   - larger gap is at the last-band end
        "reverse"   - larger gap is at the first-band end
        "ambiguous" - no gap exceeds ratio_threshold * median_interior_gap
    """
    projs = sorted(band_projections)
    gaps = [projs[i + 1] - projs[i] for i in range(len(projs) - 1)]

    if len(gaps) < 3:
        return "ambiguous"

    first_gap = gaps[0]
    last_gap = gaps[-1]
    interior = gaps[1:-1]
    med_interior = median(interior)

    if med_interior == 0:
        return "ambiguous"

    first_qualifies = first_gap >= ratio_threshold * med_interior
    last_qualifies = last_gap >= ratio_threshold * med_interior

    if last_qualifies and not first_qualifies:
        return "forward"
    if first_qualifies and not last_qualifies:
        return "reverse"
    if last_qualifies and first_qualifies:
        return "forward" if last_gap >= first_gap else "reverse"
    return "ambiguous"


def _compute_band_projections(sorted_bands: List[BandInfo]) -> List[float]:
    """Recompute axis projections from band centroids via PCA."""
    centroids = np.array([b.centroid for b in sorted_bands])
    axis_vector, axis_origin = compute_pca_axis(centroids, debug=False)
    return [
        float(project_perpendicular(b.centroid, axis_vector, axis_origin))
        for b in sorted_bands
    ]


def determine_reading_direction(sorted_bands: List[BandInfo]) -> str:
    """
    Determine correct reading direction.

    Priority 0: Physical sanity checks (direction-independent):
                0a. Black at either edge → segmentation artifact
                    (black is invalid as leading digit and as tolerance)
                0b. Multiple gold/silver bands in an impossible arrangement
                    (count > 2, or count == 2 not adjacent at one end)
                0c. Single gold/silver strictly interior (not at an edge
                    or near-edge Mult position) → segmentation artifact
    Priority 1: Single tolerance band at an edge → color-based decision.
    Priority 2: No tolerance band anywhere → gap heuristic.
    Priority 2b: Gap ambiguous → forward (default).

    Args:
        sorted_bands: Bands sorted by position along axis

    Returns:
        "forward"                            - bands are in correct order
        "reverse"                            - bands need to be reversed
        "error_unknown_direction"            - direction could not be determined
        "error_black_edge"                   - black band at a boundary
        "error_multiple_tolerance_bands"     - impossible gold/silver arrangement
        "error_interior_tolerance_band"      - single gold/silver strictly interior
    """
    if not sorted_bands:
        return "error_unknown_direction"

    n = len(sorted_bands)

    # Priority 0a: Black at either edge is invalid regardless of direction.
    # - At position 0 (would-be D1): leading zero is not a valid digit
    # - At position N-1 (would-be Tol): black is not a valid tolerance color
    # Reversing does not resolve either case. Flagging here — before any
    # direction logic — prevents confidently-wrong readings like
    # [black, brown, red, gold] decoding as 0-1-2-gold = 12 × 100Ω ±5%.
    if sorted_bands[0].color_name.lower() == "black" or \
       sorted_bands[-1].color_name.lower() == "black":
        return "error_black_edge"

    # Priority 0b: Gold/silver as dual-purpose colors (tolerance AND multiplier).
    # Sub-ohm resistors legally have gold/silver at adjacent positions at one
    # end (Mult + Tol). All other multi-gold/silver arrangements are artifacts.
    tol_indices = [
        i for i, b in enumerate(sorted_bands)
        if b.color_name.lower() in {"gold", "silver"}
    ]
    if len(tol_indices) > 2:
        return "error_multiple_tolerance_bands"
    if len(tol_indices) == 2:
        valid_last_pair = tol_indices == [n - 2, n - 1]
        valid_first_pair = tol_indices == [0, 1]
        if not (valid_last_pair or valid_first_pair):
            return "error_multiple_tolerance_bands"

    # Priority 0c: Single gold/silver must be at an edge or near-edge
    # (Mult or Tol slot in forward OR reverse reading). Strictly interior
    # positions {2, ..., N-3} cannot host gold/silver on any real resistor.
    # This only triggers for N >= 5 (4-band has no strictly-interior slot).
    if len(tol_indices) == 1:
        pos = tol_indices[0]
        if pos not in (0, 1, n - 2, n - 1):
            return "error_interior_tolerance_band"

    first_is_tol = sorted_bands[0].color_name.lower() in {"gold", "silver"}
    last_is_tol = sorted_bands[-1].color_name.lower() in {"gold", "silver"}

    # Priority 1: tolerance color at an edge (at most one by Priority 0b)
    if last_is_tol:
        return "forward"
    if first_is_tol:
        return "reverse"

    # Priority 2: no tolerance color at either edge — try gap heuristic
    if n >= 2:
        projections = _compute_band_projections(sorted_bands)
        gap_result = detect_tolerance_side_by_gap(sorted_bands, projections)
        if gap_result != "ambiguous":
            return gap_result

    # Priority 2b: gap ambiguous — no remaining signal. Default to forward.
    return "forward"


# Valid tolerance colors per IEC 60062. Any color outside this set at the
# resolved tolerance position indicates a segmentation artifact.
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
            message=(
                f"'{color_tol}' is not a valid tolerance color. Valid "
                f"tolerance colors are brown, red, green, blue, violet, grey, "
                f"gold, silver. Likely a segmentation artifact."
            ),
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
            message=(
                f"'{color_tol}' is not a valid tolerance color. Valid "
                f"tolerance colors are brown, red, green, blue, violet, grey, "
                f"gold, silver. Likely a segmentation artifact."
            ),
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
