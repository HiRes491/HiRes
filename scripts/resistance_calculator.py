"""
Resistance Calculator Module

This module orchestrates the full pipeline to convert a segmentation
mask into a resistance value.
"""

import numpy as np
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
from band_extractor import extract_color_bands, filter_bands_by_area, extract_color_bands_with_visualization
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

    if reading_direction == "error_no_gold":
        return CalculationError(
            error_type="NO_TOLERANCE_BAND",
            message="No gold tolerance band found. Silver tolerance not yet supported.",
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

    if reading_direction == "error_no_gold":
        return CalculationError(
            error_type="NO_TOLERANCE_BAND",
            message="No gold tolerance band found. Silver tolerance not yet supported.",
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


def determine_reading_direction(sorted_bands: List[BandInfo]) -> str:
    """
    Determine correct reading direction by locating the gold tolerance band.

    The gold tolerance band should be LAST when reading correctly.
    If gold is at position 0, we need to reverse the order.

    Args:
        sorted_bands: Bands sorted by position along axis

    Returns:
        "forward"          - bands are in correct order
        "reverse"          - bands need to be reversed
        "error_no_gold"    - no gold tolerance band found
        "error_black_edge" - black band at a boundary (segmentation artifact)
    """
    if not sorted_bands:
        return "error_no_gold"

    first_band = sorted_bands[0]
    last_band = sorted_bands[-1]

    first_is_gold = first_band.color_name.lower() == "gold"
    last_is_gold = last_band.color_name.lower() == "gold"

    if last_is_gold and not first_is_gold:
        return "forward"
    elif first_is_gold and not last_is_gold:
        return "reverse"
    elif first_is_gold and last_is_gold:
        # Both ends are gold - unusual, but treat as forward
        return "forward"
    else:
        # No gold band found - check if any band is gold
        has_gold = any(b.color_name.lower() == "gold" for b in sorted_bands)
        if has_gold:
            # Gold is in the middle - use secondary heuristics
            return apply_secondary_heuristics(sorted_bands)
        else:
            return "error_no_gold"


def apply_secondary_heuristics(bands: List[BandInfo]) -> str:
    """
    Apply secondary heuristics when gold position is ambiguous.

    Heuristics:
    1. Black at either end is invalid (leading zero as first digit; black is
       not a valid tolerance color, so it cannot be the last band either).
       Both cases indicate a segmentation artifact, so return "error".
    2. Compare band widths (tolerance bands tend to be thinner).

    Args:
        bands: Sorted list of bands

    Returns:
        "forward", "reverse", or "error_black_edge"
    """
    first_band = bands[0]
    last_band = bands[-1]

    first_is_black = first_band.color_name.lower() == "black"
    last_is_black = last_band.color_name.lower() == "black"

    # Heuristic 1: Black at either boundary is invalid
    # - First position: leading zero is not a valid significant digit
    # - Last position: black is not in the valid tolerance color set
    # Reversing does not fix either case, so flag as a segmentation error.
    if first_is_black or last_is_black:
        return "error_black_edge"

    # Heuristic 2: Compare band widths
    first_width = first_band.bounding_box[2] - first_band.bounding_box[0]
    last_width = last_band.bounding_box[2] - last_band.bounding_box[0]

    if first_width < last_width * 0.7:
        # First band significantly thinner - might be tolerance
        return "reverse"

    # Default to forward
    return "forward"


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
