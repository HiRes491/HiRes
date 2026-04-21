"""
Resistor Color Code Lookup Tables and Constants

This module contains all the lookup tables for converting resistor
color bands to resistance values.
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np

# =============================================================================
# Class ID to Color Name Mapping (from segmentation model)
# =============================================================================

ID_TO_COLOR_NAME = {
    0: "background",    # Background (ignore)
    1: "gold",          # Tolerance band
    2: "orange",
    3: "green",
    4: "brown",
    5: "background",    # Resistor body (ignore)
    6: "blue",
    7: "yellow",
    8: "black",         # Rendered as pink in visualization
    9: "white",
    10: "grey",
    11: "red",
    12: "violet"
}

# Class IDs that represent actual color bands (not background)
BAND_CLASS_IDS = {1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12}

# Class IDs for tolerance bands
TOLERANCE_CLASS_IDS = {1}  # Gold only; silver has no segmentation class ID in current model

# =============================================================================
# Color to Digit Value Mapping
# =============================================================================

COLOR_TO_DIGIT = {
    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "violet": 7,
    "grey": 8,
    "white": 9
}

# =============================================================================
# Color to Multiplier Mapping
# =============================================================================

COLOR_TO_MULTIPLIER = {
    "black": 1,              # 10^0
    "brown": 10,             # 10^1
    "red": 100,              # 10^2
    "orange": 1_000,         # 10^3
    "yellow": 10_000,        # 10^4
    "green": 100_000,        # 10^5
    "blue": 1_000_000,       # 10^6
    "violet": 10_000_000,    # 10^7
    "grey": 100_000_000,     # 10^8
    "white": 1_000_000_000,  # 10^9
    "gold": 0.1,             # 10^-1
    "silver": 0.01           # 10^-2 (for future use)
}

# =============================================================================
# Color to Tolerance Mapping (percentage)
# =============================================================================

COLOR_TO_TOLERANCE = {
    "brown": 1.0,
    "red": 2.0,
    "green": 0.5,
    "blue": 0.25,
    "violet": 0.1,
    "grey": 0.05,
    "gold": 5.0,
    "silver": 10.0  # For future use
}

# Default tolerance when no tolerance band is detected
DEFAULT_TOLERANCE = 20.0

# =============================================================================
# Visualization Colors (RGB)
# =============================================================================

# Colors used for mask visualization
VISUALIZATION_COLORS_RGB = {
    "background": [0, 0, 0],
    "gold": [212, 175, 55],
    "orange": [255, 165, 0],
    "green": [0, 128, 0],
    "brown": [150, 75, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0],
    "black": [255, 105, 180],  # Pink for visibility on black background
    "white": [255, 255, 255],
    "grey": [128, 128, 128],
    "red": [255, 0, 0],
    "violet": [148, 0, 211]
}

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BandInfo:
    """Information about a single detected color band."""
    class_id: int
    color_name: str
    centroid: Tuple[float, float]  # (x, y)
    area: int                       # Pixel count
    bounding_box: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)

    @property
    def is_tolerance_band(self) -> bool:
        """Check if this band is a tolerance indicator."""
        return self.class_id in TOLERANCE_CLASS_IDS

    @property
    def digit_value(self) -> Optional[int]:
        """Get digit value for this color, or None if not a digit color."""
        return COLOR_TO_DIGIT.get(self.color_name.lower())

    @property
    def multiplier_value(self) -> Optional[float]:
        """Get multiplier value for this color."""
        return COLOR_TO_MULTIPLIER.get(self.color_name.lower())

    @property
    def tolerance_value(self) -> Optional[float]:
        """Get tolerance percentage for this color."""
        return COLOR_TO_TOLERANCE.get(self.color_name.lower())


@dataclass
class ResistanceResult:
    """Result of resistance calculation."""
    value: float                    # Resistance in Ohms
    tolerance: float                # Tolerance percentage
    band_count: int                 # Number of bands (4 or 5)
    bands: List[BandInfo]           # Detected bands in reading order

    @property
    def formatted(self) -> str:
        """Format resistance with appropriate unit prefix."""
        return format_resistance(self.value)

    @property
    def range(self) -> Tuple[float, float]:
        """Calculate min/max resistance based on tolerance."""
        delta = self.value * (self.tolerance / 100)
        return (self.value - delta, self.value + delta)

    def __str__(self) -> str:
        colors = " → ".join([b.color_name for b in self.bands])
        return f"{self.formatted} ±{self.tolerance}% ({colors})"


@dataclass
class CalculationError:
    """Error information when resistance calculation fails."""
    error_type: str
    message: str
    detected_bands: List[BandInfo]

    def __str__(self) -> str:
        return f"{self.error_type}: {self.message}"


# =============================================================================
# Utility Functions
# =============================================================================

def format_resistance(value: float) -> str:
    """
    Format resistance value with appropriate unit prefix.

    Args:
        value: Resistance in Ohms

    Returns:
        Formatted string (e.g., "4.7 kΩ", "1 MΩ", "220 Ω")
    """
    if value >= 1_000_000:
        formatted = value / 1_000_000
        unit = "MΩ"
    elif value >= 1_000:
        formatted = value / 1_000
        unit = "kΩ"
    elif value >= 1:
        formatted = value
        unit = "Ω"
    else:
        formatted = value * 1000
        unit = "mΩ"

    # Remove unnecessary decimal places
    if formatted == int(formatted):
        return f"{int(formatted)} {unit}"
    elif formatted * 10 == int(formatted * 10):
        return f"{formatted:.1f} {unit}"
    else:
        return f"{formatted:.2f} {unit}"


def get_color_name(class_id: int) -> str:
    """Get color name from class ID."""
    return ID_TO_COLOR_NAME.get(class_id, "unknown")


def is_valid_digit_color(color_name: str) -> bool:
    """Check if a color can represent a digit."""
    return color_name.lower() in COLOR_TO_DIGIT


def is_tolerance_color(color_name: str) -> bool:
    """Check if a color is a tolerance indicator."""
    return color_name.lower() in {"gold", "silver"}
