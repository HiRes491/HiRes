"""
Validation Module

Compares calculated resistance values with ground truth values
extracted from image filenames.
"""

import re
import os
import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of validating a single image."""
    filename: str
    true_value: float           # Ohms
    true_formatted: str         # Human-readable
    calculated_value: Optional[float]  # Ohms (None if error)
    calculated_formatted: str   # Human-readable
    error_percent: Optional[float]     # Percentage error
    is_correct: bool            # Within acceptable tolerance
    status: str                 # "correct", "incorrect", "error"


def parse_resistance_from_filename(filename: str) -> Tuple[Optional[float], str]:
    """
    Extract resistance value from filename.

    Supported formats:
    - 1kohm.heic → 1000 Ω
    - 10kohm.heic → 10000 Ω
    - 220ohm.heic → 220 Ω
    - 1.2kohm.heic → 1200 Ω
    - 1Mohm.heic → 1000000 Ω
    - 4.7kohm.heic → 4700 Ω

    Args:
        filename: Image filename

    Returns:
        (value_in_ohms, formatted_string) or (None, error_message)
    """
    # Remove extension and path
    name = os.path.splitext(os.path.basename(filename))[0]

    # Convert to lowercase for matching
    name_lower = name.lower()

    # Pattern: number (with optional decimal) + unit
    # Examples: 1kohm, 10kohm, 1.2kohm, 220ohm, 1Mohm
    pattern = r'^([\d.]+)\s*(m|k)?ohm'
    match = re.match(pattern, name_lower)

    if not match:
        return None, f"Could not parse: {name}"

    value_str = match.group(1)
    unit = match.group(2) or ''  # '', 'k', or 'm'

    try:
        value = float(value_str)
    except ValueError:
        return None, f"Invalid number: {value_str}"

    # Apply multiplier
    if unit == 'k':
        value_ohms = value * 1000
        formatted = f"{value} kΩ"
    elif unit == 'm':
        value_ohms = value * 1000000
        formatted = f"{value} MΩ"
    else:
        value_ohms = value
        formatted = f"{value} Ω"

    return value_ohms, formatted


def format_resistance_value(value: float) -> str:
    """Format resistance with appropriate unit."""
    if value >= 1_000_000:
        v = value / 1_000_000
        return f"{v:.2f} MΩ" if v != int(v) else f"{int(v)} MΩ"
    elif value >= 1_000:
        v = value / 1_000
        return f"{v:.2f} kΩ" if v != int(v) else f"{int(v)} kΩ"
    else:
        return f"{value:.2f} Ω" if value != int(value) else f"{int(value)} Ω"


def calculate_error_percent(true_value: float, calculated_value: float) -> float:
    """Calculate percentage error between true and calculated values."""
    if true_value == 0:
        return float('inf') if calculated_value != 0 else 0.0
    return abs(calculated_value - true_value) / true_value * 100


def validate_single_result(filename: str,
                          calculated_value: Optional[float],
                          tolerance_percent: float = 10.0) -> ValidationResult:
    """
    Validate a single resistance calculation.

    Args:
        filename: Image filename containing true value
        calculated_value: Calculated resistance in Ohms (None if error)
        tolerance_percent: Acceptable error percentage

    Returns:
        ValidationResult object
    """
    # Parse true value from filename
    true_value, true_formatted = parse_resistance_from_filename(filename)

    if true_value is None:
        return ValidationResult(
            filename=filename,
            true_value=0,
            true_formatted=true_formatted,  # Contains error message
            calculated_value=calculated_value,
            calculated_formatted=format_resistance_value(calculated_value) if calculated_value else "N/A",
            error_percent=None,
            is_correct=False,
            status="parse_error"
        )

    if calculated_value is None:
        return ValidationResult(
            filename=filename,
            true_value=true_value,
            true_formatted=true_formatted,
            calculated_value=None,
            calculated_formatted="Error",
            error_percent=None,
            is_correct=False,
            status="calculation_error"
        )

    # Calculate error
    error_percent = calculate_error_percent(true_value, calculated_value)
    is_correct = error_percent <= tolerance_percent

    return ValidationResult(
        filename=filename,
        true_value=true_value,
        true_formatted=true_formatted,
        calculated_value=calculated_value,
        calculated_formatted=format_resistance_value(calculated_value),
        error_percent=error_percent,
        is_correct=is_correct,
        status="correct" if is_correct else "incorrect"
    )


def generate_validation_report(results: List[ValidationResult]) -> str:
    """
    Generate a formatted validation report.

    Args:
        results: List of ValidationResult objects

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Summary statistics
    total = len(results)
    correct = sum(1 for r in results if r.status == "correct")
    incorrect = sum(1 for r in results if r.status == "incorrect")
    calc_errors = sum(1 for r in results if r.status == "calculation_error")
    parse_errors = sum(1 for r in results if r.status == "parse_error")

    accuracy = (correct / total * 100) if total > 0 else 0

    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total images:          {total}")
    lines.append(f"Correct predictions:   {correct} ({accuracy:.1f}%)")
    lines.append(f"Incorrect predictions: {incorrect}")
    lines.append(f"Calculation errors:    {calc_errors}")
    lines.append(f"Filename parse errors: {parse_errors}")
    lines.append("")

    # Calculate average error for valid results
    valid_errors = [r.error_percent for r in results if r.error_percent is not None]
    if valid_errors:
        avg_error = np.mean(valid_errors)
        median_error = np.median(valid_errors)
        lines.append(f"Average error:         {avg_error:.2f}%")
        lines.append(f"Median error:          {median_error:.2f}%")
        lines.append("")

    # Detailed results table
    lines.append("DETAILED RESULTS")
    lines.append("-" * 80)
    lines.append(f"{'Filename':<25} {'True':<12} {'Calculated':<12} {'Error':<10} {'Status':<10}")
    lines.append("-" * 80)

    # Sort by status (errors first, then incorrect, then correct)
    status_order = {"parse_error": 0, "calculation_error": 1, "incorrect": 2, "correct": 3}
    sorted_results = sorted(results, key=lambda r: (status_order.get(r.status, 4), r.filename))

    for r in sorted_results:
        error_str = f"{r.error_percent:.1f}%" if r.error_percent is not None else "N/A"
        status_icon = "✓" if r.status == "correct" else "✗" if r.status == "incorrect" else "!"

        lines.append(
            f"{r.filename:<25} {r.true_formatted:<12} {r.calculated_formatted:<12} "
            f"{error_str:<10} {status_icon} {r.status}"
        )

    lines.append("-" * 80)
    lines.append("")

    # Error analysis
    if incorrect > 0:
        lines.append("INCORRECT PREDICTIONS (Detailed)")
        lines.append("-" * 40)
        for r in sorted_results:
            if r.status == "incorrect":
                lines.append(f"  {r.filename}")
                lines.append(f"    Expected: {r.true_formatted} ({r.true_value} Ω)")
                lines.append(f"    Got:      {r.calculated_formatted} ({r.calculated_value} Ω)")
                lines.append(f"    Error:    {r.error_percent:.1f}%")
                lines.append("")

    return "\n".join(lines)


def save_validation_report(results: List[ValidationResult], output_path: str):
    """Save validation report to file."""
    report = generate_validation_report(results)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Validation report saved to: {output_path}")


def print_validation_summary(results: List[ValidationResult]):
    """Print a brief validation summary to console."""
    total = len(results)
    correct = sum(1 for r in results if r.status == "correct")
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\n{'='*50}")
    print("VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")

    # Show errors
    errors = [r for r in results if r.status in ("incorrect", "calculation_error")]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for r in errors[:10]:  # Show first 10
            if r.status == "incorrect":
                print(f"  {r.filename}: expected {r.true_formatted}, got {r.calculated_formatted} ({r.error_percent:.1f}% error)")
            else:
                print(f"  {r.filename}: calculation failed")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
