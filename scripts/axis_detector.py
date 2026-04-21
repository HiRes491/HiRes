"""
Axis Detection Module

This module determines the orientation of the resistor and sorts
bands along its principal axis using PCA.
"""

import numpy as np
from typing import List, Tuple

from color_code_tables import BandInfo


def compute_principal_axis(centroids: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use PCA to find the main axis of the resistor from band centroids.

    Args:
        centroids: List of (x, y) centroid coordinates

    Returns:
        axis_vector: (2,) unit vector representing resistor axis direction
        axis_origin: (2,) mean point of centroids (point on the axis)
    """
    if len(centroids) < 2:
        # Default to horizontal axis if only one point
        return np.array([1.0, 0.0]), np.array(centroids[0]) if centroids else np.array([128.0, 128.0])

    # Convert to numpy array
    points = np.array(centroids)  # Shape: (N, 2)

    # Calculate mean (center point)
    mean_point = points.mean(axis=0)

    # Center the points
    centered = points - mean_point

    # Compute covariance matrix
    cov_matrix = np.cov(centered.T)

    # Handle edge case where cov_matrix is 1D (only 2 points)
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix, 0], [0, 0]])
    elif cov_matrix.ndim == 1:
        cov_matrix = np.diag(cov_matrix)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Principal axis is eigenvector with largest eigenvalue
    principal_idx = np.argmax(eigenvalues)
    axis_vector = eigenvectors[:, principal_idx].real

    # Normalize to unit vector
    norm = np.linalg.norm(axis_vector)
    if norm > 0:
        axis_vector = axis_vector / norm
    else:
        axis_vector = np.array([1.0, 0.0])  # Default to horizontal

    return axis_vector, mean_point


def project_point_onto_axis(point: Tuple[float, float],
                            axis_vector: np.ndarray,
                            axis_origin: np.ndarray) -> float:
    """
    Project a point onto the principal axis.

    Args:
        point: (x, y) coordinates
        axis_vector: Unit vector of the axis
        axis_origin: Origin point on the axis

    Returns:
        Scalar projection value (position along axis)
    """
    point_array = np.array(point)
    v = point_array - axis_origin
    projection = np.dot(v, axis_vector)
    return float(projection)


def project_and_sort_bands(bands: List[BandInfo],
                           axis_vector: np.ndarray,
                           axis_origin: np.ndarray) -> List[BandInfo]:
    """
    Project band centroids onto the principal axis and sort by position.

    Args:
        bands: List of BandInfo objects
        axis_vector: Unit vector representing resistor axis
        axis_origin: Origin point on the axis

    Returns:
        Sorted list of bands along the axis (from one end to the other)
    """
    if len(bands) <= 1:
        return bands

    # Calculate projection for each band
    projections = []
    for band in bands:
        proj = project_point_onto_axis(band.centroid, axis_vector, axis_origin)
        projections.append(proj)

    # Sort bands by their projection (position along axis)
    sorted_indices = np.argsort(projections)
    sorted_bands = [bands[i] for i in sorted_indices]

    return sorted_bands


def sort_bands_by_position(bands: List[BandInfo]) -> List[BandInfo]:
    """
    Main function to sort bands along the resistor axis.

    This is the primary entry point for axis-based sorting.

    Args:
        bands: List of BandInfo objects (unordered)

    Returns:
        List of BandInfo objects sorted by position along resistor axis
    """
    if len(bands) <= 1:
        return bands

    # Extract centroids
    centroids = [band.centroid for band in bands]

    # Compute principal axis using PCA
    axis_vector, axis_origin = compute_principal_axis(centroids)

    # Sort bands along the axis
    sorted_bands = project_and_sort_bands(bands, axis_vector, axis_origin)

    return sorted_bands


def get_axis_angle(axis_vector: np.ndarray) -> float:
    """
    Get the angle of the axis in degrees.

    Args:
        axis_vector: Unit vector of the axis

    Returns:
        Angle in degrees (0 = horizontal, 90 = vertical)
    """
    angle_rad = np.arctan2(axis_vector[1], axis_vector[0])
    angle_deg = np.degrees(angle_rad)
    return float(angle_deg)


def estimate_resistor_orientation(bands: List[BandInfo]) -> dict:
    """
    Estimate the orientation of the resistor.

    Args:
        bands: List of BandInfo objects

    Returns:
        Dictionary with orientation information
    """
    if len(bands) < 2:
        return {
            "orientation": "unknown",
            "angle_degrees": 0.0,
            "axis_vector": [1.0, 0.0],
            "confidence": 0.0
        }

    centroids = [band.centroid for band in bands]
    axis_vector, _ = compute_principal_axis(centroids)
    angle = get_axis_angle(axis_vector)

    # Determine orientation category
    abs_angle = abs(angle)
    if abs_angle < 15 or abs_angle > 165:
        orientation = "horizontal"
    elif 75 < abs_angle < 105:
        orientation = "vertical"
    else:
        orientation = "diagonal"

    # Estimate confidence based on spread of points along axis
    points = np.array(centroids)
    spread = np.std(points, axis=0)
    confidence = max(spread) / (np.sum(spread) + 1e-6)

    return {
        "orientation": orientation,
        "angle_degrees": angle,
        "axis_vector": axis_vector.tolist(),
        "confidence": float(confidence)
    }
