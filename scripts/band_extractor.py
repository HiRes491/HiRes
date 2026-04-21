"""
Band Extraction Module

This module extracts individual color bands from segmentation masks
using connected component analysis and axis-based clustering to handle
segmentation artifacts.

Key improvements:
- Defines band axes perpendicular to the resistor axis
- Merges same-color fragments on the same axis
- Resolves multi-color conflicts on the same axis via majority voting
"""

import numpy as np
from scipy import ndimage
from typing import List, Tuple, Dict

from color_code_tables import (
    BandInfo,
    BAND_CLASS_IDS,
    ID_TO_COLOR_NAME,
    get_color_name
)


# Minimum area threshold to filter noise (in pixels)
MIN_BAND_AREA = 40 # 50

# Maximum distance to merge fragmented components of same color
MERGE_DISTANCE_THRESHOLD = 10.0 # 10

# Threshold for grouping regions into the same band axis (perpendicular distance)
BAND_AXIS_THRESHOLD = 15.0 # 20


def extract_color_bands(mask: np.ndarray, use_axis_clustering: bool = True) -> List[BandInfo]:
    """
    Extract individual color bands from a segmentation mask.

    Uses axis-based clustering to handle segmentation artifacts:
    - Merges same-color fragments on the same band axis
    - Resolves multi-color conflicts via majority voting

    Args:
        mask: 2D numpy array (H, W) with class IDs (0-12)
        use_axis_clustering: If True, use improved axis-based method (default).
                            If False, use legacy connected component method.

    Returns:
        List of BandInfo objects for each detected band
    """
    if use_axis_clustering:
        return extract_color_bands_with_axis(mask)

    # Legacy method (kept for compatibility)
    bands = []

    for class_id in BAND_CLASS_IDS:
        # Create binary mask for this class
        binary_mask = (mask == class_id).astype(np.uint8)

        # Skip if no pixels of this class
        if binary_mask.sum() == 0:
            continue

        # Find connected components
        labeled_array, num_features = ndimage.label(binary_mask)

        # Process each connected component
        for component_id in range(1, num_features + 1):
            component_mask = (labeled_array == component_id)
            area = component_mask.sum()

            # Filter out noise (small components)
            if area < MIN_BAND_AREA:
                continue

            # Calculate centroid
            y_coords, x_coords = np.where(component_mask)
            centroid = (float(x_coords.mean()), float(y_coords.mean()))

            # Calculate bounding box
            bounding_box = (
                int(x_coords.min()),
                int(y_coords.min()),
                int(x_coords.max()),
                int(y_coords.max())
            )

            # Create BandInfo object
            band = BandInfo(
                class_id=class_id,
                color_name=get_color_name(class_id),
                centroid=centroid,
                area=int(area),
                bounding_box=bounding_box
            )
            bands.append(band)

    # Merge nearby components of the same color
    bands = merge_nearby_components(bands)

    return bands


def merge_nearby_components(bands: List[BandInfo],
                            distance_threshold: float = MERGE_DISTANCE_THRESHOLD
                            ) -> List[BandInfo]:
    """
    Merge band components that are close together and have the same color.

    This handles cases where segmentation fragments a single band
    into multiple components.

    Args:
        bands: List of BandInfo objects
        distance_threshold: Maximum distance between centroids to merge

    Returns:
        List of merged BandInfo objects
    """
    if len(bands) <= 1:
        return bands

    # Group bands by class_id
    bands_by_class = {}
    for band in bands:
        if band.class_id not in bands_by_class:
            bands_by_class[band.class_id] = []
        bands_by_class[band.class_id].append(band)

    merged_bands = []

    for class_id, class_bands in bands_by_class.items():
        if len(class_bands) == 1:
            merged_bands.append(class_bands[0])
            continue

        # Find clusters of nearby bands
        used = set()
        for i, band1 in enumerate(class_bands):
            if i in used:
                continue

            cluster = [band1]
            used.add(i)

            for j, band2 in enumerate(class_bands):
                if j in used:
                    continue

                # Calculate distance between centroids
                dist = _centroid_distance(band1.centroid, band2.centroid)

                if dist < distance_threshold:
                    cluster.append(band2)
                    used.add(j)

            # Merge cluster into single band
            merged_band = _merge_band_cluster(cluster)
            merged_bands.append(merged_band)

    return merged_bands


def _centroid_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two centroids."""
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def _merge_band_cluster(cluster: List[BandInfo]) -> BandInfo:
    """
    Merge a cluster of bands into a single BandInfo.

    The merged band has:
    - Combined area
    - Weighted average centroid
    - Expanded bounding box
    """
    if len(cluster) == 1:
        return cluster[0]

    # Use first band's class info
    class_id = cluster[0].class_id
    color_name = cluster[0].color_name

    # Calculate total area
    total_area = sum(b.area for b in cluster)

    # Calculate weighted average centroid
    weighted_x = sum(b.centroid[0] * b.area for b in cluster) / total_area
    weighted_y = sum(b.centroid[1] * b.area for b in cluster) / total_area
    centroid = (weighted_x, weighted_y)

    # Calculate combined bounding box
    x_min = min(b.bounding_box[0] for b in cluster)
    y_min = min(b.bounding_box[1] for b in cluster)
    x_max = max(b.bounding_box[2] for b in cluster)
    y_max = max(b.bounding_box[3] for b in cluster)
    bounding_box = (x_min, y_min, x_max, y_max)

    return BandInfo(
        class_id=class_id,
        color_name=color_name,
        centroid=centroid,
        area=total_area,
        bounding_box=bounding_box
    )


def extract_raw_regions(mask: np.ndarray) -> List[Dict]:
    """
    Extract raw color regions from mask without merging.

    Returns a list of dictionaries containing region info including
    the actual pixel mask for voting purposes.

    Args:
        mask: 2D numpy array (H, W) with class IDs (0-12)

    Returns:
        List of region dictionaries with class_id, centroid, area,
        bounding_box, and pixel_mask
    """
    regions = []

    for class_id in BAND_CLASS_IDS:
        # Create binary mask for this class
        binary_mask = (mask == class_id).astype(np.uint8)

        # Skip if no pixels of this class
        if binary_mask.sum() == 0:
            continue

        # Find connected components
        labeled_array, num_features = ndimage.label(binary_mask)

        # Process each connected component
        for component_id in range(1, num_features + 1):
            component_mask = (labeled_array == component_id)
            area = component_mask.sum()

            # Filter out very small noise
            if area < MIN_BAND_AREA // 2:  # Lower threshold for initial extraction
                continue

            # Calculate centroid
            y_coords, x_coords = np.where(component_mask)
            centroid = (float(x_coords.mean()), float(y_coords.mean()))

            # Calculate bounding box
            bounding_box = (
                int(x_coords.min()),
                int(y_coords.min()),
                int(x_coords.max()),
                int(y_coords.max())
            )

            regions.append({
                'class_id': class_id,
                'color_name': get_color_name(class_id),
                'centroid': centroid,
                'area': int(area),
                'bounding_box': bounding_box,
                'pixel_mask': component_mask
            })

    return regions


def compute_axis_from_regions(regions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the principal axis of the resistor from region centroids.

    Args:
        regions: List of region dictionaries

    Returns:
        axis_vector: Unit vector along resistor axis
        axis_origin: Mean point (origin on axis)
    """
    if len(regions) < 2:
        # Default to horizontal axis
        if regions:
            return np.array([1.0, 0.0]), np.array(regions[0]['centroid'])
        return np.array([1.0, 0.0]), np.array([128.0, 128.0])

    # Extract centroids
    centroids = np.array([r['centroid'] for r in regions])

    # Use the clean PCA implementation
    return compute_pca_axis(centroids)


def compute_pca_axis(points: np.ndarray, debug: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the principal axis (best fit line) through a set of 2D points using PCA.

    This finds the direction of maximum variance - the line that best fits the points
    in a least-squares sense (minimizing perpendicular distances).

    Args:
        points: Nx2 array of (x, y) coordinates
        debug: If True, print debug information

    Returns:
        axis_vector: Unit vector along the principal axis direction
        axis_origin: Mean point (centroid of all points, lies on the axis)
    """
    if len(points) < 2:
        if len(points) == 1:
            return np.array([1.0, 0.0]), points[0].copy()
        return np.array([1.0, 0.0]), np.array([128.0, 128.0])

    # Ensure points is a numpy array with shape (N, 2)
    points = np.asarray(points, dtype=np.float64)

    if debug:
        print(f"\n[PCA DEBUG] Input points ({len(points)} centroids):")
        for i, p in enumerate(points):
            print(f"  Centroid {i+1}: x={p[0]:.2f}, y={p[1]:.2f}")

    # Step 1: Compute the centroid (mean point)
    centroid = np.mean(points, axis=0)

    if debug:
        print(f"\n[PCA DEBUG] Centroid (mean): x={centroid[0]:.2f}, y={centroid[1]:.2f}")

    # Step 2: Center the points by subtracting the centroid
    centered = points - centroid

    if debug:
        print(f"\n[PCA DEBUG] Centered points:")
        for i, p in enumerate(centered):
            print(f"  Centered {i+1}: dx={p[0]:.2f}, dy={p[1]:.2f}")

    # Step 3: Compute the 2x2 covariance matrix manually for clarity
    # Cov = (1/(n-1)) * X^T * X, where X is the centered data
    n = len(points)

    # For 2D points: [[var_x, cov_xy], [cov_xy, var_y]]
    var_x = np.sum(centered[:, 0] ** 2) / (n - 1)
    var_y = np.sum(centered[:, 1] ** 2) / (n - 1)
    cov_xy = np.sum(centered[:, 0] * centered[:, 1]) / (n - 1)

    cov_matrix = np.array([[var_x, cov_xy],
                           [cov_xy, var_y]])

    if debug:
        print(f"\n[PCA DEBUG] Covariance matrix:")
        print(f"  var_x={var_x:.2f}, var_y={var_y:.2f}, cov_xy={cov_xy:.2f}")
        print(f"  [[{cov_matrix[0,0]:.2f}, {cov_matrix[0,1]:.2f}],")
        print(f"   [{cov_matrix[1,0]:.2f}, {cov_matrix[1,1]:.2f}]]")

    # Step 4: Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # eigh for symmetric matrices

    if debug:
        print(f"\n[PCA DEBUG] Eigenvalues: {eigenvalues}")
        print(f"[PCA DEBUG] Eigenvectors:")
        print(f"  v1 (eigenval={eigenvalues[0]:.2f}): [{eigenvectors[0,0]:.4f}, {eigenvectors[1,0]:.4f}]")
        print(f"  v2 (eigenval={eigenvalues[1]:.2f}): [{eigenvectors[0,1]:.4f}, {eigenvectors[1,1]:.4f}]")

    # Step 5: The principal axis is the eigenvector with the LARGEST eigenvalue
    # eigh returns eigenvalues in ascending order, so the last one is largest
    principal_axis = eigenvectors[:, -1]

    # Ensure it's a unit vector (should already be, but just in case)
    norm = np.linalg.norm(principal_axis)
    if norm > 0:
        principal_axis = principal_axis / norm

    # Calculate angle for debug
    if debug:
        angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
        angle_deg = np.degrees(angle_rad)
        print(f"\n[PCA DEBUG] Principal axis vector: [{principal_axis[0]:.4f}, {principal_axis[1]:.4f}]")
        print(f"[PCA DEBUG] Principal axis angle: {angle_deg:.2f} degrees")
        print(f"[PCA DEBUG] (0° = horizontal right, 90° = down, -90° = up)")

    return principal_axis, centroid


def compute_mask_orientation(pixel_mask: np.ndarray) -> float:
    """
    Compute the orientation angle of a binary mask using image moments.

    Args:
        pixel_mask: 2D boolean array representing the mask

    Returns:
        Orientation angle in radians (0 = horizontal, pi/2 = vertical)
    """
    y_coords, x_coords = np.where(pixel_mask)

    if len(x_coords) < 2:
        return 0.0

    # Compute centroid
    cx = x_coords.mean()
    cy = y_coords.mean()

    # Compute central moments
    mu20 = np.sum((x_coords - cx) ** 2)  # Variance in x
    mu02 = np.sum((y_coords - cy) ** 2)  # Variance in y
    mu11 = np.sum((x_coords - cx) * (y_coords - cy))  # Covariance

    # Compute orientation angle
    # This gives the angle of the major axis of the ellipse fitting the mask
    angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

    return angle


def compute_axis_from_band_orientations(regions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the resistor axis as perpendicular to the average band orientation.

    This method is more robust than PCA on centroids because it uses the
    actual shape/orientation of each band mask, not just their centers.

    Args:
        regions: List of region dictionaries with 'pixel_mask' and 'centroid'

    Returns:
        axis_vector: Unit vector along resistor axis
        axis_origin: Mean point of centroids
    """
    if len(regions) < 1:
        return np.array([1.0, 0.0]), np.array([128.0, 128.0])

    # Calculate mean centroid as axis origin
    centroids = np.array([r['centroid'] for r in regions])
    mean_point = centroids.mean(axis=0)

    if len(regions) < 2:
        return np.array([1.0, 0.0]), mean_point

    # Compute orientation of each band mask
    orientations = []
    weights = []
    for r in regions:
        if 'pixel_mask' in r:
            angle = compute_mask_orientation(r['pixel_mask'])
            orientations.append(angle)
            weights.append(r['area'])  # Weight by area

    if not orientations:
        # Fall back to PCA on centroids
        return compute_axis_from_regions(regions)

    # Compute weighted average orientation
    # Use circular mean to handle angle wraparound
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Convert angles to unit vectors, average, then back to angle
    sin_sum = np.sum(weights * np.sin(2 * np.array(orientations)))
    cos_sum = np.sum(weights * np.cos(2 * np.array(orientations)))
    avg_orientation = 0.5 * np.arctan2(sin_sum, cos_sum)

    # The band orientation is perpendicular to the resistor axis
    # If bands are vertical (angle ≈ ±90°), resistor axis is horizontal
    # So resistor axis angle = band angle - 90° = band angle + 90°
    resistor_angle = avg_orientation + np.pi / 2

    # Convert to unit vector
    axis_vector = np.array([np.cos(resistor_angle), np.sin(resistor_angle)])

    return axis_vector, mean_point


def compute_axis_from_endpoints(bands: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute resistor axis as the vector from first to last band centroid.

    This is the simplest and most robust method - the axis is literally
    the line connecting the first and last detected bands.

    Args:
        bands: List of BandInfo objects (must be sorted by position)

    Returns:
        axis_vector: Unit vector along resistor axis
        axis_origin: Midpoint between first and last band
    """
    if len(bands) < 2:
        if bands:
            return np.array([1.0, 0.0]), np.array(bands[0].centroid)
        return np.array([1.0, 0.0]), np.array([128.0, 128.0])

    # Get first and last centroids
    first = np.array(bands[0].centroid)
    last = np.array(bands[-1].centroid)

    # Axis vector from first to last
    axis_vector = last - first

    # Normalize
    norm = np.linalg.norm(axis_vector)
    if norm > 0:
        axis_vector = axis_vector / norm
    else:
        axis_vector = np.array([1.0, 0.0])

    # Origin is midpoint of all band centroids
    all_centroids = np.array([b.centroid for b in bands])
    axis_origin = all_centroids.mean(axis=0)

    return axis_vector, axis_origin


def project_perpendicular(point: Tuple[float, float],
                          axis_vector: np.ndarray,
                          axis_origin: np.ndarray) -> float:
    """
    Calculate perpendicular distance from a point to the axis.

    This is used to determine which "band axis" a region belongs to.

    Args:
        point: (x, y) coordinates
        axis_vector: Unit vector along resistor axis
        axis_origin: Point on the axis

    Returns:
        Signed perpendicular distance (position along axis)
    """
    point_arr = np.array(point)
    v = point_arr - axis_origin

    # Project onto axis (parallel component)
    parallel = np.dot(v, axis_vector)

    return parallel


def cluster_regions_by_axis(regions: List[Dict],
                            axis_vector: np.ndarray,
                            axis_origin: np.ndarray,
                            threshold: float = BAND_AXIS_THRESHOLD) -> List[List[Dict]]:
    """
    Cluster regions into band axes based on their position along the resistor.

    Regions that have similar projection values along the axis belong
    to the same band axis.

    Uses a greedy clustering approach where adjacent sorted regions are
    grouped if the gap between them is within the threshold.

    Args:
        regions: List of region dictionaries
        axis_vector: Unit vector along resistor axis
        axis_origin: Origin point on axis
        threshold: Maximum distance between projections to be same band

    Returns:
        List of clusters, where each cluster is a list of regions on the same band axis
    """
    if not regions:
        return []

    # Calculate projection for each region
    projections = []
    for r in regions:
        proj = project_perpendicular(r['centroid'], axis_vector, axis_origin)
        projections.append((proj, r))

    # Sort by projection
    projections.sort(key=lambda x: x[0])

    # Cluster based on gaps between adjacent sorted regions
    clusters = []
    current_cluster = [projections[0][1]]
    current_cluster_projs = [projections[0][0]]

    for proj, region in projections[1:]:
        # Compare with the last projection in the current cluster
        last_proj = current_cluster_projs[-1]

        if abs(proj - last_proj) <= threshold:
            # Same band axis - gap is small
            current_cluster.append(region)
            current_cluster_projs.append(proj)
        else:
            # New band axis - gap is large
            clusters.append(current_cluster)
            current_cluster = [region]
            current_cluster_projs = [proj]

    # Don't forget the last cluster
    clusters.append(current_cluster)

    return clusters


def resolve_band_axis(cluster: List[Dict], mask: np.ndarray) -> BandInfo:
    """
    Resolve a cluster of regions into a single band using majority voting.

    If all regions are the same color, merge them.
    If multiple colors, the color with the most pixels wins.

    Args:
        cluster: List of region dictionaries on the same band axis
        mask: Original segmentation mask (for accurate pixel counting)

    Returns:
        Single BandInfo representing the resolved band
    """
    if len(cluster) == 1:
        r = cluster[0]
        return BandInfo(
            class_id=r['class_id'],
            color_name=r['color_name'],
            centroid=r['centroid'],
            area=r['area'],
            bounding_box=r['bounding_box']
        )

    # Count pixels by color
    color_pixels = {}
    for r in cluster:
        cid = r['class_id']
        if cid not in color_pixels:
            color_pixels[cid] = 0
        color_pixels[cid] += r['area']

    # Find winning color (majority vote)
    winning_class_id = max(color_pixels.keys(), key=lambda k: color_pixels[k])

    # Gather all regions of the winning color
    winning_regions = [r for r in cluster if r['class_id'] == winning_class_id]

    # Merge winning regions
    total_area = sum(r['area'] for r in winning_regions)

    # Weighted centroid
    weighted_x = sum(r['centroid'][0] * r['area'] for r in winning_regions) / total_area
    weighted_y = sum(r['centroid'][1] * r['area'] for r in winning_regions) / total_area

    # Combined bounding box
    x_min = min(r['bounding_box'][0] for r in winning_regions)
    y_min = min(r['bounding_box'][1] for r in winning_regions)
    x_max = max(r['bounding_box'][2] for r in winning_regions)
    y_max = max(r['bounding_box'][3] for r in winning_regions)

    return BandInfo(
        class_id=winning_class_id,
        color_name=get_color_name(winning_class_id),
        centroid=(weighted_x, weighted_y),
        area=total_area,
        bounding_box=(x_min, y_min, x_max, y_max)
    )


def extract_color_bands_with_axis(mask: np.ndarray, return_axis_info: bool = False):
    """
    Extract color bands using axis-based clustering and voting.

    This improved method:
    1. Extracts all raw color regions
    2. Computes the resistor's principal axis
    3. Clusters regions by their position along the axis (band axes)
    4. Resolves each band axis using majority voting

    Args:
        mask: 2D numpy array (H, W) with class IDs (0-12)
        return_axis_info: If True, also return axis visualization data

    Returns:
        If return_axis_info is False:
            List of BandInfo objects, one per band axis
        If return_axis_info is True:
            Tuple of (bands, axis_info) where axis_info is a dict containing:
            - 'axis_vector': unit vector along resistor axis
            - 'axis_origin': center point on axis
            - 'band_projections': list of projection values for each band axis
    """
    # Step 1: Extract raw regions
    regions = extract_raw_regions(mask)

    if not regions:
        if return_axis_info:
            return [], None
        return []

    # Filter very small regions before axis computation
    regions = [r for r in regions if r['area'] >= MIN_BAND_AREA // 2]

    if len(regions) < 2:
        # Not enough for axis computation, return as-is
        if regions:
            r = regions[0]
            band = BandInfo(
                class_id=r['class_id'],
                color_name=r['color_name'],
                centroid=r['centroid'],
                area=r['area'],
                bounding_box=r['bounding_box']
            )
            if return_axis_info:
                return [band], None
            return [band]
        if return_axis_info:
            return [], None
        return []

    # Step 2: Compute principal axis from band mask orientations
    # This is more robust than PCA on centroids - it uses actual band shapes
    axis_vector, axis_origin = compute_axis_from_band_orientations(regions)

    # Step 3: Cluster regions by band axis
    clusters = cluster_regions_by_axis(regions, axis_vector, axis_origin)

    # Step 4: Resolve each cluster to a single band
    bands = []
    for cluster in clusters:
        band = resolve_band_axis(cluster, mask)
        # Final area filter
        if band.area >= MIN_BAND_AREA:
            bands.append(band)

    # Step 5: Re-compute axis using PCA on final band centroids ONLY
    # This ensures the axis is the best fit line through the actual detected bands
    if len(bands) >= 2:
        final_centroids = np.array([b.centroid for b in bands])
        axis_vector, axis_origin = compute_pca_axis(final_centroids)

    # Compute band projections using the final axis
    band_projections = []
    for band in bands:
        proj = project_perpendicular(band.centroid, axis_vector, axis_origin)
        band_projections.append(proj)

    if return_axis_info:
        axis_info = {
            'axis_vector': axis_vector,
            'axis_origin': axis_origin,
            'band_projections': band_projections
        }
        return bands, axis_info

    return bands


def extract_color_bands_with_visualization(mask: np.ndarray) -> Tuple[List[BandInfo], dict]:
    """
    Extract color bands and return axis info for visualization.

    Convenience wrapper around extract_color_bands_with_axis.

    Args:
        mask: 2D numpy array (H, W) with class IDs (0-12)

    Returns:
        Tuple of (bands, axis_info)
    """
    return extract_color_bands_with_axis(mask, return_axis_info=True)


def filter_bands_by_area(bands: List[BandInfo],
                         min_area: int = MIN_BAND_AREA,
                         max_bands: int = 6) -> List[BandInfo]:
    """
    Filter bands by area, keeping only the largest components.

    Used when too many bands are detected (likely noise).

    Args:
        bands: List of BandInfo objects
        min_area: Minimum area threshold
        max_bands: Maximum number of bands to keep

    Returns:
        Filtered list of BandInfo objects
    """
    # Filter by minimum area
    filtered = [b for b in bands if b.area >= min_area]

    # If still too many, keep only the largest
    if len(filtered) > max_bands:
        filtered = sorted(filtered, key=lambda b: b.area, reverse=True)
        filtered = filtered[:max_bands]

    return filtered


def get_band_statistics(bands: List[BandInfo]) -> dict:
    """
    Get statistics about detected bands.

    Useful for debugging and validation.

    Args:
        bands: List of BandInfo objects

    Returns:
        Dictionary with statistics
    """
    if not bands:
        return {
            "count": 0,
            "colors": [],
            "total_area": 0,
            "has_tolerance": False
        }

    return {
        "count": len(bands),
        "colors": [b.color_name for b in bands],
        "total_area": sum(b.area for b in bands),
        "has_tolerance": any(b.is_tolerance_band for b in bands),
        "areas": [b.area for b in bands],
        "centroids": [b.centroid for b in bands]
    }
