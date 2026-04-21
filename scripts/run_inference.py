import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil

# Import resistance calculator
from resistance_calculator import calculate_resistance, calculate_resistance_with_axis_info, ResistanceResult, CalculationError

# Import validation module
from validate_results import (
    validate_single_result,
    generate_validation_report,
    save_validation_report,
    print_validation_summary,
    parse_resistance_from_filename
)

# Try to import pillow_heif for HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
    print("HEIC support enabled!")
except ImportError:
    HEIC_SUPPORT = False
    print("Warning: pillow-heif not installed. Install with: pip install pillow-heif")
    print("Will skip HEIC files and only process JPG/PNG files.")

# Color mappings
ID_TO_NAME = {
    0: "Background",
    1: "Gold",
    2: "orange",
    3: "green",
    4: "brown",
    5: "background",
    6: "blue",
    7: "yellow",
    8: "black",
    9: "white",
    10: "grey",
    11: "red",
    12: "violet"
}

RESISTOR_COLORS_RGB = {
    "Background": [0, 0, 0],
    "Gold": [212, 175, 55],
    "orange": [255, 165, 0],
    "green": [0, 128, 0],
    "brown": [150, 75, 0],
    "background": [0, 0, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0],
    "black": [255, 105, 180],   # pink mask for black band
    "white": [255, 255, 255],
    "grey": [128, 128, 128],
    "red": [255, 0, 0],
    "violet": [148, 0, 211]
}

def load_model(model_path):
    """Load the Keras model"""
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully!")
    return model

def preprocess_image(image_path, img_size=(256, 256)):
    """Load and preprocess an image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, 0)  # Add batch dimension
    return img_array, img

def ids_to_color(mask_ids):
    """Convert class IDs to RGB color mask"""
    h, w = mask_ids.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for cid in np.unique(mask_ids):
        name = ID_TO_NAME.get(int(cid), "Background")
        rgb = RESISTOR_COLORS_RGB.get(name, [0, 0, 0])
        out[mask_ids == cid] = rgb

    return out

def get_band_category(band_index, total_bands):
    """Get the category name for a band based on its position and total band count."""
    if total_bands == 4:
        # 4-band: Digit1, Digit2, Multiplier, Tolerance
        categories = ["Band 1 (Digit)", "Band 2 (Digit)", "Multiplier", "Tolerance"]
    elif total_bands == 5:
        # 5-band: Digit1, Digit2, Digit3, Multiplier, Tolerance
        categories = ["Band 1 (Digit)", "Band 2 (Digit)", "Band 3 (Digit)", "Multiplier", "Tolerance"]
    elif total_bands == 3:
        # 3-band: Digit1, Digit2, Multiplier (no tolerance visible)
        categories = ["Band 1 (Digit)", "Band 2 (Digit)", "Multiplier"]
    else:
        # Unknown configuration
        categories = [f"Band {i+1}" for i in range(total_bands)]

    if band_index < len(categories):
        return categories[band_index]
    return f"Band {band_index + 1}"


def draw_axis_visualization(ax, axis_info, bands, img_size=256):
    """
    Draw the principal axis and band axes on the segmentation mask.

    Args:
        ax: matplotlib axis to draw on
        axis_info: dict with 'axis_vector', 'axis_origin', 'band_projections'
        bands: list of BandInfo objects
        img_size: size of the image (assumes square)
    """
    if axis_info is None:
        return

    axis_vector = axis_info['axis_vector']
    axis_origin = axis_info['axis_origin']
    band_projections = axis_info.get('band_projections', [])

    # Calculate angle for debug display
    angle_rad = np.arctan2(axis_vector[1], axis_vector[0])
    angle_deg = np.degrees(angle_rad)

    # Calculate perpendicular vector for band axes
    perp_vector = np.array([-axis_vector[1], axis_vector[0]])

    # Draw principal axis - simple approach: extend from origin in both directions
    # Line equation: point = axis_origin + t * axis_vector
    # Extend far enough to cross the image
    t_extent = img_size * 2  # Large enough to span the image
    p1 = axis_origin - t_extent * axis_vector
    p2 = axis_origin + t_extent * axis_vector

    # Draw principal axis as a dashed cyan line
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color='cyan', linewidth=1.5, linestyle='--', alpha=0.7)

    # Draw the axis origin point (should be at centroid of all bands)
    ax.plot(axis_origin[0], axis_origin[1], 'c*', markersize=12,
            markeredgecolor='white', markeredgewidth=1)

    # Draw band axes (perpendicular lines at each band position)
    band_axis_length = 40  # Length of band axis lines on each side

    for i, band in enumerate(bands):
        cx, cy = band.centroid

        # Calculate band axis endpoints
        p1 = np.array([cx, cy]) - band_axis_length * perp_vector
        p2 = np.array([cx, cy]) + band_axis_length * perp_vector

        # Draw band axis as a solid line with slight transparency
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='lime', linewidth=1.5, alpha=0.6)

    # Add debug text showing axis angle and vector
    debug_text = f"Axis: [{axis_vector[0]:.3f}, {axis_vector[1]:.3f}]\nAngle: {angle_deg:.1f}°"
    ax.text(5, img_size - 10, debug_text, fontsize=7, color='yellow',
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # Add small legend in corner (only using empty plots for consistent legend)
    ax.plot([], [], color='cyan', linestyle='--', linewidth=1.5, label='Principal Axis')
    ax.plot([], [], color='lime', linewidth=1.5, label='Band Axes')
    ax.plot([], [], 'c*', markersize=8, label='Axis Origin')
    ax.legend(loc='upper left', fontsize=6, facecolor='black',
              edgecolor='white', labelcolor='white', framealpha=0.7)


def predict_and_visualize(model, image_path, output_dir=None, show_plot=False):
    """Run prediction, calculate resistance, and visualize results"""
    print(f"Processing: {os.path.basename(image_path)}...", end=" ")

    # Preprocess
    img_array, original_img = preprocess_image(image_path)

    # Predict
    pred = model.predict(img_array, verbose=0)[0]
    pred_ids = np.argmax(pred, axis=-1)

    # Convert to color mask
    rgb_mask = ids_to_color(pred_ids)

    # Get detected colors
    unique_ids = np.unique(pred_ids)
    detected_colors = [ID_TO_NAME.get(int(uid), "Unknown")
                       for uid in unique_ids
                       if ID_TO_NAME.get(int(uid), "Unknown") not in ["Background", "background"]]

    # Calculate resistance from the mask (with axis info for visualization)
    resistance_result, axis_info = calculate_resistance_with_axis_info(pred_ids)

    # Format resistance string for display
    if isinstance(resistance_result, ResistanceResult):
        resistance_str = f"{resistance_result.formatted} ±{resistance_result.tolerance}%"
        resistance_value = resistance_result.value
        bands = resistance_result.bands
        band_count = resistance_result.band_count
    else:
        resistance_str = f"Error: {resistance_result.message}"
        resistance_value = None
        # Try to get bands from error result if available
        bands = getattr(resistance_result, 'detected_bands', [])
        band_count = len(bands) if bands else 0

    # Parse true resistance value from filename
    true_value, true_formatted = parse_resistance_from_filename(image_path)
    if true_value is not None:
        true_resistance_str = f"True Value: {true_formatted}"
    else:
        true_resistance_str = "True Value: Unknown"

    # Visualize with black background
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
    for ax in axes:
        ax.set_facecolor('black')

    axes[0].imshow(original_img)
    axes[0].set_title(f"{true_resistance_str}\nOriginal Image", color='white')
    axes[0].axis("off")

    # Segmentation mask with band annotations
    axes[1].imshow(rgb_mask)

    # Draw principal axis and band axes
    if bands:
        draw_axis_visualization(axes[1], axis_info, bands)

    # Add annotations for each detected band
    if bands:
        num_bands = len(bands)
        # Calculate evenly spaced y positions for labels on the right side
        label_x = 270  # Position labels to the right of the 256px image
        label_spacing = 256 / (num_bands + 1)  # Evenly space labels vertically

        for i, band in enumerate(bands):
            # Get centroid position
            cx, cy = band.centroid  # centroid is (x, y)

            # Get band info
            color_name = band.color_name
            category = get_band_category(i, band_count)

            # Create label text (color name and category only)
            label = f"{color_name}\n{category}"

            # Calculate evenly spaced y position for this label
            label_y = label_spacing * (i + 1)

            # Add text annotation with arrow connecting to centroid
            axes[1].annotate(
                label,
                xy=(cx, cy),
                xytext=(label_x, label_y),
                textcoords='data',
                fontsize=7,
                color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                ha='left',
                va='center',
                arrowprops=dict(
                    arrowstyle='-',
                    color='white',
                    linewidth=1,
                    shrinkA=0,
                    shrinkB=5
                )
            )

            # Add a white dot marker at the centroid
            axes[1].plot(cx, cy, 'wo', markersize=6, markeredgecolor='black', markeredgewidth=1)

    # Add band count info at the bottom
    band_type_str = f"{band_count}-band resistor" if band_count > 0 else "No bands detected"
    axes[1].set_title(f"Segmentation Mask\n({band_type_str})", color='white')
    # Extend x-axis to show labels on the right
    axes[1].set_xlim(-10, 380)
    axes[1].set_ylim(266, -10)  # Inverted y-axis for image coordinates
    axes[1].axis("off")

    # Overlay with resistance value
    overlay = np.array(original_img.resize((256, 256))) * 0.5 + rgb_mask * 0.5
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title(f"Resistance: {resistance_str}", color='white')
    axes[2].axis("off")

    plt.tight_layout()

    if output_dir:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_result.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"Saved! Resistance: {resistance_str}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return rgb_mask, pred_ids, detected_colors, resistance_result

def get_next_result_folder(base_dir):
    """
    Create a new result folder with naming: Results_#_dd-mm-yyyy_hh-mm-ss

    Args:
        base_dir: Base inference_results directory

    Returns:
        Path to the new result folder
    """
    from datetime import datetime

    # Get current timestamp (using - instead of / and : which are invalid on Windows)
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")  # dd-mm-yyyy_hh-mm-ss

    # Find the next result number
    existing_folders = [f for f in os.listdir(base_dir) if f.startswith("Results_") and os.path.isdir(os.path.join(base_dir, f))]

    # Extract numbers from existing folders
    max_num = 0
    for folder in existing_folders:
        try:
            # Extract number after "Results_"
            parts = folder.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                max_num = max(max_num, int(parts[1]))
        except (IndexError, ValueError):
            pass

    next_num = max_num + 1
    folder_name = f"Results_{next_num}_{timestamp}"

    return os.path.join(base_dir, folder_name)


def main():
    # Paths (relative to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from scripts/ to project root

    model_path = os.path.join(project_root, "models", "resistor_unet.keras")
    images_dir = os.path.join(project_root, "data", "raw_images")
    base_output_dir = os.path.join(project_root, "data", "inference_results")

    # Create base output directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)

    # Create new result folder with timestamp
    output_dir = get_next_result_folder(base_output_dir)

    # Create output directories (main, Correct, Incorrect)
    os.makedirs(output_dir, exist_ok=True)
    correct_dir = os.path.join(output_dir, "Correct")
    incorrect_dir = os.path.join(output_dir, "Incorrect")
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"  Correct predictions: {correct_dir}")
    print(f"  Incorrect predictions: {incorrect_dir}")

    # Load model
    model = load_model(model_path)

    # List available images
    if HEIC_SUPPORT:
        extensions = ('.heic', '.jpg', '.jpeg', '.png')
    else:
        extensions = ('.jpg', '.jpeg', '.png')

    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(extensions)
                   and not f.startswith('._')]

    if not image_files:
        print("No images found!")
        return

    print(f"\nProcessing {len(image_files)} images...\n")

    # Process all images
    results = []
    validation_results = []

    for i, img_file in enumerate(image_files):
        image_path = os.path.join(images_dir, img_file)
        print(f"[{i+1}/{len(image_files)}] ", end="")
        try:
            _, _, colors, resistance = predict_and_visualize(
                model, image_path, output_dir=output_dir, show_plot=False
            )
            # Get the saved result file path
            base_name = os.path.splitext(img_file)[0]
            result_filename = f"{base_name}_result.png"
            result_path = os.path.join(output_dir, result_filename)

            if isinstance(resistance, ResistanceResult):
                results.append({
                    "file": img_file,
                    "colors": colors,
                    "resistance": resistance.formatted,
                    "resistance_value": resistance.value,
                    "tolerance": resistance.tolerance,
                    "status": "success"
                })
                # Validate against filename
                validation = validate_single_result(img_file, resistance.value)
                validation_results.append(validation)
            else:
                results.append({
                    "file": img_file,
                    "colors": colors,
                    "resistance": None,
                    "resistance_value": None,
                    "error": resistance.message,
                    "status": "resistance_error"
                })
                # Validate with None (calculation error)
                validation = validate_single_result(img_file, None)
                validation_results.append(validation)

            # Move result to Correct or Incorrect folder
            if os.path.exists(result_path):
                target_dir = correct_dir if validation.is_correct else incorrect_dir
                shutil.move(result_path, os.path.join(target_dir, result_filename))
        except Exception as e:
            print(f"Error: {e}")
            results.append({"file": img_file, "colors": [], "status": f"error: {e}"})
            validation = validate_single_result(img_file, None)
            validation_results.append(validation)
            # Move any partial result to Incorrect folder
            base_name = os.path.splitext(img_file)[0]
            result_filename = f"{base_name}_result.png"
            result_path = os.path.join(output_dir, result_filename)
            if os.path.exists(result_path):
                shutil.move(result_path, os.path.join(incorrect_dir, result_filename))

    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    resistance_errors = sum(1 for r in results if r["status"] == "resistance_error")
    correct_count = sum(1 for v in validation_results if v.is_correct)
    incorrect_count = len(validation_results) - correct_count
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"  Successfully calculated resistance: {successful}/{len(image_files)} images")
    if resistance_errors > 0:
        print(f"  Resistance calculation errors: {resistance_errors}")
    print(f"  Correct predictions: {correct_count} (saved to Correct/)")
    print(f"  Incorrect predictions: {incorrect_count} (saved to Incorrect/)")
    print(f"  Results saved to: {output_dir}")

    # Validation report
    print_validation_summary(validation_results)

    # Save detailed validation report
    report_path = os.path.join(output_dir, "validation_report.txt")
    save_validation_report(validation_results, report_path)

if __name__ == "__main__":
    main()
