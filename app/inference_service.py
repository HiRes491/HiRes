from __future__ import annotations

import base64
import os
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "resistor_unet.keras"


def _ensure_scripts_path() -> None:
    scripts_path = str(SCRIPTS_DIR)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)


_ensure_scripts_path()

from color_code_tables import CalculationError, ID_TO_COLOR_NAME, ResistanceResult, VISUALIZATION_COLORS_RGB  # noqa: E402
from resistance_calculator import calculate_resistance_with_axis_info  # noqa: E402


_MODEL: Any | None = None


@dataclass
class InferenceResponse:
    ok: bool
    resistance_text: str | None
    tolerance: float | None
    error: str | None
    overlay_base64: str | None
    predicted_bands: list[str]
    debug_view_base64: str | None = None
    band_distances_px: list[float] = field(default_factory=list)


def _resolve_model_path() -> Path:
    model_path = os.getenv("MODEL_PATH")
    if model_path:
        return Path(model_path).expanduser().resolve()
    return DEFAULT_MODEL_PATH


def get_model() -> Any:
    global _MODEL
    if _MODEL is None:
        model_path = _resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        _MODEL = tf.keras.models.load_model(str(model_path), compile=False)
    return _MODEL


def preprocess_image_bytes(image_bytes: bytes, size: tuple[int, int] = (256, 256)) -> tuple[np.ndarray, np.ndarray]:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    resized = cropped.resize(size, Image.Resampling.BILINEAR)

    image_np = np.asarray(resized, dtype=np.uint8)
    image_for_model = image_np.astype(np.float32) / 255.0
    return np.expand_dims(image_for_model, axis=0), image_np


def predict_mask(model: Any, image_batch: np.ndarray) -> np.ndarray:
    prediction = model.predict(image_batch, verbose=0)[0]
    if prediction.ndim != 3:
        raise ValueError("Model prediction output has unexpected shape.")
    return np.argmax(prediction, axis=-1).astype(np.uint8)


def mask_to_rgb(mask_ids: np.ndarray) -> np.ndarray:
    rgb_mask = np.zeros((*mask_ids.shape, 3), dtype=np.uint8)
    for class_id in np.unique(mask_ids):
        color_name = ID_TO_COLOR_NAME.get(int(class_id), "background")
        rgb = VISUALIZATION_COLORS_RGB.get(color_name, [0, 0, 0])
        rgb_mask[mask_ids == class_id] = np.array(rgb, dtype=np.uint8)
    return rgb_mask


def build_overlay_base64(base_image: np.ndarray, rgb_mask: np.ndarray, alpha: float = 0.5) -> str:
    overlay = ((1.0 - alpha) * base_image.astype(np.float32) + alpha * rgb_mask.astype(np.float32)).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _band_category(index: int, total: int) -> str:
    mapping = {
        3: ["Band 1 (Digit)", "Band 2 (Digit)", "Multiplier"],
        4: ["Band 1 (Digit)", "Band 2 (Digit)", "Multiplier", "Tolerance"],
        5: ["Band 1 (Digit)", "Band 2 (Digit)", "Band 3 (Digit)", "Multiplier", "Tolerance"],
    }
    labels = mapping.get(total)
    if labels and index < len(labels):
        return labels[index]
    return f"Band {index + 1}"


def render_debug_view_base64(rgb_mask: np.ndarray, bands: list, axis_info: dict | None, band_count: int) -> str:
    h, w = rgb_mask.shape[:2]
    fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(rgb_mask)

    perp = None
    if axis_info is not None:
        axis_vec = np.asarray(axis_info["axis_vector"], dtype=float)
        origin = np.asarray(axis_info["axis_origin"], dtype=float)
        perp = np.array([-axis_vec[1], axis_vec[0]])
        t = max(h, w) * 2
        p1 = origin - t * axis_vec
        p2 = origin + t * axis_vec
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color="cyan", linewidth=1.5, linestyle="--", alpha=0.7, label="Principal Axis")
        ax.plot(origin[0], origin[1], "c*", markersize=12,
                markeredgecolor="white", markeredgewidth=1, label="Axis Origin")
        for band in bands:
            cx, cy = band.centroid
            bp1 = np.array([cx, cy]) - 40 * perp
            bp2 = np.array([cx, cy]) + 40 * perp
            ax.plot([bp1[0], bp2[0]], [bp1[1], bp2[1]],
                    color="lime", linewidth=1.5, alpha=0.6)
        ax.plot([], [], color="lime", linewidth=1.5, label="Band Axes")
        angle_deg = float(np.degrees(np.arctan2(axis_vec[1], axis_vec[0])))
        ax.text(5, h - 10,
                f"Axis: [{axis_vec[0]:.3f}, {axis_vec[1]:.3f}]\nAngle: {angle_deg:.1f}°",
                fontsize=7, color="yellow", va="bottom", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

    if bands:
        n = len(bands)
        label_x = w * 1.08
        label_spacing = h / (n + 1)
        for i, band in enumerate(bands):
            cx, cy = band.centroid
            cat = _band_category(i, band_count)
            ax.annotate(
                f"{band.color_name}\n{cat}",
                xy=(cx, cy),
                xytext=(label_x, label_spacing * (i + 1)),
                textcoords="data",
                fontsize=7, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="white", linewidth=1, shrinkA=0, shrinkB=5),
            )
            ax.plot(cx, cy, "wo", markersize=5, markeredgecolor="black", markeredgewidth=1)

    distances = []
    if len(bands) >= 2:
        for i in range(len(bands) - 1):
            c1 = np.array(bands[i].centroid, dtype=float)
            c2 = np.array(bands[i + 1].centroid, dtype=float)
            d = float(np.linalg.norm(c2 - c1))
            distances.append(d)
            mid = (c1 + c2) / 2.0
            offset = (perp if perp is not None else np.array([0.0, -1.0])) * 14.0
            ax.text(mid[0] + offset[0], mid[1] + offset[1], f"{d:.1f}px",
                    fontsize=7, color="#ffd479", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.65))

    title = f"Segmentation ({band_count}-band)" if band_count > 0 else "Segmentation (no bands detected)"
    ax.set_title(title, color="white")
    ax.set_xlim(-10, w * 1.55)
    ax.set_ylim(h + 10, -10)
    ax.axis("off")
    if axis_info is not None:
        ax.legend(loc="upper left", fontsize=6, facecolor="black",
                  edgecolor="white", labelcolor="white", framealpha=0.7)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=110, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), distances


def run_inference(image_bytes: bytes) -> InferenceResponse:
    model = get_model()
    image_batch, resized_image = preprocess_image_bytes(image_bytes)
    mask_ids = predict_mask(model, image_batch)
    rgb_mask = mask_to_rgb(mask_ids)
    overlay_base64 = build_overlay_base64(resized_image, rgb_mask)

    result, axis_info = calculate_resistance_with_axis_info(mask_ids)

    if isinstance(result, ResistanceResult):
        bands = list(result.bands)
        band_count = result.band_count
        resistance_text = result.formatted
        tolerance = float(result.tolerance)
        error = None
        ok = True
    elif isinstance(result, CalculationError):
        bands = list(getattr(result, "detected_bands", []) or [])
        band_count = len(bands)
        resistance_text = None
        tolerance = None
        error = result.message
        ok = False
    else:
        bands = []
        band_count = 0
        resistance_text = None
        tolerance = None
        error = "Unexpected inference result type."
        ok = False

    debug_view_base64, distances = render_debug_view_base64(rgb_mask, bands, axis_info, band_count)

    return InferenceResponse(
        ok=ok,
        resistance_text=resistance_text,
        tolerance=tolerance,
        error=error,
        overlay_base64=overlay_base64,
        predicted_bands=[b.color_name for b in bands],
        debug_view_base64=debug_view_base64,
        band_distances_px=distances,
    )
