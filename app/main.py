# Inference runs synchronously; clients should set their own request timeout.
import logging
import os

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.inference_service import get_model, run_inference

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

def _read_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default

MAX_UPLOAD_MB = _read_float_env("MAX_UPLOAD_MB", 10.0)
CAPTURE_INTERVAL_DEFAULT_S = _read_float_env("CAPTURE_INTERVAL_DEFAULT_S", 0.5)

app = FastAPI(title="Phone Camera Inference API")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.on_event("startup")
async def _startup_warmup() -> None:
    try:
        get_model()
    except Exception as exc:
        logging.getLogger(__name__).warning("Model warm-up skipped: %s", exc)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/config")
async def config() -> dict:
    return {
        "capture_interval_default_s": CAPTURE_INTERVAL_DEFAULT_S,
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.post("/api/predict")
async def predict(image: UploadFile | None = File(default=None)):
    if image is None:
        raise HTTPException(status_code=400, detail="Missing image upload in form field 'image'.")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Uploaded file must be an image.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.",
        )

    try:
        result = run_inference(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "ok": result.ok,
        "resistance_text": result.resistance_text,
        "tolerance": result.tolerance,
        "error": result.error,
        "overlay_base64": result.overlay_base64,
        "predicted_bands": result.predicted_bands,
        "debug_view_base64": result.debug_view_base64,
        "band_distances_px": result.band_distances_px,
    }
