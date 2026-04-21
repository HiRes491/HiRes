const videoEl = document.getElementById("camera-preview");
const frozenImgEl = document.getElementById("camera-frozen");
const canvasEl = document.getElementById("snapshot-canvas");
const cameraStatusEl = document.getElementById("camera-status");
const resistanceTextEl = document.getElementById("resistance-text");
const statusTextEl = document.getElementById("status-text");
const resultDetailEl = document.getElementById("result-detail");
const overlayImgEl = document.getElementById("overlay-preview");
const overlayPlaceholderEl = document.getElementById("overlay-placeholder");
const segmentationImgEl = document.getElementById("segmentation-view");
const segmentationPlaceholderEl = document.getElementById("segmentation-placeholder");
const captureIntervalInputEl = document.getElementById("capture-interval");

const startBtn = document.getElementById("start-live-btn");
const stopBtn = document.getElementById("stop-live-btn");
const takePhotoBtn = document.getElementById("take-photo-btn");
const flipBtn = document.getElementById("flip-camera-btn");

let activeStream = null;
let facingMode = "environment";
let liveLoopTimer = null;
let liveLoopEnabled = false;
let requestInFlight = false;

function setStatus(message, isError = false) {
  statusTextEl.textContent = message;
  statusTextEl.setAttribute("data-state", isError ? "error" : "ok");
}

function setResultPayload(payload) {
  const resistanceText = payload?.resistance_text ?? null;
  const tolerance = payload?.tolerance ?? null;
  const resistanceValue =
    resistanceText != null && tolerance != null
      ? `${resistanceText} ±${tolerance}%`
      : resistanceText ?? "--";
  const statusValue = payload?.ok ? "Prediction received" : "Prediction failed";

  resistanceTextEl.textContent = String(resistanceValue);
  setStatus(statusValue, !payload?.ok);
  resultDetailEl.textContent = JSON.stringify(payload, null, 2);
}

function setError(message, detail = null) {
  resistanceTextEl.textContent = "--";
  setStatus(message, true);
  resultDetailEl.textContent = JSON.stringify(
    { ok: false, error: message, detail },
    null,
    2,
  );
}

function setRequestError(message, detail = null) {
  setStatus(message, true);
  resultDetailEl.textContent = JSON.stringify(
    { ok: false, error: message, detail },
    null,
    2,
  );
}

function getCaptureIntervalMs() {
  const defaultMs = 500;
  if (!captureIntervalInputEl) {
    return defaultMs;
  }

  const seconds = Number.parseFloat(captureIntervalInputEl.value);
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return defaultMs;
  }

  return Math.max(100, Math.round(seconds * 1000));
}

function stopLiveLoop() {
  liveLoopEnabled = false;
  if (liveLoopTimer !== null) {
    clearInterval(liveLoopTimer);
    liveLoopTimer = null;
  }
}

function stopCameraStream() {
  stopLiveLoop();
  if (activeStream) {
    for (const track of activeStream.getTracks()) {
      track.stop();
    }
  }
  activeStream = null;
  videoEl.srcObject = null;
  cameraStatusEl.textContent = "Camera is stopped.";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  takePhotoBtn.disabled = true;
}

async function startCameraStream() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setError("Camera API not supported on this device.");
    return;
  }

  stopCameraStream();
  clearFrozenFrame();
  cameraStatusEl.textContent = "Starting camera...";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: facingMode },
      },
      audio: false,
    });

    activeStream = stream;
    videoEl.srcObject = stream;
    await videoEl.play();
    cameraStatusEl.textContent = `Camera running (${facingMode}).`;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    takePhotoBtn.disabled = false;
  } catch (error) {
    stopCameraStream();
    setError("Unable to start camera.", String(error));
  }
}

function startLiveLoop() {
  if (!activeStream) {
    cameraStatusEl.textContent = "Live loop paused: camera unavailable.";
    return;
  }

  stopLiveLoop();
  liveLoopEnabled = true;
  const intervalMs = getCaptureIntervalMs();
  cameraStatusEl.textContent = `Live loop running (${(intervalMs / 1000).toFixed(1)}s interval).`;

  const tick = async () => {
    if (!liveLoopEnabled || requestInFlight || !activeStream) {
      return;
    }

    await snapOnce("live");
  };

  liveLoopTimer = window.setInterval(tick, intervalMs);
  tick();
}

function captureFrameAsBlob() {
  return new Promise((resolve, reject) => {
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    if (!vw || !vh) {
      reject(new Error("Camera frame is not ready."));
      return;
    }

    const side = Math.min(vw, vh);
    const sx = Math.floor((vw - side) / 2);
    const sy = Math.floor((vh - side) / 2);

    canvasEl.width = side;
    canvasEl.height = side;
    const ctx = canvasEl.getContext("2d");
    if (!ctx) {
      reject(new Error("2D canvas context unavailable."));
      return;
    }

    ctx.drawImage(videoEl, sx, sy, side, side, 0, 0, side, side);
    canvasEl.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("Failed to encode image blob."));
          return;
        }
        resolve(blob);
      },
      "image/jpeg",
      0.92,
    );
  });
}

function renderOverlay(previewBlob, payload) {
  const overlaySource =
    payload?.overlay_image_url ??
    payload?.overlayUrl ??
    payload?.overlay_base64 ??
    null;

  if (typeof overlaySource === "string" && overlaySource.length > 0) {
    if (overlaySource.startsWith("data:")) {
      overlayImgEl.src = overlaySource;
    } else if (overlaySource.startsWith("http") || overlaySource.startsWith("/")) {
      overlayImgEl.src = overlaySource;
    } else {
      overlayImgEl.src = `data:image/png;base64,${overlaySource}`;
    }
  } else {
    overlayImgEl.src = URL.createObjectURL(previewBlob);
  }

  overlayImgEl.hidden = false;
  overlayPlaceholderEl.hidden = true;

  const debugSource = payload?.debug_view_base64 ?? null;
  if (typeof debugSource === "string" && debugSource.length > 0) {
    segmentationImgEl.src = debugSource.startsWith("data:")
      ? debugSource
      : `data:image/png;base64,${debugSource}`;
    segmentationImgEl.hidden = false;
    segmentationPlaceholderEl.hidden = true;
  }
}

async function snapOnce(source = "manual") {
  if (requestInFlight) {
    return;
  }

  if (!activeStream) {
    await startCameraStream();
    if (!activeStream) {
      setRequestError("Camera stream unavailable.");
      return;
    }
  }

  requestInFlight = true;
  cameraStatusEl.textContent =
    source === "live" ? "Live capture in progress..." : "Capturing frame...";

  try {
    const blob = await captureFrameAsBlob();
    const formData = new FormData();
    formData.append("image", blob, "snapshot.jpg");

    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (response.status === 413) {
      stopLiveLoop();
      setRequestError("Image too large for server.", payload?.detail ?? null);
      cameraStatusEl.textContent = "Capture failed.";
      return;
    }
    if (!response.ok || payload?.ok === false) {
      const detail = payload?.detail ?? payload?.error ?? "Request failed.";
      setRequestError("Prediction request failed.", detail);
      cameraStatusEl.textContent = "Capture failed.";
      return;
    }

    setResultPayload(payload);
    renderOverlay(blob, payload);
    cameraStatusEl.textContent =
      source === "live" ? "Live loop running." : "Frame captured.";
  } catch (error) {
    setRequestError("Snap once failed.", String(error));
    cameraStatusEl.textContent = "Capture failed.";
  } finally {
    requestInFlight = false;
  }
}

function showFrozenFrame(blob) {
  frozenImgEl.src = URL.createObjectURL(blob);
  frozenImgEl.hidden = false;
  videoEl.hidden = true;
}

function clearFrozenFrame() {
  if (frozenImgEl.src) URL.revokeObjectURL(frozenImgEl.src);
  frozenImgEl.removeAttribute("src");
  frozenImgEl.hidden = true;
  videoEl.hidden = false;
}

async function takePhoto() {
  if (!activeStream) return;
  stopLiveLoop();
  takePhotoBtn.disabled = true;

  for (let i = 0; i < 40 && requestInFlight; i++) {
    await new Promise((r) => setTimeout(r, 50));
  }

  let blob;
  try {
    blob = await captureFrameAsBlob();
  } catch (error) {
    setRequestError("Frame capture failed.", String(error));
    cameraStatusEl.textContent = "Capture failed.";
    stopCameraStream();
    return;
  }

  showFrozenFrame(blob);
  if (activeStream) {
    for (const track of activeStream.getTracks()) track.stop();
    activeStream = null;
    videoEl.srcObject = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  cameraStatusEl.textContent = "Frame locked. Uploading...";

  requestInFlight = true;
  try {
    const formData = new FormData();
    formData.append("image", blob, "snapshot.jpg");
    const response = await fetch("/api/predict", { method: "POST", body: formData });
    const payload = await response.json();
    if (response.status === 413) {
      setRequestError("Image too large for server.", payload?.detail ?? null);
      cameraStatusEl.textContent = "Upload failed (too large).";
      return;
    }
    if (!response.ok || payload?.ok === false) {
      const detail = payload?.detail ?? payload?.error ?? "Request failed.";
      setRequestError("Prediction request failed.", detail);
      cameraStatusEl.textContent = "Prediction failed.";
      return;
    }
    setResultPayload(payload);
    renderOverlay(blob, payload);
    cameraStatusEl.textContent = "Frame captured. Tap Start Live to resume.";
  } catch (error) {
    setRequestError("Upload failed.", String(error));
    cameraStatusEl.textContent = "Upload failed.";
  } finally {
    requestInFlight = false;
  }
}

async function startLiveCapture() {
  await startCameraStream();
  if (!activeStream) {
    setRequestError("Live loop unavailable: camera stream failed to start.");
    return;
  }

  startLiveLoop();
}

async function flipCamera() {
  const wasLive = liveLoopEnabled;
  stopLiveLoop();
  facingMode = facingMode === "environment" ? "user" : "environment";
  cameraStatusEl.textContent = `Switching camera (${facingMode})...`;
  if (activeStream) {
    await startCameraStream();
  }
  if (wasLive && activeStream) {
    startLiveLoop();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  fetch("/api/config")
    .then((r) => r.json())
    .then((cfg) => {
      if (typeof cfg.capture_interval_default_s === "number") {
        captureIntervalInputEl.value = cfg.capture_interval_default_s.toFixed(1);
      }
      const hintEl = document.getElementById("capture-hint");
      if (hintEl && typeof cfg.max_upload_mb === "number") {
        hintEl.textContent += ` (server max upload: ${cfg.max_upload_mb} MB)`;
      }
    })
    .catch(() => {});
});

startBtn.addEventListener("click", startLiveCapture);
stopBtn.addEventListener("click", stopCameraStream);
takePhotoBtn.addEventListener("click", takePhoto);
flipBtn.addEventListener("click", flipCamera);
captureIntervalInputEl?.addEventListener("change", () => {
  if (liveLoopEnabled) {
    startLiveLoop();
  }
});
