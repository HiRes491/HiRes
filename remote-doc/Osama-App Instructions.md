# Osama-App — Running the Phone Camera Webapp

FastAPI backend with phone-first camera UI. Captures live camera frames in the browser, sends them to a server-side U-Net segmentation model, and overlays the decoded resistor value.

---

## One-time setup

### 1. Conda environment

```bash
conda create -n resistor-2 python=3.10
conda activate resistor-2
pip install -r app/requirements.txt
```

### 2. Model weights

Place the trained Keras U-Net at:
```
models/resistor_unet.keras
```

If missing, request from the team — the file is too large for git and is gitignored.

### 3. ngrok (optional — public URL)

Install ngrok, then register your authtoken once:
```bash
ngrok config add-authtoken <your-token>
```

---

## Running the app

### Option 1 — Public URL (phone over internet)

```bash
bash scripts/dev_server.sh
```

Spawns uvicorn on `:8000` plus an ngrok tunnel. Prints a `https://*.ngrok-free.app` URL — open it on the phone.

### Option 2 — LAN only (phone on same WiFi)

```bash
conda run -n resistor-2 --no-capture-output \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Find your PC's LAN IP (`ipconfig` on Windows / `ifconfig` on Linux). Open `http://<your-ip>:8000` on the phone.

### Option 3 — Localhost only (PC browser)

```bash
conda run -n resistor-2 --no-capture-output uvicorn app.main:app
```

Open `http://127.0.0.1:8000` in your browser.

---

## Shell environments

`scripts/dev_server.sh` auto-detects:

- **Git Bash on Windows** ✓ recommended
- **MSYS2** ✓
- **Cygwin** ✓
- **WSL** ⚠ partial — curl/ngrok detection works, but `conda run -n resistor-2` will fail because the conda env lives on the Windows side. Run from Git Bash instead.

Recommended path: **Git Bash on Windows** with conda already initialized in shell.

---

## Useful commands

| Action | Command |
|--------|---------|
| Start full stack with public URL | `bash scripts/dev_server.sh` |
| Start uvicorn only (LAN) | `conda run -n resistor-2 --no-capture-output uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| Custom port | `PORT=9090 bash scripts/dev_server.sh` |
| Health check | `curl http://127.0.0.1:8000/health` |
| Stop everything | `Ctrl+C` (script traps signals and kills both uvicorn + ngrok) |
| View ngrok dashboard | open `http://127.0.0.1:4040` |
| Re-fetch ngrok URL | `curl -s http://127.0.0.1:4040/api/tunnels` |

---

## App endpoints

| Path | Method | Purpose |
|------|--------|---------|
| `/` | GET | Phone camera UI (HTML) |
| `/health` | GET | Liveness probe |
| `/infer` | POST | Upload one image, returns segmentation + resistance reading |
| `/static/*` | GET | JS / CSS / icons |

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `8000` | uvicorn port |
| `MAX_UPLOAD_MB` | `10.0` | Reject uploads larger than this |
| `CAPTURE_INTERVAL_DEFAULT_S` | `0.5` | Phone UI default capture interval (live loop mode) |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `curl missing on PATH` | Ensure curl is reachable. Script auto-checks Git Bash, MSYS2, WSL, Cygwin. If still missing, install via scoop/winget/choco. |
| `ngrok missing on PATH` | Install ngrok and ensure it's on PATH or in `scoop/shims` / `chocolatey/bin`. |
| Phone shows "camera not allowed" | ngrok URL must be HTTPS. Browsers block camera access on `http://`. |
| `conda run` fails on WSL | Run from Git Bash on Windows instead — the conda env is Windows-side. |
| ngrok URL lookup failed | Visit `http://127.0.0.1:4040` in browser to grab tunnel URL manually. |
| Server starts but `/infer` 500s | Check `models/resistor_unet.keras` exists and matches what `app/inference_service.py` expects. |

---

## Project layout (Osama-App branch)

```
ECEN 491/
├── app/
│   ├── main.py                # FastAPI entry
│   ├── inference_service.py   # Model load + inference helpers
│   ├── static/                # Phone UI JS/CSS
│   ├── templates/             # Jinja HTML
│   └── requirements.txt
├── scripts/
│   ├── dev_server.sh          # uvicorn + ngrok launcher
│   ├── band_extractor.py      # Legacy band extraction
│   ├── resistance_calculator.py
│   └── ...
├── models/
│   └── resistor_unet.keras    # Trained U-Net (gitignored)
├── remote-doc/
│   └── Osama-App Instructions.md   # This file
└── README.md
```
