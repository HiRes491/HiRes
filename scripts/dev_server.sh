#!/usr/bin/env bash
# Launch uvicorn + ngrok together. Ctrl+C stops both.
# Run from an environment that already has `uvicorn` + `curl` on PATH
# (e.g. inside the activated `resistor-2` conda env, or via `conda run`).
set -e
PORT="${PORT:-8000}"

# Augment PATH with common Windows tool locations (idempotent)
# Covers Git Bash (/c/...), MSYS2 (/c/...), WSL (/mnt/c/...), and Cygwin (/cygdrive/c/...)
for extra in \
  "/c/Windows/System32" \
  "/c/Windows/system32" \
  "/mnt/c/Windows/System32" \
  "/cygdrive/c/Windows/System32" \
  "/mingw64/bin" \
  "/usr/bin" \
  "$HOME/AppData/Local/Microsoft/WinGet/Links" \
  "/mnt/c/ProgramData/chocolatey/bin" \
  "/c/ProgramData/chocolatey/bin" \
  "$HOME/scoop/shims"; do
  case ":$PATH:" in
    *":$extra:"*) ;;
    *) [ -d "$extra" ] && export PATH="$PATH:$extra" ;;
  esac
done

# Resolve curl/ngrok binary names — on WSL/Cygwin, native `command -v` does
# not auto-resolve `.exe`, so check both forms and stash absolute paths.
resolve_tool() {
  local name="$1"
  command -v "$name" 2>/dev/null && return 0
  command -v "${name}.exe" 2>/dev/null && return 0
  for p in \
    "/mnt/c/Windows/System32/${name}.exe" \
    "/c/Windows/System32/${name}.exe" \
    "/mingw64/bin/${name}.exe" \
    "/cygdrive/c/Windows/System32/${name}.exe" \
    "$HOME/scoop/shims/${name}.exe" \
    "/mnt/c/ProgramData/chocolatey/bin/${name}.exe" \
    "/c/ProgramData/chocolatey/bin/${name}.exe"; do
    [ -x "$p" ] && echo "$p" && return 0
  done
  return 1
}
CURL="$(resolve_tool curl)"  || { echo "curl missing on PATH"; exit 1; }
NGROK="$(resolve_tool ngrok)" || { echo "ngrok missing on PATH"; exit 1; }

# Resolve conda if not on PATH (common when outer shell is fish/zsh)
if ! command -v conda >/dev/null 2>&1; then
  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/d/Anaconda3/etc/profile.d/conda.sh" \
    "/c/ProgramData/Anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$candidate" ] && . "$candidate" && break
  done
fi

cleanup() { kill $UVI_PID $NGROK_PID 2>/dev/null || true; exit 0; }
trap cleanup INT TERM

conda run -n resistor-2 --no-capture-output \
  uvicorn app.main:app --host 0.0.0.0 --port "$PORT" &
UVI_PID=$!

until "$CURL" -sf "http://127.0.0.1:$PORT/health" >/dev/null; do sleep 1; done
echo "[dev_server] uvicorn ready on :$PORT"

"$NGROK" http "$PORT" --log=stdout >/dev/null 2>&1 &
NGROK_PID=$!

URL=""
for i in {1..20}; do
  URL=$("$CURL" -sf http://127.0.0.1:4040/api/tunnels 2>/dev/null \
    | grep -oE 'https://[a-z0-9-]+\.ngrok-free\.[a-z]+' | head -1)
  [ -n "$URL" ] && break
  sleep 0.5
done

if [ -n "$URL" ]; then
  echo ""
  echo "  ┌─────────────────────────────────────────────────────────┐"
  printf "  │  Phone URL: %-44s│\n" "$URL"
  echo "  └─────────────────────────────────────────────────────────┘"
  echo ""
else
  echo "[dev_server] ngrok URL lookup failed — check http://127.0.0.1:4040"
fi

wait
