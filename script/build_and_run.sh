#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/opt/anaconda3/envs/py311/bin/python"
PORT="${WEB_PORT:-7860}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"

# Stop the previous dashboard instance if it is still holding the default port.
if command -v lsof >/dev/null 2>&1; then
  mapfile -t pids < <(lsof -ti "tcp:${PORT}" || true)
  if [[ "${#pids[@]}" -gt 0 ]]; then
    kill "${pids[@]}" || true
    sleep 1
  fi
fi

export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" webapp.py
