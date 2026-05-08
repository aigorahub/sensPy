#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSTER_DIR="$ROOT_DIR/poster"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.12)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Error: python3.12 or python3 not found on PATH." >&2
    exit 1
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"

if [[ ! -d "$POSTER_DIR/.venv" ]]; then
  "$PYTHON_BIN" -m venv "$POSTER_DIR/.venv"
fi

"$POSTER_DIR/.venv/bin/python" -m pip install --upgrade pip
"$POSTER_DIR/.venv/bin/python" -m pip install -r "$POSTER_DIR/requirements.txt"

"$POSTER_DIR/.venv/bin/python" "$POSTER_DIR/scripts/collect_metrics.py"
"$POSTER_DIR/.venv/bin/python" "$POSTER_DIR/scripts/render_charts.py"
"$POSTER_DIR/.venv/bin/python" "$POSTER_DIR/scripts/build_pptx.py"
"$POSTER_DIR/.venv/bin/python" "$POSTER_DIR/scripts/export_png.py"
"$POSTER_DIR/.venv/bin/python" "$POSTER_DIR/scripts/qa_check.py"
