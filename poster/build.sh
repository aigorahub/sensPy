#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSTER_DIR="$ROOT_DIR/poster"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

cd "$ROOT_DIR"

if [[ ! -d "$POSTER_DIR/.venv" ]]; then
  "$PYTHON_BIN" -m venv "$POSTER_DIR/.venv"
fi

"$POSTER_DIR/.venv/bin/python" -m pip install --upgrade pip
"$POSTER_DIR/.venv/bin/python" -m pip install -r "$POSTER_DIR/requirements.txt"

"$POSTER_DIR/.venv/bin/python" "$POSTER_DIR/scripts/collect_metrics.py"
"$POSTER_DIR/.venv/bin/python" "$POSTER_DIR/scripts/render_charts.py"
