"""Export the sensPy poster PPTX to a print PNG."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
POSTER = ROOT / "poster"
OUT = POSTER / "print_artifacts"
PPTX = OUT / "senspy-sensometrics-2026-poster.pptx"
PDF = OUT / "senspy-sensometrics-2026-poster.pdf"
PNG = OUT / "senspy-sensometrics-2026-poster.png"

DPI = 150
EXPECTED_W = 4174
EXPECTED_H = 5905


def tool(name: str, env_name: str) -> str:
    override = os.environ.get(env_name)
    if override:
        if Path(override).exists():
            return override
        sys.exit(f"[export] {env_name} points to missing path: {override}")
    resolved = shutil.which(name)
    if resolved:
        return resolved
    sys.exit(f"[export] missing required tool on PATH: {name}")


def main() -> None:
    if not PPTX.exists():
        sys.exit(f"[export] missing {PPTX.relative_to(ROOT)}")

    soffice = tool("soffice", "POSTER_SOFFICE")
    pdftoppm = tool("pdftoppm", "POSTER_PDFTOPPM")
    OUT.mkdir(parents=True, exist_ok=True)

    print("[export] pptx -> pdf")
    subprocess.run(
        [
            soffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(OUT),
            str(PPTX),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    if not PDF.exists():
        sys.exit("[export] LibreOffice did not produce PDF")

    print(f"[export] pdf -> png at {DPI} DPI")
    tmp_root = OUT / "senspy-sensometrics-2026-poster_tmp"
    subprocess.run(
        [pdftoppm, "-r", str(DPI), "-png", str(PDF), str(tmp_root)],
        check=True,
        capture_output=True,
        text=True,
    )

    produced = OUT / "senspy-sensometrics-2026-poster_tmp-1.png"
    if not produced.exists():
        sys.exit(f"[export] missing expected raster output {produced.name}")
    shutil.move(str(produced), str(PNG))
    PDF.unlink(missing_ok=True)

    with Image.open(PNG) as img:
        width, height = img.size
    print(f"[export] PNG dimensions: {width} x {height}")
    if width < EXPECTED_W or height < EXPECTED_H:
        sys.exit(f"[export] PNG too small: got {width} x {height}")
    print(f"[export] wrote {PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
