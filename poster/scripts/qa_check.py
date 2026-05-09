"""QA checks for the generated sensPy poster artifacts."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[2]
POSTER = ROOT / "poster"
ASSETS = POSTER / "assets"
CHARTS = POSTER / "charts"
CHART_DATA = POSTER / "chart_data"
OUT = POSTER / "print_artifacts"
PPTX = OUT / "senspy-sensometrics-2026-poster.pptx"
PNG = OUT / "senspy-sensometrics-2026-poster.png"

REQUIRED_CHARTS = [
    "protocol_coverage.png",
    "psychometric_curves.png",
    "test_inventory.png",
    "roc_bridge.png",
    "architecture_pipeline.png",
]
REQUIRED_ASSETS = [
    "qr-senspy-github.png",
    "sensometrics-2026-logo.png",
]


def fail(message: str) -> None:
    print(f"[qa] FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def assert_nonblank(path: Path, min_stddev: float = 5.0) -> None:
    if not path.exists():
        fail(f"missing {path.relative_to(ROOT)}")
    with Image.open(path) as img:
        stat = ImageStat.Stat(img.convert("L"))
        if stat.stddev[0] < min_stddev:
            fail(f"{path.relative_to(ROOT)} appears blank")


def check_summary() -> None:
    summary_path = CHART_DATA / "summary.json"
    if not summary_path.exists():
        fail("missing chart_data/summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "version": "0.2.0",
        "single_protocol_count": 8,
        "double_protocol_count": 5,
        "total_protocol_variants": 13,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            fail(f"summary {key}={summary.get(key)!r}, expected {value!r}")
    if summary.get("test_functions", 0) < 740:
        fail("summary reports fewer than 740 test functions")


def check_artifacts() -> None:
    if not PPTX.exists() or PPTX.stat().st_size < 100_000:
        fail("missing or tiny PPTX artifact")
    assert_nonblank(PNG, min_stddev=10.0)
    with Image.open(PNG) as img:
        width, height = img.size
        if width < 4174 or height < 5905:
            fail(f"PNG dimensions too small: {width} x {height}")


def check_pptx_text() -> None:
    banned = ["TODO", "TBD", "lorem ipsum", "/Users/", "/opt/homebrew"]
    with zipfile.ZipFile(PPTX) as zf:
        text = "\n".join(
            zf.read(name).decode("utf-8", errors="ignore")
            for name in zf.namelist()
            if name.startswith("ppt/slides/") and name.endswith(".xml")
        )
    lower_text = text.lower()
    for token in banned:
        if token.lower() in lower_text:
            fail(f"PPTX contains banned placeholder/path token: {token}")


def main() -> None:
    check_summary()
    for asset in REQUIRED_ASSETS:
        assert_nonblank(ASSETS / asset, min_stddev=2.0)
    for chart in REQUIRED_CHARTS:
        assert_nonblank(CHARTS / chart)
    check_artifacts()
    check_pptx_text()
    print("[qa] poster QA passed")


if __name__ == "__main__":
    main()
