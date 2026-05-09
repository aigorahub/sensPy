# sensPy Sensometrics 2026 Poster

This directory builds a B1 portrait poster for the Sensometrics 2026 abstract:

> Introducing sensPy: Enabling the gold standard analyses of sensR for sensometricians using Python

The poster is modeled as a reproducible artifact pipeline: collect repo-backed
metrics, render chart images, assemble a one-slide PowerPoint, export a print
PNG, and run QA checks.

## Deliverables

- `print_artifacts/senspy-sensometrics-2026-poster.pptx` - editable B1 PowerPoint
- `print_artifacts/senspy-sensometrics-2026-poster.png` - 4175 x 5906 px PNG at 150 DPI

## Build

```bash
bash poster/build.sh
```

The build creates `poster/.venv`, installs poster-only dependencies, writes
`chart_data/`, renders `charts/`, assembles the PPTX, exports the PNG through
LibreOffice and Poppler, and runs `scripts/qa_check.py`. The `chart_data/` and
`charts/` directories are generated intermediates and are intentionally ignored.

Required command-line tools:

- `python3` or `python3.12`
- `soffice` from LibreOffice
- `pdftoppm` from Poppler

The exporter also honors these overrides:

- `POSTER_SOFFICE=/path/to/soffice`
- `POSTER_PDFTOPPM=/path/to/pdftoppm`

## Source Evidence

The poster claims are generated from this repository:

- package version from `pyproject.toml`
- protocol coverage from `senspy/core/types.py` and `senspy/links/double.py`
- sensR fixture version from `tests/fixtures/golden_sensr.json`
- test inventory from AST inspection of `tests/test_*.py`
- public API and dataclass counts from `senspy/__init__.py` and `senspy/**/*.py`

Current headline metrics:

- 13 protocol variants: 8 single + 5 double
- 740 test functions and 851 pytest-collected cases
- sensR fixture version 1.5.3
- 21 typed dataclasses

## Notes

- The poster does not modify sensPy statistical code.
- Charts are generated PNGs embedded in the PowerPoint so the data visuals are reproducible.
- Text/layout remains editable in PowerPoint for final conference or author-line tweaks.
