# Project Learnings

## Repo Conventions

- 2026-05-07: Keep poster work under `poster/` and do not modify `senspy` statistical code for a communications artifact.

## Validation and Tooling

- 2026-05-07: Global `python3` in this environment does not have `pytest`; poster tooling should create and use `poster/.venv`.
- 2026-05-07: Poster export requires LibreOffice (`soffice`) and Poppler (`pdftoppm`) on PATH, with tool paths overridable in scripts when needed.
- 2026-05-07: Poster helper scripts should remain Python 3.10-compatible; use `tomli` as the `tomllib` fallback when needed.
- 2026-05-07: Do not hide converter subprocess output in poster exports; visible LibreOffice and Poppler logs make PR/CI failures easier to diagnose.
- 2026-05-07: Poster metric collectors should prefer package source-of-truth APIs for protocol facts and warn when static AST fallbacks undercount dynamic pytest parametrizations.
- 2026-05-07: External model review should use an explicitly configured credential or authenticated CLI session; do not search unrelated repositories for API keys.

## Review Heuristics

- 2026-05-07: Poster claims should be grounded in repo evidence: README, `pyproject.toml`, `senspy/__init__.py`, tests, and fixture metadata.

## Product and Domain Invariants

- 2026-05-07: sensPy is positioned as a Python port of sensR for Thurstonian sensory discrimination methods, preserving numerical parity while adding Python-native ergonomics.

## Known Traps

- 2026-05-07: The abstract does not include an author list; avoid inventing a full scientific author list beyond conservative Aigora/Sensometrics attribution.

## Retired Learnings

- None yet.
