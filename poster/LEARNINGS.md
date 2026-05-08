# Project Learnings

## Repo Conventions

- 2026-05-07: Keep poster work under `poster/` and do not modify `senspy` statistical code for a communications artifact.

## Validation and Tooling

- 2026-05-07: Global `python3` in this environment does not have `pytest`; poster tooling should create and use `poster/.venv`.
- 2026-05-07: LibreOffice export is available at `/opt/homebrew/bin/soffice`; PNG rasterization can use `/opt/homebrew/bin/pdftoppm`.

## Review Heuristics

- 2026-05-07: Poster claims should be grounded in repo evidence: README, `pyproject.toml`, `senspy/__init__.py`, tests, and fixture metadata.

## Product and Domain Invariants

- 2026-05-07: sensPy is positioned as a Python port of sensR for Thurstonian sensory discrimination methods, preserving numerical parity while adding Python-native ergonomics.

## Known Traps

- 2026-05-07: The abstract does not include an author list; avoid inventing a full scientific author list beyond conservative Aigora/Sensometrics attribution.

## Retired Learnings

- None yet.
