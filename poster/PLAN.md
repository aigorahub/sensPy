# Plan: sensPy Sensometrics 2026 Poster

## Mission

Create a print-ready Sensometrics 2026 poster for the abstract, "sensPy: Bringing the gold standard of sensR to the Python sensory ecosystem." The deliverable should feel like a sibling to the `cd-database-proto` Sensometrics poster: editorial, data-forward, and suitable for fabric printing. Done means the repo contains a reproducible poster build pipeline plus final PNG and PPTX artifacts.

## Scope

### In Scope
- A B1 portrait conference poster at 27.83 in x 39.37 in.
- A reproducible `poster/build.sh` pipeline that creates chart data, renders charts, assembles a PowerPoint, exports a print PNG, and runs QA.
- Poster content grounded in this repo: supported protocols, sensR parity, test inventory, typed dataclasses, SciPy architecture, and Plotly visualizations.
- Final artifacts in `poster/print_artifacts/`.

### Out of Scope
- Changes to the `senspy` package API or statistical behavior.
- Publishing to PyPI, docs hosting, or conference submission systems.
- Full brand-system reconstruction from the CD poster. This poster will echo the style with available local fonts and colors.
- Inventing new scientific results beyond the abstract and repo evidence.

## Batches

### Batch 1: Session Setup And Poster Story

**Tasks:**
- [ ] Create Elves session docs and `.elves-session.json`.
- [ ] Define the poster narrative spine and content map.
- [ ] Record project evidence from README, docs, tests, and package metadata.

**Acceptance criteria:**
- [ ] `poster/PLAN.md`, `poster/SURVIVAL_GUIDE.md`, `poster/EXECUTION_LOG.md`, and `poster/LEARNINGS.md` exist.
- [ ] `.elves-session.json` records finite mode, branch, batch list, and stop guard.
- [ ] Poster story names only repo-backed claims or explicitly documented judgment calls.

**Docs likely touched:** poster run docs.

**Risk:** The abstract does not list authors; the run must choose a conservative presenter/affiliation line and document that choice.

### Batch 2: Data And Visual Proof Objects

**Tasks:**
- [ ] Add scripts to collect repo metrics and chart data.
- [ ] Render visual proof objects: protocol coverage, psychometric curves, validation/test inventory, ROC/SDT bridge, and architecture pipeline.
- [ ] Add requirements and a build script that creates a local poster venv.

**Acceptance criteria:**
- [ ] `poster/chart_data/summary.json` reports the package version, 8 single protocols, 5 double protocols, and >=740 test functions.
- [ ] Chart PNGs are non-empty and generated under `poster/charts/`.
- [ ] The scripts run from a clean checkout using `bash poster/build.sh`.

**Docs likely touched:** poster README and execution log.

**Risk:** Local Python lacks project dependencies, so the build must isolate dependencies in `poster/.venv`.

### Batch 3: Poster Assembly And Export

**Tasks:**
- [ ] Build a one-slide editable PPTX at exact B1 dimensions.
- [ ] Export a 150 DPI PNG via LibreOffice and Poppler.
- [ ] Use the CD poster as a taste reference while adapting the story to sensPy.

**Acceptance criteria:**
- [ ] `poster/print_artifacts/senspy-sensometrics-2026-poster.pptx` exists.
- [ ] `poster/print_artifacts/senspy-sensometrics-2026-poster.png` exists and is approximately 4175 x 5906 px.
- [ ] The rendered poster is visually inspected for layout, legibility, nonblank charts, and no obvious overlap.

**Docs likely touched:** poster README and execution log.

**Risk:** PowerPoint-to-PNG export can shift fonts or line breaks; QA must inspect the actual rendered PNG, not only the PPTX.

### Batch 4: QA, Validation, And Handoff

**Tasks:**
- [ ] Run poster QA checks.
- [ ] Run relevant repo validation after the poster venv is available.
- [ ] Update repository README.md so the public protocol table and test count agree with repo-derived poster claims.
- [ ] Commit and push the completed branch.
- [ ] Provide a concise handoff with artifact paths and verification notes.

**Acceptance criteria:**
- [ ] `python poster/scripts/qa_check.py` passes.
- [ ] `python -m pytest` passes or any unrelated/pre-existing failure is documented.
- [ ] Git status is clean except pre-existing untracked files that were present before the run.

**Docs likely touched:** execution log, survival guide, README.

**Risk:** Full pytest may be slower than the poster task; touched-surface poster QA is the required gate, full pytest is best-effort branch confidence.

## Non-Negotiables

- Do not modify `senspy` statistical code or tests for this poster task.
- Do not invent quantitative validation claims; compute or cite them from repo files.
- Do not remove or alter the pre-existing untracked `.agents/` directory.
- Commit messages must follow the Elves progress format.
- Never merge; the user controls merge/review decisions.

## Test Strategy

- **Primary poster gate:** `bash poster/build.sh`
- **Poster QA:** `poster/.venv/bin/python poster/scripts/qa_check.py`
- **Repo validation:** `poster/.venv/bin/python -m pytest -q`
- **Artifact inspection:** open and visually inspect `poster/print_artifacts/senspy-sensometrics-2026-poster.png`
- **Known environment note:** global `python3` has no `pytest`; the poster pipeline creates `poster/.venv`.

## Batch Sizing

```yaml
team-size: 4
sprint-length: 2 weeks
notes: This is a finite, single-deliverable run; batches are smaller than default Elves batches so each artifact layer can be validated.
```

## Notes

- Reference poster: sibling `cd-database-proto` Sensometrics poster at `poster/print_artifacts/sensometrics-2026-poster.png` when that repo/worktree is available locally.
- The previous poster’s print target is B1 portrait at 27.83 in x 39.37 in and 150 DPI.
- The sensPy repo is at `v0.2.0`; AST inspection finds 740 test functions, and pytest collects 851 cases in the poster venv.
- Use `sensR v1.5-3`/`v1.5.3` wording carefully; repo docs use both representations.
