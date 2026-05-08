# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART

## Mission

Build a print-ready Sensometrics 2026 poster for sensPy using a recoverable Elves-style run. The final branch should contain a reproducible poster pipeline plus PNG and PPTX artifacts under `poster/print_artifacts/`.

## Run Control

- **Run mode:** finite
- **Stop policy:** stop only after all planned batches are complete, a genuine blocker appears, or the user explicitly stops the run
- **User intent:** "plan it out using [$elves](/Users/johnennis/.codex/skills/elves/SKILL.md) then do it"
- **Checkpoint due by:** none
- **Checkpoint semantics:** none
- **May continue after checkpoint:** yes
- **Actual stop conditions:** all four poster batches complete, or a hard blocker with no viable workaround
- **Final-response policy:** disallowed until all planned batches are complete or blocked
- **Batch completion rule:** Every completed batch ends with execution log update, survival guide update, commit, and push when possible.
- **Re-read rule:** Immediately after every commit and push, re-read this survival guide before doing anything else.

## Session Budget

- **Started:** 2026-05-07 20:30 EDT
- **User returns:** not specified
- **Checkpoint expectation:** final poster artifacts in this turn
- **Time budget:** finite, best effort until complete
- **Average batch time so far:** ~5m
- **Batches remaining:** 3 of 4

## Stop Gate

- **Planned batches remaining:** 3
- **Stop allowed right now:** no
- **Why:** poster data, charts, PowerPoint, PNG, and QA are still incomplete
- **Next required action:** start Batch 2 data and chart pipeline

## Effort Standard

- Work as hard as possible for the full run.
- Maintain the same level of effort on QA as on visual assembly.
- Do not stop at a first successful export; inspect the rendered artifact.

## Memory Surfaces

- **Plan:** `poster/PLAN.md`
- **Survival guide:** `poster/SURVIVAL_GUIDE.md`
- **Learnings:** `poster/LEARNINGS.md`
- **Execution log:** `poster/EXECUTION_LOG.md`
- **Structured state:** `.elves-session.json`
- **Durable docs manifest:** `.ai-docs/manifest.md`

## Non-Negotiables

- Do not modify `senspy` statistical code or tests for this poster task.
- Do not invent quantitative validation claims; compute or cite them from repo files.
- Do not remove or alter the pre-existing untracked `.agents/` directory.
- Never merge. The user reviews and merges.
- Never run destructive git commands: `git reset --hard`, `git checkout .`, `git clean -fd`, `git push --force`, or shared-branch rebase.

## Launch Readiness

- [x] Plan cleaned and saved to disk
- [x] Survival guide updated from the current plan
- [x] Learnings file initialized
- [x] Execution log initialized with batch breakdown and preflight notes
- [x] Branch created or confirmed
- [x] PR opened or existing PR recorded
- [x] Preflight run and critical failures cleared
- [x] Run mode, return time, and non-negotiables recorded
- [x] Stop Gate initialized with `Stop allowed right now: no`
- [x] Launch prompt implicit in user command: continue through planned batches without pausing

## Current Phase

**Status:** In progress

**Active batch:** Batch 2: Data And Visual Proof Objects

**What was just finished:** Batch 1 setup committed, pushed, tagged, and PR #1 opened.

**Single next action:** Add poster requirements, build script, metrics collection, and chart rendering scripts.

## Active Compute

No active paid or long-running compute.

## Next Exact Batch

**Batch:** 2: Data And Visual Proof Objects

**Scope:**
- Add a self-contained poster build environment.
- Collect repo-backed metrics and write chart data.
- Render protocol coverage, psychometric, validation, ROC, and architecture charts.

**Acceptance criteria:**
- [ ] `poster/chart_data/summary.json` reports version, protocol counts, and >=740 test functions.
- [ ] `poster/charts/*.png` files are generated and nonblank.
- [ ] `bash poster/build.sh` reaches chart rendering.

**Risk:** Dependency installation must stay isolated under `poster/.venv`.

**Rollback tag:** `elves/pre-batch-2`

## Tool Configuration

```yaml
lint: poster/.venv/bin/python -m compileall poster/scripts
typecheck:
build: bash poster/build.sh
test: poster/.venv/bin/python -m pytest -q
e2e:
smoke: poster/.venv/bin/python poster/scripts/qa_check.py
review: github-pr-comments
notification: pr-comment
```

## Plan and Log Paths

- **Plan:** `poster/PLAN.md`
- **Learnings:** `poster/LEARNINGS.md`
- **Execution log:** `poster/EXECUTION_LOG.md`
- **Durable docs manifest:** `.ai-docs/manifest.md`
- **Architecture doc:** `.ai-docs/architecture.md`
- **Conventions doc:** `.ai-docs/conventions.md`
- **Gotchas doc:** `.ai-docs/gotchas.md`
- **Branch:** `codex/senspy-sensometrics-poster`
- **PR number:** #1
- **Plan hash at session start:** `799508c5f4fcd399a7f3b1cc8447d3db`

## After Any Compaction

1. Read this file first.
2. Read `.elves-session.json`.
3. Read `poster/LEARNINGS.md`.
4. Read `poster/PLAN.md`.
5. Read `poster/EXECUTION_LOG.md`.
6. Continue with the first incomplete batch named in the Stop Gate or structured state.

# READ THIS FILE FIRST AFTER ANY COMPACTION OR RESTART
