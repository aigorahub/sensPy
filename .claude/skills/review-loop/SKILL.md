---
name: review-loop
description: Submit code for automated review via Command Center. Triggers build verification, quality checks, and merge readiness analysis. Iterates on feedback until merge-ready.
argument-hint: [optional description of what the code does]
---

# Review Loop Skill

You are an autonomous coding agent running a review loop. You trigger Command Center's review pipeline, wait for results, fix any blocking issues yourself, and repeat until the PR is merge-ready.

**IMPORTANT: This is a loop. You MUST keep going until the review passes or max attempts are exhausted. Do NOT stop after receiving feedback — fix the issues and re-trigger.**

## Phase 1: Setup & Context Detection

1. Resolve configuration. Check these sources in order for each value:

   **APP_URL** (Command Center endpoint):
   - Default: `https://command-center.aigora.ai`
   - Override: `CC_APP_URL` env var, or `APP_URL` in `.env.local`

   **API_SECRET** (service-to-service auth):
   - Shell env var: `CC_API_SECRET`
   - Fallback: `INTERNAL_API_SECRET` in `.env.local`
   - If neither exists, tell the user to set `CC_API_SECRET` in their shell profile and stop.

   ```bash
   # Read CC_API_SECRET from env, fall back to .env.local (handles quoted values)
   echo "${CC_API_SECRET:-$(python3 -c "
for line in open('.env.local'):
  if line.strip().startswith('INTERNAL_API_SECRET='):
    print(line.strip().split('=',1)[1].strip('\"').strip(\"'\"))
" 2>/dev/null)}"
   ```

   ```bash
   # Read APP_URL from env, fall back to .env.local (handles quoted values)
   echo "${CC_APP_URL:-$(python3 -c "
for line in open('.env.local'):
  if line.strip().startswith('APP_URL='):
    print(line.strip().split('=',1)[1].strip('\"').strip(\"'\"))
" 2>/dev/null)}"
   ```

2. Detect repo and branch from git:
   ```bash
   git remote get-url origin 2>/dev/null | sed 's|.*github.com[:/]||;s|\.git$||'
   ```
   ```bash
   git branch --show-current
   ```

3. Find the PR number:
   ```bash
   gh pr view --json number -q .number 2>/dev/null
   ```
   If no PR exists, tell the user to open a PR first and stop.

4. Build the prompt text. Check these sources in order:
   - If the user provided `$ARGUMENTS`, use that as the prompt.
   - Otherwise, look for a plan file in `docs/plans/` or `.ai-docs/plans/` that mentions the branch name.
   - Otherwise, extract a prompt from the PR title and body:
     ```bash
     gh pr view --json title,body -q '"Review: " + .title + "\n\nPR Description:\n" + .body'
     ```
   - If `gh` fails or returns empty, use a generic prompt: "Review the code changes on this branch for quality and merge readiness."

## Phase 2: Trigger the Review

Call the trigger API using `curl`, saving the response to a file. Use `python3` to parse the saved JSON (never pipe curl to python3 — connection hiccups cause empty stdin and `JSONDecodeError`).

```bash
curl -s -X POST "${APP_URL}/api/orchestration/trigger" \
  -H "Content-Type: application/json" \
  -H "x-internal-api-secret: ${API_SECRET}" \
  -d '{"action":"start","repo":"REPO","branch":"BRANCH","prNumber":PR_NUM,"prompt":"PROMPT_TEXT"}' \
  -o /tmp/cc-trigger.json
```

```bash
python3 -c "import json; data=json.load(open('/tmp/cc-trigger.json')); print(data.get('sessionId', data.get('error', 'Unknown response')))"
```

Extract `sessionId` from the parsed response. If the response contains an error, show it and stop.

Tell the user: "Review triggered (session: SESSION_ID). Waiting for results..."

## Phase 3: Poll for Results

Poll the status endpoint. **Wait 30 seconds between polls.** Use `sleep 30` between each curl call.

Save the response to a file, then parse it:

```bash
curl -s "${APP_URL}/api/orchestration/status?repo=REPO&sessionId=SESSION_ID&orgId=legacy-default" \
  -H "x-internal-api-secret: ${API_SECRET}" \
  -o /tmp/cc-review-status.json
```

```bash
python3 -c "
import json
data = json.load(open('/tmp/cc-review-status.json'))
state = data.get('state', {})
phase = state.get('phase', 'unknown')
print(f'Phase: {phase}')
if state.get('error'):
    print(f'Error: {state[\"error\"]}')
if state.get('lastLogMessage'):
    print(f'Log: {state[\"lastLogMessage\"]}')
"
```

Check `state.phase` against this table:

| Phase                | Action                                        |
|----------------------|-----------------------------------------------|
| `completed`          | Go to Phase 4 (success)                       |
| `failed`             | Go to Phase 4 (failure)                       |
| `waiting_for_nudge`  | Go to Phase 5 (fix blocking issues)           |
| `pending`            | Just started — poll again after 30s           |
| `routing`            | Auto-detecting Vercel project — poll again    |
| `dispatching`        | Starting checks — poll again                  |
| `executing`          | Running analysis — poll again                 |
| `verifying_build`    | Checking Vercel build — poll again            |
| `quality_check`      | Running quality review — poll again           |
| `reviewing`          | Running merge readiness — poll again          |
| Anything else        | Show phase + latest log message, poll again   |

After parsing poll results, print a progress summary:

```
Iteration {N} — Phase: {phase}
  Blocking: {count} | Resolved: {delta.resolvedCount} | New: {delta.newCount} | Regressed: {delta.regressedCount}
  Stale: {findingsStale} | Commit: {analyzedCommitSha}
```

**Keep polling until you reach one of the terminal/actionable states.** Do NOT give up after a few polls.

## Phase 4: Terminal States

### Success (phase=completed)
Tell the user:
- The review passed
- Show the PR URL
- Show any non-blocking warnings from `state.lastMergeResult.warnings`
- The PR is ready for human review and merge

### Failure (phase=failed)
Tell the user:
- Show `state.error`
- Show what's still failing from `state.lastQualityResult` and `state.lastMergeResult`
- Suggest manual fixes if max rework attempts were exhausted

## Phase 5: Fix Blocking Issues (THE CORE LOOP)

**This is the most important phase. You are an autonomous agent — fix the issues yourself.**

Print a progress summary at the start of each iteration:

```
Iteration {N} — Phase: {phase}
  Blocking: {count} | Resolved: {delta.resolvedCount} | New: {delta.newCount} | Regressed: {delta.regressedCount}
  Stale: {findingsStale} | Commit: {analyzedCommitSha}
```

1. **Extract findings** from the status response. Prefer structured findings when available:
   - `state.structuredFindings` — array of structured findings with `findingId`, `title`, `severity`, `evidence`, `file`, `line`, `acceptanceCriteria`, and `status`
   - `state.findingDelta` — shows `newCount`, `resolvedCount`, `regressedCount`, `remainingCount` since last iteration
   - `state.findingsStale` — if `true`, analysis is in progress; wait and re-poll before acting
   - **Fallback** (if structuredFindings is null): `state.lastMergeResult.blockingIssues`, `state.lastMergeResult.summary`, `state.lastQualityResult.findings`

   ### Triage Findings Before Fixing

   Not all blocking findings are your responsibility. Triage each finding before acting:

   - **Check `firstSeenIteration`**: If a finding's `firstSeenIteration` is > 1, it pre-dates your changes and is likely a pre-existing issue.
   - **Check if the file was modified by you**: Run `git diff --name-only HEAD~1` to get your modified files. If a finding's `file` is NOT in that list, it's likely pre-existing.
   - **Priority order**: Fix findings in YOUR modified files first. Findings in untouched files are likely pre-existing — skip them with a justification commit (see false-positive escape below).
   - **High `newCount` but all in unmodified files?** That's pre-existing noise, not your fault. Skip with justification.

   ### Interpret `findingDelta` to Guide Your Actions

   Use the delta to understand what happened since your last push:

   - `resolvedCount > 0` → Your last push fixed things. Good progress — keep going.
   - `newCount > 0` in YOUR files → Your fix introduced new issues. Address these.
   - `newCount > 0` in unmodified files → Pre-existing issues surfaced by deeper analysis. Consider skipping.
   - `regressedCount > 0` → Previously resolved findings came back. Check if you accidentally reverted something.
   - `remainingCount == 0 && newCount == 0` → All issues addressed. This iteration will likely pass.

2. **Read and understand each blocking issue.** Common issue types:
   - **Code bugs** (e.g., case-sensitivity, missing error handling) → Fix the code
   - **Documentation updates** (e.g., "TODO.md needs update", "CLAUDE.md needs update") → Update the docs
   - **Skeleton regeneration** → Run `pnpm regen-skeleton` if available
   - **Test failures** → Fix the failing tests

3. **Fix each issue** using your normal code editing tools (Read, Edit, Write). Do thorough fixes — don't just add comments or TODOs.

4. **Commit and push** the fixes. Use descriptive commit messages:
   ```bash
   git add <specific-files-you-changed>
   ```
   ```bash
   git commit -m "fix: address review feedback - <brief description>"
   ```
   ```bash
   git push
   ```

   **False-positive escape**: If you determine a blocking finding is a false positive (the code is correct and the reviewer is wrong), you can skip it by pushing a commit with this message format:
   ```
   review: skip finding <findingId> — <brief technical justification>
   ```
   The reviewer will evaluate your justification. If it's valid, the finding will be resolved automatically. If not, it will be re-flagged at high confidence. Only use this for genuine false positives — do not abuse it to skip real issues.

   **Batching skips**: If multiple findings are false positives, batch all skips into a single commit with each on its own line:
   ```
   review: skip findings

   skip finding <findingId1> — <justification1>
   skip finding <findingId2> — <justification2>
   skip finding <findingId3> — <justification3>
   ```

5. **Verify push and wait for bot reviews.** The quality check and merge readiness analysis read PR review comments — bot reviewers (linters, security scanners) need time to post theirs after a new push.
   ```bash
   PUSHED_SHA=$(git rev-parse --short HEAD)
   ```
   Tell the user: "Pushed $PUSHED_SHA, waiting 3 minutes for bot reviews before nudging re-review..."
   ```bash
   sleep 180
   ```

6. **Nudge the orchestrator** to re-review. Save the response to a file:
   ```bash
   curl -s -X POST "${APP_URL}/api/orchestration/trigger" \
     -H "Content-Type: application/json" \
     -H "x-internal-api-secret: ${API_SECRET}" \
     -d '{"action":"fix_pushed","sessionId":"SESSION_ID"}' \
     -o /tmp/cc-nudge.json
   ```

   Tell the user: "Pushed $PUSHED_SHA, nudging re-review."

7. **Go back to Phase 3** (poll again). The orchestrator will re-run build verification, quality check, and merge readiness on your new code.

**Repeat Phase 3→5 until the review passes or fails permanently.**

## Important Notes

- **Do NOT use `jq`** — it may not be available. Use `python3` to parse JSON from saved files: `python3 -c "import json; data=json.load(open('/tmp/cc-review-status.json')); print(data['key'])"`. Never pipe curl output directly to python3 — save to a file first to avoid `JSONDecodeError` on connection hiccups.
- **Do NOT run long-lived bash loops** — poll by making individual curl calls with `sleep 30` between them.
- **Always push before nudging** — the orchestrator checks the latest commit on the branch.
- **Be thorough with fixes** — superficial fixes will just fail the next review cycle.
- **Max 3 rework cycles** — after that the orchestrator gives up. Make each fix count.
