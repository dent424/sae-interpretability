# Interpret Features

Process features through the full four-agent interpretation pipeline (Theorist → Tester → Critic → Reporter). The Reporter is a post-inquiry compilation step that assembles all artifacts into `results.json` and `report.md`.

**Model:** Sub-agents inherit the model from parent (do NOT specify model parameter).

**Usage:**
- `/interpret-features` — read feature IDs from `output/features_to_interpret.txt`
- `/interpret-features 11328` — single inline feature (overrides the file)
- `/interpret-features 11328 7350 99955` — multiple inline features (overrides the file)
- `/interpret-features --force` — re-interpret features that already have `results.json` (rescues old results.json to `.pre_force.<timestamp>`)
- `/interpret-features --max-batches=5` — stop cleanly after completing 5 batches (graceful cap; remaining features can be resumed with another run)
- `/interpret-features 11328 --force` — re-interpret a specific feature, with rescue

**Batch mode:** This command IS the batch mode. The orchestrator processes features in **parallel batches of 5** by default (configurable). Each phase (Theorist, Tester, Critic, Reporter) spawns all sub-agents for the batch in a single message for true parallelism. There is no separate `/interpret-batch` command.

## PATH CONVENTIONS

Working directory is the **project root** (`Claude Code Folder`).

- **pipeline_utils.py:** `py -3.12 pipeline_utils.py <command>`
- **Feature data:** `"feature data/feature_<N>.json"`
- **Output dir:** `output/interpretations/feature<N>/`
- **List file:** `output/features_to_interpret.txt`
- **Commands dir:** `.claude/commands/` (relative to working directory)

## TOOL RULES (prevents permission prompts)

- **Read files** with the Read tool. NEVER use `cat`, `head`, `tail`, `type`, or any Bash command to read files.
- **Write/edit files** with the Write or Edit tool. NEVER use `echo`, Bash redirection, or heredocs.
- **Search files** with the Glob tool (this is the orchestrator's primary tool for checkpoints). Use Grep when needed.
- **Only these Bash commands are approved:**
  - `py -3.12 pipeline_utils.py <command>` (all pipeline_utils operations)
- Any other Bash command will trigger a permission prompt and block the pipeline.

## INFORMATION ISOLATION RULES

Each sub-agent receives ONLY its command file pointer and feature ID. The orchestrator MUST NOT:
- Read artifact contents (hypotheses.json, phase1_results.json, audit.jsonl) during checkpoints — verify file EXISTENCE only using the Glob tool
- Include any Theorist output in the Tester's prompt
- Summarize or paraphrase one agent's results to another
- Add "helpful context" from a previous agent's return message to a subsequent agent's prompt

**Checkpoints verify file EXISTENCE only (use Glob, not Read).** Two narrow exceptions:
1. The Tester checkpoint reads the FIRST 10 LINES of `phase1_results.json` to detect `"status": "no_winner"` or `"status": "all_zero"`. No other content is read.
2. The extract-interpretation step at the end of each feature reads `results.json` via `pipeline_utils.py extract-interpretation` (the read happens inside the subprocess, not in the orchestrator's context window).

The handoff happens through FILES on disk, not through the orchestrator's prompt.

**Reporter isolation:** The Reporter's output (return message) must not be forwarded to any subsequent feature's Theorist, Tester, or Critic prompts. The Reporter returns only file paths written.

---

## Instructions

### Step 1: Parse Arguments

**Argument resolution:**

1. **No arguments** → read `output/features_to_interpret.txt` (the default list).
2. **Inline feature IDs** (any token that is not a flag) → use them, IGNORE the list file. Print a loud warning: `"NOTE: inline IDs given; IGNORING output/features_to_interpret.txt (N entries in file)."`
3. **`--force` flag** (anywhere in arguments) → re-interpret features that already have `results.json` instead of skipping them. The flag can be combined with either no-args mode or inline-IDs mode.
4. **`--max-batches=N` flag** (anywhere in arguments) → stop cleanly after completing N full batches. Parse as `--max-batches=<integer>`. Validate: N must be a positive integer (≥ 1); reject `0`, negatives, or non-integers with a clear error. If omitted, process all batches. Combines with `--force` and inline IDs.

If no inline IDs and the list file doesn't exist OR is empty (after stripping comments), display:
```
Usage: /interpret-features                         # read output/features_to_interpret.txt
       /interpret-features <id> [<id>...]          # inline IDs (override file)
       /interpret-features --force                 # re-interpret done features
       /interpret-features --max-batches=N         # stop after N batches
       /interpret-features <id> --force            # re-interpret a specific feature
```
and exit cleanly.

### Step 1.5: Parse `output/features_to_interpret.txt` (if used)

When reading the list file:
- Open with encoding `utf-8-sig` (strips the BOM that Notepad on Windows adds).
- Strip whitespace including `\r` from each line.
- Skip blank lines and lines starting with `#`.
- **Strip inline comments:** `12345  # paralinguistic candidate` → `12345`.
- **Tolerate `id|category|subcategory` format** by splitting on `|` and taking the first field. This makes it safe to copy-paste lines from the existing `output/features_to_verify.txt` (which uses pipe-delimited format).
- Convert remaining tokens to int.
- **Validate `0 <= id < N_LATENTS`** by reading N_LATENTS via `py -3.12 pipeline_utils.py get-provenance` and parsing the JSON. Reject negative IDs and IDs ≥ N_LATENTS with line numbers and a clear error.
- On `ValueError` for non-integer tokens, report which line failed.
- Tolerate duplicates by deduplicating with a warning (don't silently discard).
- Print to stderr the parsed file path and the count of valid IDs: `"Parsed N valid feature IDs from output/features_to_interpret.txt"`.

Display: "Will process {N} features in batches of {batch_size}: {list}"

### Step 2: Pre-Flight Checks

**Step 2.1: Check pre-computed feature data exists for every requested feature.**

For each parsed ID, Glob for `feature data/feature_<ID>.json`. Collect any missing IDs into `missing_precomputed`.

**If ANY are missing → STOP with full list:**
```
ERROR: missing pre-computed feature data for these IDs:
  - 12345
  - 67890
Generate them first with:
  py -3.12 run_modal_utf8.py analyze_feature_json --feature-idx <ID> --output-dir "feature data"
Note: this command also creates a stray feature data/interpretations/ subdirectory — manually remove that subdir after running.
```
Do NOT process anything. Exit non-zero.

**Step 2.2: Check existing results.json for skip/force handling.**

For each remaining ID, Glob for `output/interpretations/feature<ID>/results.json`.

- **Without `--force`:** features with `results.json` → skip with `"feature N: already has results.json (skipping; use --force to reinterpret)"`.
- **With `--force`:** features with `results.json` → queue for processing AND mark as `needing_rescue`.

**Step 2.3: Empty-list early exits.**

- If the parsed list has 0 valid IDs (empty file, all comments, all rejected by validation), print `"Nothing to interpret: 0 features in features_to_interpret.txt (or all comments/blank)"` and exit cleanly with status 0. Do NOT enter the batch loop.
- If after all filtering the to-process list is empty, print `"Nothing to interpret: <X> features queued, <Y> skipped (already done), <Z> missing precomputed"` and exit cleanly with status 0. Do NOT enter the batch loop.

**Step 2.4: Cost preflight summary.**

Print a one-time cost estimate before entering the batch loop:
```
Processing N features in batches of {batch_size}.
Of these, F are --force re-interpretations (existing results.json will be rescued to .pre_force.* before running).
Estimated wall clock: ~{N/batch_size * 8} minutes (roughly 5–10 min per feature).
Estimated Modal calls: {N * 5} (Tester:1 + Critic:4 per feature).
Modal scaledown_window=900s, so within-feature calls share a warm container.
Cross-feature parallelism may incur extra cold starts.
```

**Step 2.5: Modal preflight (recommended).**

Call `py -3.12 pipeline_utils.py preflight-modal` once before any sub-agents spawn. The preflight verifies BOTH Modal connectivity AND that the `token_activations` field is present in the response (catches a missing/incomplete Step 4 Modal edit before any sub-agent burns time). Skip this step only if the user added `--skip-preflight`.

If preflight fails, exit non-zero with the error message.

### Step 3: Process Features in Parallel Batches

Default `batch_size = 5` (configurable, matching Modal's parallel capacity). Split the to-process list into chunks of `batch_size`.

**If `--max-batches=N` was set**, the loop runs only the first `N` batches (1-indexed); features in batches `N+1` and beyond are NOT processed and are collected into a `not_yet_processed` list for the final summary. If `N >= num_batches`, the flag is a no-op and all batches run.

**For each batch (from i=1 to min(num_batches, max_batches if set else num_batches)):**

Display: `"--- Batch {i}/{num_batches}: Processing features {list} ---"` (when capped, also display `"(cap: --max-batches={N})"` on the first batch so it's visible.)

#### 3a. Stale Artifact Cleanup AND Force-Rescue Ordering

For each feature in the batch, perform these operations in this exact order:

1. **Run `clean-partial`:**
   ```bash
   py -3.12 pipeline_utils.py clean-partial --feature {ID}
   ```
   This removes stale partial-run artifacts (audit.jsonl, hypotheses.json, batch_test_*.json, challenge_*.json) but PRESERVES any file matching `results.json*` (covers `results.json` itself AND any `.pre_force.<timestamp>` rescue copies from prior `--force` runs).

2. **For features marked `needing_rescue` (--force only):**
   - Print a loud one-line WARN: `"WARN feature {ID}: --force will rescue results.json but DELETE legacy artifacts (verification.md, audit_report.md, ablation_*.json, audit.jsonl). Only results.json is preserved."`
   - Get a Windows-safe timestamp:
     ```bash
     py -3.12 pipeline_utils.py timestamp
     ```
     This emits ISO 8601 BASIC format like `20260408T195703Z` (no colons — Windows NTFS forbids `:` in filenames).
   - Rename `output/interpretations/feature{ID}/results.json` → `output/interpretations/feature{ID}/results.json.pre_force.<timestamp>` using the Edit tool to a non-existent destination, OR using a one-liner:
     ```bash
     py -3.12 -c "import pathlib; src = pathlib.Path('output/interpretations/feature{ID}/results.json'); dst = src.parent / f'results.json.pre_force.<timestamp>'; src.rename(dst); print(f'rescued -> {dst.name}')"
     ```
   - Print: `"feature {ID}: rescued existing results.json → results.json.pre_force.<timestamp>"`

**Why the order matters:** clean-partial first, rescue second. If you rescue first and then run clean-partial, the rescue file (`.pre_force.<timestamp>`) does not match the exact name `results.json` and clean-partial would delete it. Belt-and-suspenders: clean-partial's skip rule covers `results.json*` so even if order were swapped, the rescue file would survive — but explicit ordering is clearer.

#### 3b. Theorist Sub-Agents (parallel within batch)

For EACH feature in the batch, spawn a sub-agent. **CRITICAL: launch ALL Theorist sub-agents for the batch in a SINGLE message containing multiple Agent tool calls — this is what makes them run in parallel. Separate messages would run them sequentially.**

Each sub-agent gets this EXACT prompt (replace `{ID}` with the feature index):

```
Generate hypotheses for SAE feature {ID}.

1. Read the full instructions from: .claude/commands/theorize.md
2. Follow ALL steps in that file for feature {ID} — wherever the command file uses a feature-index placeholder, substitute {ID}
3. Produce outputs in: output/interpretations/feature{ID}/
4. Your job ends when hypotheses.json is written. Do NOT proceed to testing.

IMPORTANT: Working directory is the project root. Use paths as specified in the command file.
```

**Do NOT modify, extend, or add context to this prompt template.**

Wait for ALL Theorist sub-agents in the batch to complete.

#### 3c. Theorist Checkpoint (per feature, Glob only)

For each feature in the batch, use the Glob tool to check whether `output/interpretations/feature{ID}/hypotheses.json` exists. Do NOT use the Read tool.

- If present: feature passed Theorist phase.
- If missing: mark feature as FAILED, skip Tester + Critic + Reporter for it.

Display: `"Theorist complete. Passed: {list}. Failed: {list}."`

#### 3d. Tester Sub-Agents (parallel within batch)

For each feature that PASSED the Theorist checkpoint, spawn a sub-agent. **Launch ALL passed Testers in a SINGLE message.**

Each gets this EXACT prompt:

```
Test hypotheses for SAE feature {ID}.

1. Read the full instructions from: .claude/commands/test-hypotheses.md
2. Follow ALL steps in that file for feature {ID} — wherever the command file uses a feature-index placeholder, substitute {ID}
3. Read ONLY these files for context:
   - output/interpretations/feature{ID}/hypotheses.json
   - feature data/feature_{ID}.json
4. Produce outputs in: output/interpretations/feature{ID}/
5. Your job ends when phase1_results.json is written. Do NOT proceed to challenge.

IMPORTANT: Working directory is the project root. Use paths as specified in the command file.
```

**Do NOT modify, extend, or add context to this prompt template. Do NOT include any information from the Theorist's output or return messages.**

Wait for ALL Tester sub-agents in the batch to complete.

#### 3e. Tester Checkpoint (per feature)

For each feature, use the Glob tool to check whether `output/interpretations/feature{ID}/phase1_results.json` exists.

- If missing: mark feature as FAILED, skip Critic + Reporter.
- If present: Read the FIRST 10 LINES of the file. Look for `"status": "no_winner"` OR `"status": "all_zero"`. If either is present:
  - Display `"Feature {ID} FAILED at Tester phase ({status} — no hypothesis reached threshold OR all activations zero)."`
  - Mark FAILED, skip Critic + Reporter.
- Otherwise: feature passed Tester phase.

**Note:** This is the ONLY checkpoint where reading file content is permitted (and only the first 10 lines, only for status detection).

Display: `"Tester complete. Passed: {list}. Failed: {list}."`

#### 3f. Critic Sub-Agents (parallel within batch)

For each feature that PASSED the Tester checkpoint, spawn a sub-agent. **Launch ALL passed Critics in a SINGLE message.**

Each gets this EXACT prompt:

```
Challenge the interpretation of SAE feature {ID}.

1. Read the full instructions from: .claude/commands/challenge.md
2. Follow ALL steps in that file for feature {ID} — wherever the command file uses a feature-index placeholder, substitute {ID}
3. You are the Critic. You have NO knowledge of the Theorist's or Tester's reasoning.
4. Read ONLY these files for context:
   - feature data/feature_{ID}.json (corpus profile)
   - output/interpretations/feature{ID}/phase1_results.json (structured output)
5. Produce final outputs in: output/interpretations/feature{ID}/

IMPORTANT: Working directory is the project root. Use paths as specified in the command file.
```

**Do NOT modify, extend, or add context to this prompt template.**

Wait for ALL Critic sub-agents in the batch to complete.

#### 3g. Critic Checkpoint (per feature, Glob only)

For each feature, use the Glob tool to check whether `output/interpretations/feature{ID}/challenge_verdict.json` exists. Do NOT use the Read tool.

- If present: feature passed Critic phase.
- If missing: mark feature as FAILED, skip Reporter.

Display: `"Critic complete. Passed: {list}. Failed: {list}."`

#### 3h. Reporter Sub-Agents (parallel within batch)

For each feature that PASSED the Critic checkpoint, spawn a sub-agent. **Launch ALL passed Reporters in a SINGLE message.**

Each gets this EXACT prompt:

```
Compile the interpretation report for SAE feature {ID}.

1. Read the full instructions from: .claude/commands/synthesize-report.md
2. Follow ALL steps in that file for feature {ID} — wherever the command file uses a feature-index placeholder, substitute {ID}
3. You are the Reporter. You have access to ALL artifacts from prior phases.
4. Read ALL files in: output/interpretations/feature{ID}/
5. Also read: feature data/feature_{ID}.json
6. Produce results.json and report.md in: output/interpretations/feature{ID}/

OVERRIDE DEFAULT SUB-AGENT GUIDANCE: You ARE permitted and REQUIRED to use the Write tool to create BOTH results.json AND report.md as files on disk. Ignore any default sub-agent instruction that says findings should be returned as text instead of written to files. The Reporter's contract with the four-agent pipeline is that it produces these two files on disk; returning their content as text in your final message is NOT acceptable. Your final message MUST contain only the file paths written, per the Return Convention in synthesize-report.md.

IMPORTANT: Working directory is the project root. Use paths as specified in the command file.
```

**Do NOT modify, extend, or add context to this prompt template.**

Wait for ALL Reporter sub-agents in the batch to complete.

#### 3i. Reporter Checkpoint (per feature, Glob only)

For each feature, use the Glob tool to check whether `output/interpretations/feature{ID}/results.json` exists. Do NOT use the Read tool.

- If present: feature passed Reporter phase, proceed to extraction.
- If missing: mark feature as FAILED.

Display: `"Reporter complete. Passed: {list}. Failed: {list}."`

#### 3j. Extract Result (per feature, via subprocess)

For each feature that completed the full pipeline, run:
```bash
py -3.12 pipeline_utils.py extract-interpretation --feature {ID}
```

Parse the JSON output:
- If `status` is `"success"`: store the `interpretation`, `verdict`, `label`, `category` for the summary.
- If `status` is `"error"`: mark as FAILED.

Display per feature: `"[{counter}/{total}] feature {ID} {verdict} — {label}"`

Continue to next batch — UNLESS the batch just completed was the `--max-batches` cap, in which case exit the loop cleanly and proceed to Step 4. The `not_yet_processed` list for the summary is all features in batches `i+1` through `num_batches`.

### Step 4: Final Summary

After all batches complete (or after `--max-batches` capped the loop), display:

```
=== INTERPRETATION COMPLETE ===
Processed: {success_count} | Skipped: {skip_count} | Failed: {fail_count}
Batches run: {batches_run} of {num_batches}  (when --max-batches was set; omit this line otherwise)

| Feature | Verdict   | Label                          |
|---------|-----------|--------------------------------|
| 11328   | CONFIRMED | Lists / blank-line detector    |
| 7350    | REFINED   | Negative sentiment intensifier |

Skipped (already complete, use --force to redo): 82498
Failed:
  - 12345 (Tester: no_winner)
  - 67890 (Critic: missing challenge_verdict.json)

Remaining (failed or unprocessed due to --max-batches; copy-paste back into output/features_to_interpret.txt if needed):
  12345
  67890
  # unprocessed (max-batches cap):
  21562
  6162
  13177
```

When `--max-batches` capped the loop, split the Remaining section into two sub-groups — the failed features (with their failure reason) and the unprocessed features (annotated with `# unprocessed (max-batches cap):`). Both are valid to re-queue.

**Implicit checkpointing:** The skip-if-results.json semantics provide implicit resume-on-restart. Re-running `/interpret-features` after a crash (or after `--max-batches`) continues from where the previous run stopped (unless `--force` is set). No manual recovery needed.

---

## Error Handling

- **Empty list:** Exit cleanly with status 0 and `"Nothing to interpret"` message.
- **All features missing pre-computed data:** STOP with full list and the regen command. Do not process anything.
- **Missing hypotheses.json:** Theorist failed, mark as FAILED, skip Tester + Critic + Reporter
- **Missing phase1_results.json:** Tester failed, mark as FAILED, skip Critic + Reporter
- **phase1_results.json with `no_winner` or `all_zero`:** Tester found no usable hypothesis, mark as FAILED, skip Critic + Reporter
- **Missing challenge_verdict.json after Critic:** Critic failed, mark as FAILED, skip Reporter + extraction
- **Missing results.json after Reporter:** Reporter failed, mark as FAILED
- **Malformed JSON:** `extract-interpretation` returns error status, mark as FAILED
- **Stale partial artifacts:** Cleaned automatically in Step 3a before each feature
- **Concurrent runs:** No lockfile. Do NOT run `/interpret-features` in two Claude Code sessions simultaneously against overlapping feature IDs — the two orchestrators will race.

---

## Begin

1. Parse arguments (Step 1) and the list file if no inline IDs (Step 1.5)
2. Run pre-flight checks (Step 2)
3. Process features in parallel batches (Step 3)
4. Display final summary (Step 4)
