# Audit SAE Feature Interpretation

Perform comprehensive post-interpretation audit of GPT-2 SAE features. Combines structural validation, process verification, provenance checking, logic assessment, causal masking review, and optional reproducibility testing into a 7-layer audit.

This file has **two roles**:
1. **Batch orchestrator** (top half) — runs when a user types `/audit` directly. Parses the list file or inline IDs, runs pre-flight checks, spawns parallel sub-agents, and produces an aggregate summary.
2. **Per-feature Auditor** (bottom half, starting at Step A.0) — runs inside each sub-agent spawned by the orchestrator. Executes the 7-layer audit on one feature and writes `audit_report.md`.

---

## SUB-AGENT BYPASS (read this first)

**If your invoking prompt starts with "Perform the full 7-layer audit for SAE feature {ID}" and tells you to "Follow ALL steps in that file for feature {ID}", you are a spawned Auditor sub-agent. SKIP THE ENTIRE "BATCH ORCHESTRATION" SECTION BELOW AND JUMP DIRECTLY TO "Step A.0: Load Artifacts".** Treat the feature ID given in your prompt as the feature-index value throughout the per-feature steps. Do not parse the list file. Do not spawn other sub-agents. Do not write an aggregate summary.

Otherwise (you were invoked directly via the `/audit` slash command), continue reading below as the Batch Orchestrator.

---

## BATCH ORCHESTRATION

### Usage

| Invocation | Behavior |
|---|---|
| `/audit` | Read feature IDs from `output/features_to_interpret.txt` |
| `/audit <id> [<id>...]` | Inline IDs override the file (loud warning) |
| `/audit --force` | Re-audit features that already have `audit_report.md` (rescue old report) |
| `/audit --skip-repro` | Skip Layer 7 Modal calls for all features in the batch |
| `/audit --skip-preflight` | Skip the `preflight-modal` connectivity check (mirrors `/interpret-features`) |
| `/audit --max-batches=N` | Stop cleanly after completing N batches (graceful cap; remaining features resumable) |
| `/audit <id> --force` | Combines inline + force |
| `/audit --force --skip-repro` | Combines flags |

Flags are order-independent and combinable. Any token starting with `--` is a flag; everything else is a feature ID. `--max-batches=N` must parse as a positive integer (`N >= 1`); reject `0`, negatives, or non-integers with a clear error.

### PATH CONVENTIONS (orchestrator)

Working directory is the **project root** (`Claude Code Folder`).

- **pipeline_utils.py:** `py -3.12 pipeline_utils.py <command>`
- **Feature data:** `"feature data/feature_<N>.json"`
- **Output dir:** `output/interpretations/feature<N>/`
- **List file:** `output/features_to_interpret.txt`
- **Command file:** `.claude/commands/audit.md` (this file)

### TOOL RULES (orchestrator)

Same rules as the per-feature Auditor below: use Read, Glob, Grep, Write, Edit; avoid `cat`/`grep`/`find`/`ls`; approved Bash commands are limited to `py -3.12 pipeline_utils.py <command>` (and Modal calls are issued only inside sub-agents, not the orchestrator).

### INFORMATION ISOLATION RULES

- Each sub-agent receives ONLY its command file pointer and feature ID. The orchestrator MUST NOT summarize one audit's results to another sub-agent's prompt.
- Checkpoints verify file EXISTENCE only (use Glob, not Read), with one narrow exception: the post-sub-agent header parse reads the FIRST 10 LINES of `audit_report.md` to extract the `**Overall:** <TOKEN>` line — no other content.

### Step 1: Parse Arguments

1. **No arguments** → read `output/features_to_interpret.txt`.
2. **Inline feature IDs** (any token that is not a flag) → use them, IGNORE the list file. Print: `"NOTE: inline IDs given; IGNORING output/features_to_interpret.txt (N entries in file)."`
3. **Flags** (`--force`, `--skip-repro`, `--skip-preflight`, `--max-batches=N`) — any token starting with `--`. Order-independent, combinable. For `--max-batches=N`, parse N as a positive integer (≥ 1); reject `0`, negatives, or non-integers with a clear error message and exit.

If no inline IDs and the list file doesn't exist OR is empty after stripping comments, display:
```
Usage: /audit                              # read output/features_to_interpret.txt
       /audit <id> [<id>...]               # inline IDs (override file)
       /audit --force                      # re-audit done features
       /audit --skip-repro                 # skip Layer 7 Modal calls
       /audit --skip-preflight             # skip Modal connectivity preflight
       /audit --max-batches=N              # stop after N batches
       /audit <id> --force --skip-repro    # combine flags
```
and exit cleanly.

### Step 1.5: Parse `output/features_to_interpret.txt` (if used)

Same rules as `/interpret-features` Step 1.5:
- Open with encoding `utf-8-sig` (strips the Notepad BOM).
- Strip whitespace including `\r` from each line.
- Skip blank lines and lines starting with `#`.
- Strip inline comments: `12345  # paralinguistic candidate` → `12345`.
- Tolerate `id|category|subcategory` format by splitting on `|` and taking the first field.
- Convert remaining tokens to int.
- Validate `0 <= id < N_LATENTS` by reading N_LATENTS via `py -3.12 pipeline_utils.py get-provenance` and parsing the JSON. Reject out-of-range IDs with line numbers and a clear error.
- On `ValueError` for non-integer tokens, report which line failed.
- Deduplicate with a warning (don't silently discard).
- Print to stderr: `"Parsed N valid feature IDs from output/features_to_interpret.txt"`.

Display: `"Will audit {N} features in batches of {batch_size}: {list}"`

### Step 2: Pre-Flight Checks

**Step 2.1: Check `results.json` exists for every requested ID.**

For each parsed ID, Glob for `output/interpretations/feature<ID>/results.json`. Collect any missing IDs into `missing_results`.

**If ANY are missing → STOP with full list:**
```
ERROR: cannot audit features without an interpretation. Missing results.json for:
  - 12345
  - 67890
Run /interpret-features first.
```
Do NOT process anything. Exit non-zero.

**Step 2.2: Check existing `audit_report.md` for skip/force handling.**

For each remaining ID, Glob for `output/interpretations/feature<ID>/audit_report.md`.

- **Without `--force`:** features with `audit_report.md` → skip with `"feature N: already has audit_report.md (skipping; use --force to re-audit)"`.
- **With `--force`:** features with `audit_report.md` → queue for processing AND mark as `needing_rescue`.

**Step 2.3: Empty-list early exits.**

- If the parsed list has 0 valid IDs, print `"Nothing to audit: 0 features in features_to_interpret.txt (or all comments/blank)"` and exit cleanly with status 0.
- If after filtering the to-process list is empty, print `"Nothing to audit: <X> features queued, <Y> skipped (already done), <Z> missing results.json"` and exit cleanly with status 0.

**Step 2.4: Cost preflight summary.**

```
Auditing N features in batches of 5.
Of these, F are --force re-audits (audit_report.md will be rescued to .pre_force.* before running).
--skip-repro: [true/false]
```

**Step 2.5: Modal preflight (conditional).**

Run `py -3.12 pipeline_utils.py preflight-modal` ONLY if `--skip-repro` is NOT set AND `--skip-preflight` is NOT set. If preflight fails, exit non-zero with the error message.

### Step 3: Process Features in Parallel Batches

Default `batch_size = 5`, matching `/interpret-features` and Modal's parallel capacity. Split the to-process list into chunks of `batch_size`.

**If `--max-batches=N` was set**, the loop runs only the first `N` batches; features in batches `N+1` and beyond are NOT processed and are collected into a `not_yet_processed` list for the final summary. If `N >= num_batches`, the flag is a no-op and all batches run.

**For each batch (from i=1 to min(num_batches, max_batches if set else num_batches)):**

Display: `"--- Batch {i}/{num_batches}: Auditing features {list} ---"` (when capped, also display `"(cap: --max-batches={N})"` on the first batch.)

#### 3a. Force-Rescue (for features marked `needing_rescue`)

For each feature marked `needing_rescue`:
1. Get a Windows-safe timestamp: `py -3.12 pipeline_utils.py timestamp`
2. Rename `output/interpretations/feature{ID}/audit_report.md` → `output/interpretations/feature{ID}/audit_report.md.pre_force.<timestamp>` using a Python one-liner:
   ```bash
   py -3.12 -c "import pathlib; src = pathlib.Path('output/interpretations/feature{ID}/audit_report.md'); dst = src.parent / f'audit_report.md.pre_force.<timestamp>'; src.rename(dst); print(f'rescued -> {dst.name}')"
   ```
3. Print: `"feature {ID}: rescued existing audit_report.md → audit_report.md.pre_force.<timestamp>"`

Prior `audit_report.md.pre_force.*` siblings are preserved — never deleted.

#### 3b. Spawn Auditor Sub-Agents (parallel within batch)

**CRITICAL: launch ALL Auditor sub-agents for the batch in a SINGLE message containing multiple Agent tool calls — this is what makes them run in parallel. Separate messages would run them sequentially.**

Each sub-agent gets this EXACT prompt template. The `--skip-repro` branch is a literal text substitution based on whether the flag is active:

**If `--skip-repro` is active:**
```
Perform the full 7-layer audit for SAE feature {ID}.

1. Read the full instructions from: .claude/commands/audit.md
2. Follow ALL steps in that file for feature {ID} — wherever the command file uses a feature-index placeholder, substitute {ID}
3. SKIP Layer 7 entirely. Record REPRO_STATUS = SKIPPED and proceed to Step A.8.
4. You are the Auditor. Read-only except for writing audit_report.md.
5. ALWAYS write output/interpretations/feature{ID}/audit_report.md, even if the Pre-Flight Gate fires. The report header MUST be exactly `**Date:** YYYY-MM-DD | **Overall:** X` where X is one of: PASS, WARN, FAIL, MISSING_RESULTS, BROKEN_FILE, IO_ERROR, LEGACY_FORMAT, INCONSISTENT_FORMAT. Do not omit the file on any termination path.

IMPORTANT: Working directory is the project root. Use paths as specified in the command file.
Your final return message should be a brief summary of the audit outcome (Overall status only), not the full report content.
```

**Otherwise (full Layer 7 run):**
```
Perform the full 7-layer audit for SAE feature {ID}.

1. Read the full instructions from: .claude/commands/audit.md
2. Follow ALL steps in that file for feature {ID} — wherever the command file uses a feature-index placeholder, substitute {ID}
3. Run the full Layer 7 reproducibility check.
4. You are the Auditor. Read-only except for writing audit_report.md.
5. ALWAYS write output/interpretations/feature{ID}/audit_report.md, even if the Pre-Flight Gate fires. The report header MUST be exactly `**Date:** YYYY-MM-DD | **Overall:** X` where X is one of: PASS, WARN, FAIL, MISSING_RESULTS, BROKEN_FILE, IO_ERROR, LEGACY_FORMAT, INCONSISTENT_FORMAT. Do not omit the file on any termination path.

IMPORTANT: Working directory is the project root. Use paths as specified in the command file.
Your final return message should be a brief summary of the audit outcome (Overall status only), not the full report content.
```

**Do NOT modify, extend, or add context to these prompt templates.**

Wait for ALL Auditor sub-agents in the batch to complete.

#### 3c. Checkpoint (per feature, after sub-agent returns)

For each feature in the batch:
1. **Glob** for `output/interpretations/feature{ID}/audit_report.md`.
2. If missing → mark `FAILED_NO_REPORT`.
3. If present → **Read the first 10 lines** and match the header line `**Overall:** <TOKEN>` where `<TOKEN>` is one of the 8 tokens listed in the sub-agent prompt above. Record `<TOKEN>` as the feature's Overall status.
4. If no matching header in the first 10 lines → mark `UNPARSEABLE_HEADER`.

Display per feature: `"[{counter}/{total}] feature {ID} {OVERALL}"`

Continue to next batch — UNLESS the batch just completed was the `--max-batches` cap, in which case exit the loop cleanly and proceed to Step 4. The `not_yet_processed` list for the summary is all features in batches `i+1` through `num_batches`.

### Step 4: Final Summary

After all batches complete (or after `--max-batches` capped the loop), display:

```
=== AUDIT COMPLETE ===
Processed: {success_count} | Skipped: {skip_count} | Failed: {fail_count}
Batches run: {batches_run} of {num_batches}  (when --max-batches was set; omit this line otherwise)

| Feature | Overall          |
|---------|------------------|
| 11328   | PASS             |
| 208     | WARN             |

Skipped (already audited, use --force to redo): 82498
Failed:
  - 12345 (FAILED_NO_REPORT: audit_report.md not written)
  - 77777 (UNPARSEABLE_HEADER: header format unrecognized)

Remaining (failed or unprocessed due to --max-batches; copy-paste back into output/features_to_interpret.txt if needed):
  12345
  77777
  # unprocessed (max-batches cap):
  21562
  6162

=== BREAKDOWN ===
<counts per classification — render only non-zero buckets>
```

The BREAKDOWN line shows one bucket per encountered token: e.g., `PASS: 3 | WARN: 1 | LEGACY_FORMAT: 2`. Do not emit zero-count buckets. When `--max-batches` capped the loop, split the Remaining section into two sub-groups (failed + unprocessed), matching `/interpret-features`'s format.

### Implicit Checkpointing (resume-on-restart)

The skip-if-`audit_report.md` semantics provide implicit resume. Re-running `/audit` after a crash or 529 overload continues from where the previous run died (unless `--force` is set). No manual recovery.

### Concurrent-Runs Warning

**Do NOT** run `/audit` in two Claude Code sessions simultaneously against overlapping feature IDs — the two orchestrators will race on `audit_report.md` writes. Same warning as `/interpret-features`.

---

# Per-Feature Audit (Sub-Agent Path)

From this point down, instructions apply to a single feature identified by `$ARGUMENTS` (for direct CLI invocation) or by the feature ID given in your sub-agent prompt (for batch invocation). The Batch Orchestration section above does not apply in this mode — you were directed here by the SUB-AGENT BYPASS instruction at the top of the file.

---

## CRITICAL RESTRICTIONS

**DO NOT read, access, or reference these files under any circumstances:**
- Any `.ipynb` notebook files
- Anything in `src/` (the Modal backend code is not your concern — except for running `batch_test` in Layer 7)
- Any reference documents outside this command
- Any other feature's interpretation outputs — cross-feature isolation applies. (Note: `feature data/Feature_output.csv` is a legacy file from the old pipeline; do not read it either.)

**All data must come from the files specified in Step A.0.** Do not explore the codebase or look for alternative data sources.

**DO NOT modify any interpretation artifacts.** The Auditor is read-only. The only file you CREATE is `audit_report.md`.

## TOOL RULES (prevents permission prompts)

- **Read files** with the Read tool. NEVER use `cat`, `head`, `tail`, `type`, or any Bash command to read files.
- **Write/edit files** with the Write or Edit tool. NEVER use `echo`, Bash redirection, or heredocs.
- **Search files** with the Grep or Glob tool. NEVER use `grep`, `find`, `ls`, or `dir` via Bash.
- **Do all computation** (JSON parsing, metric calculation, field comparison) in your own reasoning. NEVER write or run Python scripts.
- **Only these Bash commands are approved:**
  - `py -3.12 run_modal_utf8.py batch_test --args` (Layer 7 repro only)
  - `py -3.12 pipeline_utils.py timestamp`
- Any other Bash command will trigger a permission prompt and block the pipeline.

## PATH CONVENTIONS

Working directory is the **project root** (`Claude Code Folder`).

- **Feature data:** `"feature data/feature_$ARGUMENTS.json"`
- **Output dir:** `output/interpretations/feature$ARGUMENTS/`
- **pipeline_utils.py:** `py -3.12 pipeline_utils.py <command>`
- **Modal commands:** `py -3.12 run_modal_utf8.py <entrypoint> --args`

---

## ACCURACY PROTOCOL

The Auditor copies values from source files into the audit report. Same transcription risk as the Reporter.

1. **Numerical precision:** Copy all decimal places exactly. If the source says `6.8438`, write `6.8438`.
2. **Copy, don't paraphrase** — When reporting field values, copy exactly from the file.
3. **Fresh reads only** — Before writing any value in the report, re-read it from the source (not from memory).

---

## Step A.0: Load Artifacts

Read ALL of these files for the target feature:

1. `output/interpretations/feature$ARGUMENTS/results.json`
2. `output/interpretations/feature$ARGUMENTS/audit.jsonl`
3. `output/interpretations/feature$ARGUMENTS/report.md`
4. `feature data/feature_$ARGUMENTS.json`
5. `output/interpretations/feature$ARGUMENTS/phase1_results.json`
6. `output/interpretations/feature$ARGUMENTS/batch_test_$ARGUMENTS.json`

Also check whether these exist (use Glob):
7. `challenge_verdict.json` — if present, this is a Reporter-generated output (full provenance checks available)
8. `challenge_c1_counterexamples.json` through `challenge_c4_surprising.json`

**If `results.json` does not exist:** Write a pre-flight stub to `output/interpretations/feature$ARGUMENTS/audit_report.md` using the Step A.9 template format, then STOP. The stub must contain exactly:

```markdown
# Audit Report: Feature $ARGUMENTS

**Date:** <today, YYYY-MM-DD> | **Overall:** MISSING_RESULTS

## Pre-Flight Gate

No interpretation found for feature $ARGUMENTS — `results.json` does not exist. Run `/interpret-features $ARGUMENTS` first. No audit layers were run.
```

Use the Write tool. This stub guarantees the batch orchestrator's checkpoint Glob always finds a file with a machine-parseable `**Overall:** MISSING_RESULTS` header line. After writing, print `"=== AUDIT: Feature $ARGUMENTS ===\nOverall: MISSING_RESULTS"` to console and exit.

---

## Step A.0.5: Pre-Flight Gate (BEFORE Layer 1) — File Classification

**Critical:** This gate runs BEFORE any layer. It catches malformed/legacy files and exits with a clear classification rather than crashing inside Layer 3 on the old action vocabulary.

All checks use **defensive type guards** to avoid crashing on malformed files. Each check uses Python-style logic (in your reasoning) — do NOT actually run Python.

**Every exit from this gate MUST write `output/interpretations/feature$ARGUMENTS/audit_report.md` before exiting** — see the "Pre-flight stub format" block below. This is load-bearing: the batch orchestrator's checkpoint Glob requires a file with a parseable `**Overall:** <TOKEN>` header for every pre-flight termination.

1. **Check `results.json` exists.** If missing, write the pre-flight stub with `**Overall:** MISSING_RESULTS` and body "No `results.json` present, no audit possible." then exit. (Note: 2 of the ~746 legacy folders lack results.json entirely; this is the expected result for those. Step A.0 may have already written this stub; re-writing with the same content is safe.)

2. **Check `results.json` is parseable.** If JSON parse fails, write the pre-flight stub with `**Overall:** BROKEN_FILE` and body "Invalid JSON: &lt;error&gt;." then exit.

3. **Check file-system permission.** If `PermissionError`, write the pre-flight stub with `**Overall:** IO_ERROR` and body "Cannot read results.json: &lt;reason&gt;." then exit.

4. **Check `interpretation_phase` is a dict.** If `data.get("interpretation_phase")` is missing OR `None` OR not a dict → write the pre-flight stub with `**Overall:** BROKEN_FILE` and body "`interpretation_phase` missing or wrong type." then exit.

5. **Check `test_results` is a non-empty list.** Get `test_results = interpretation_phase.get("test_results")`. If missing OR `None` OR not a list OR length 0 → write the pre-flight stub with `**Overall:** BROKEN_FILE` and body "Empty or invalid `test_results`." then exit.

6. **Check each entry in `test_results` is a dict.** If any entry is not a dict → write the pre-flight stub with `**Overall:** BROKEN_FILE` and body "Non-dict `test_result` entry." then exit.

7. **Now check for `token_activations`:**
   - If **NO entry** has a `token_activations` key → write the pre-flight stub with `**Overall:** LEGACY_FORMAT` and body "Pre-four-agent format detected. Re-run `/interpret-features --force` to regenerate." then exit. Do not run any layer.
   - If **SOME but not ALL** entries have `token_activations` → write the pre-flight stub with `**Overall:** INCONSISTENT_FORMAT` and body "Likely failed re-interpretation, manual inspection required." then exit. Do not run any layer.

8. **Otherwise** → all entries have `token_activations`. Proceed to Layer 1.

**Pre-flight stub format** (use the Write tool; substitute the actual status for `<STATUS>` and the actual body text for `<BODY>`):

```markdown
# Audit Report: Feature $ARGUMENTS

**Date:** <today, YYYY-MM-DD> | **Overall:** <STATUS>

## Pre-Flight Gate

<BODY>
```

After writing the stub, print `"=== AUDIT: Feature $ARGUMENTS ===\nOverall: <STATUS>"` to console and exit. Do not write any of the Layer 1–7 sections.

The gate ordering matters: Layer 3 (Process Validation) uses the new action vocabulary (`observation`, `abduction`, etc.) and would FAIL on legacy `audit.jsonl` files that use the old vocabulary (`read_file`, `hypothesis_generation`, etc.). The pre-flight gate prevents that confusion by classifying legacy files cleanly before any layer runs.

---

## Step A.1: Layer 1 — Schema Validation

Check all required fields exist in `results.json` with correct types. **GPT-2-adapted schema** (NOT the original Llama Layer 1):

| Field | Required Type |
|-------|--------------|
| `feature_idx` | number |
| `status` | string `"complete"` |
| `label` | non-empty string |
| `category` | string |
| `description` | non-empty string |
| `confidence` | number 0.0–1.0 |
| `verdict` | one of: CONFIRMED, REFINED, REFUTED, UNCERTAIN |
| `necessary_conditions` | array |
| `boundary_conditions` | array |
| `does_not_detect` | array |
| `corpus_stats` | object with: `activation_rate`, `mean_when_active` (NOT `mean_activation`), `max_activation`, `std_when_active`, `total_activations`, AND nested `sampling.tokens_scanned` |
| `top_tokens` | array |
| `top_activations` | object/array (GPT-2 has nested structure) |
| `ngram_analysis` | object |
| `provenance` | object — see below |
| `interpretation_phase` | object |
| `interpretation_phase.hypotheses` | array, exactly 3 items |
| `interpretation_phase.test_results` | non-empty array |
| `interpretation_phase.winner` | object with `id`, `accuracy`, `justification` |
| `challenge_phase` | object |
| `challenge_phase.counterexamples` | non-empty array |
| `challenge_phase.verdict` | string |
| `challenge_phase.verdict_justification` | non-empty string |
| `key_examples` | non-empty array |
| `executive_summary` | non-empty string, ≥50 characters |
| `linguistic_function` | non-empty string |
| `potential_applications` | non-empty string |

**Provenance handling:** The `provenance` block can take two shapes:
1. **Success:** sub-fields `model`, `sae_checkpoint`, `layer`, `window_size`, `n_latents`, `k_active`, `expansion`, `pipeline_version`, `run_timestamp`, `corpus_data_source` — all required.
2. **Fallback:** `provenance.status == "unavailable"` — emit **WARN (not FAIL)** and skip the per-field check. The Reporter writes this fallback when `pipeline_utils.py get-provenance` fails (e.g., src/config.py is missing). The pipeline still completed; treat as a soft failure.

Also check: `report.md` exists and is non-empty.

**Record:** SCHEMA_STATUS = PASS, WARN (provenance unavailable), or FAIL (with list of missing/invalid fields).

---

## Step A.2: Layer 2 — Content Validation

Substantive checks (values, not just field existence):

- **Hypotheses:** All 3 have distinct descriptions (no two are near-duplicates)
- **Test results:** Entries have actual activation values (at least some non-zero)
- **Token activations:** `token_activations` arrays are non-empty in ALL test entries — both `interpretation_phase.test_results` and all `challenge_phase` subsections (`counterexamples`, `alternative_tests`, `minimal_pairs.grid`, `surprising_predictions`)
- **Token activations cross-check:** For each test entry, `max(token_activations[*].activation)` should approximately equal `activation` (the top-level max). Use `math.isclose(abs_tol=5e-5)` to absorb the 4-decimal rounding asymmetry between the unrounded `max_activation` source and the rounded per-token values. If the cross-check fails for >1 entry, FAIL.
- **Winner:** Has `justification` text (not empty)
- **Counterexamples:** Have activation values (numbers, not placeholders)
- **Key examples:** Each has a `meaning` annotation (non-empty string)
- **Executive summary:** ≥50 characters and describes the feature (not a generic placeholder)

**Record:** CONTENT_STATUS = PASS or FAIL (with specific issues).

---

## Step A.3: Layer 3 — Process Validation

Read `audit.jsonl`. Parse each line as JSON.

**Identify phases:**
- Phase 1 entries: `phase` is `"theorist"`, `"tester"`, or `"phase1"` (all three are valid labels)
- Phase 2 entries: `phase` is `"phase2"`

**Check Peircean stage ordering in Phase 1:**
Required stages (in this order): `observation` → `abduction` → `deduction` → `testing` → `evaluation`

For each required stage, verify:
1. At least one Phase 1 entry has this `action` value
2. The first occurrence of each stage comes before the first occurrence of the next stage

**Check Phase 2:**
- At least one entry with `action` = `"challenge"`
- At least one entry with `action` = `"evaluation"`

**Check phase ordering:**
- All Phase 1 entries appear before all Phase 2 entries (by position in file)

**Check timestamps:**
- Every entry has a `timestamp` field. Accept either ISO 8601 BASIC format (`YYYYMMDDTHHMMSSZ` — what `pipeline_utils.py timestamp` emits) OR ISO 8601 EXTENDED format (`YYYY-MM-DDTHH:MM:SSZ` — legacy from Llama or hand-typed entries).
- No placeholder values like `<ISO 8601>` or `<ISO 8601 BASIC>`

**Record:** PROCESS_STATUS = PASS or FAIL (with missing stages or ordering violations).

---

## Step A.4: Layer 4 — Provenance Validation

**Pre-Reporter detection:** If `challenge_verdict.json` does NOT exist, report: "Pre-Reporter output detected — judgment field provenance checks skipped." Run only corpus stats and count checks.

### Corpus Stats (always run)

**Re-read `feature data/feature_$ARGUMENTS.json` now.** Compare against `results.json`. **GPT-2 schema preservation: source field names match target field names exactly — no rename. Tokens_scanned is NESTED under `sampling`, not flat.**

| Source Field | Target Field | Source Value | Target Value | Match? |
|-------------|-------------|-------------|-------------|--------|
| `stats.activation_rate` | `corpus_stats.activation_rate` | ___ | ___ | |
| `stats.mean_when_active` | `corpus_stats.mean_when_active` | ___ | ___ | |
| `stats.max_activation` | `corpus_stats.max_activation` | ___ | ___ | |
| `stats.std_when_active` | `corpus_stats.std_when_active` | ___ | ___ | |
| `stats.total_activations` | `corpus_stats.total_activations` | ___ | ___ | |
| `stats.sampling.tokens_scanned` | `corpus_stats.sampling.tokens_scanned` | ___ | ___ | |

**Note:** Unlike Llama (which renames `mean_when_active` → `mean_activation`), the GPT-2 port preserves the source name to match `classify_utils.py` and the existing legacy interpretations.

### Judgment Fields (only when `challenge_verdict.json` exists)

**Re-read `challenge_verdict.json` now.** Compare against `results.json`:

| Field (source) | Field (target) | Match? |
|----------------|----------------|--------|
| `verdict` | `verdict` | |
| `verdict_justification` | `challenge_phase.verdict_justification` | |
| `label` | `label` | |
| `category` | `category` | |
| `confidence` | `confidence` | |
| `description` | `description` | |
| `executive_summary` | `executive_summary` | |

All must be exact matches (string or numeric).

### Count Checks (always run)

| Source | Target | Expected | Actual | Match? |
|--------|--------|----------|--------|--------|
| `phase1_results.json` → `test_results` length | `interpretation_phase.test_results` length | ___ | ___ | |
| `challenge_c1_counterexamples.json` length | `challenge_phase.counterexamples` length | ___ | ___ | |
| `challenge_c2_alternatives.json` length | `challenge_phase.alternative_tests` length | ___ | ___ | |
| `challenge_c3_minimal_pairs.json` → `grid` length | `challenge_phase.minimal_pairs.grid` length | ___ | ___ | |
| `challenge_c4_surprising.json` length | `challenge_phase.surprising_predictions` length | ___ | ___ | |
| `phase1_results.json` → `winner.accuracy` | `interpretation_phase.winner.accuracy` | ___ | ___ | |

If any checkpoint file is missing, skip that count check and note: "Checkpoint file missing: {filename}."

**Record:** PROVENANCE_STATUS = PASS or FAIL (with specific mismatches).

---

## Step A.5: Layer 5 — Logic Validation

### Derive expected/actual for Phase 1 tests

Read the winner hypothesis ID: `interpretation_phase.winner.id` (numeric, e.g., `1`). Construct tag: `"H" + str(id)` → e.g., `"H1"`.

For each test in `interpretation_phase.test_results`:

**Actual:** `actual = (activation > 0)`

**Expected:**
- If the test has an `expected` field: use it directly
- If `supports` == `"all"`: expected = true
- If `supports` == `"none"`: expected = false
- If the winner tag (e.g., `"H1"`) appears in `supports`: expected = true
- If the winner tag does NOT appear in `supports`: expected = false
- If expected cannot be determined: skip, note in report

### Calculate metrics

```
TP = expected true AND actual true
FP = expected false AND actual true
TN = expected false AND actual false
FN = expected true AND actual false

accuracy = (TP + TN) / total
precision = TP / (TP + FP) if (TP + FP) > 0 else N/A
recall = TP / (TP + FN) if (TP + FN) > 0 else N/A
```

### Flag mismatches (WARN, not FAIL)

| Condition | Flag |
|-----------|------|
| CONFIRMED/REFINED verdict with accuracy < 70% | WARN: Low accuracy for supported verdict |
| REFUTED verdict with accuracy > 65% | WARN: High accuracy for refuted verdict |
| Confidence > 90% with accuracy < 80% | WARN: High confidence with moderate accuracy |
| Confidence > 90% with FP > 2 | WARN: High confidence with multiple false positives |

**Record:** LOGIC_STATUS = PASS (no flags) or WARN (flags present). Logic never FAILs.

---

## Step A.6: Layer 6 — Causal Validation

GPT-2 uses causal (autoregressive) attention with a **64-token context window**. At position N, the model sees tokens 0 to N-1 only. A feature firing on token X can only depend on tokens to the LEFT of X.

### Check interpretive text fields

Scan these fields for right-context violations:
- `label`, `description`
- Each entry in `necessary_conditions[]`, `boundary_conditions[]`, `does_not_detect[]`
- Each `key_examples[].meaning`
- `executive_summary`, `linguistic_function`
- `challenge_phase.verdict_justification`

### Violations (claims right-context dependency)

- "fires when followed by X"
- "detects tokens before [punctuation/word]"
- "anticipates/predicts future tokens"
- "depends on what comes after"
- "fires when next token is"
- "preceding X" (confused with "preceded by X")
- "in preparation for"
- "to signal upcoming"
- "before a [comma/period/space]"
- "when the next word is"

### Valid patterns (left-context)

- "fires when preceded by X"
- "fires on [token] following [context]"
- "left context contains X"
- "after seeing X, fires on Y"
- "requires prior token to be X"
- "activates after X appears"

**GPT-2-specific note:** Claims about distant left context (e.g., "fires 50 tokens after X") are valid only if the test texts have been confirmed to exceed 50 tokens. With a 64-token window, very-long-range dependencies are usually unrealistic.

### Assess

- **PASS:** All text correctly describes left-context patterns
- **WARN:** Ambiguous phrasing that could be interpreted either way
- **FAIL:** Clear violation claiming right-context dependency

**Record:** CAUSAL_STATUS = PASS, WARN, or FAIL (with specific violations quoted).

---

## Step A.7: Layer 7 — Reproducibility Check

**If your invoking prompt contains an explicit instruction to skip Layer 7 (e.g., "SKIP Layer 7 entirely") OR `$ARGUMENTS` contains `--skip-repro`:** Record REPRO_STATUS = SKIPPED. Proceed to Step A.8. (This dual-branch check is load-bearing: direct CLI invocation uses the `$ARGUMENTS` path, while sub-agent invocation from the batch orchestrator uses the prompt-text path.)

**Cost note:** GPT-2 reproducibility check is ~1 minute per feature (warm container + 5 batch_test calls). For batch audits, use `--skip-repro` and spot-check a sample with full repro.

### Select 5 tests

From `interpretation_phase.test_results`:
- 2 tests with the highest activation values
- 3 tests that did NOT activate (activation = 0 or very low)

If fewer than 5 tests available, use all.

### Re-run tests

```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --output-dir output/interpretations/feature$ARGUMENTS/repro --fresh --texts "text1|text2|text3|text4|text5"
```

**Shell escaping:** If any test text contains `$`, backtick, `!`, or `\`:
1. First attempt: run as-is
2. If returned text differs from input (shell corruption): escape special characters and re-run

### Compare results

Read the output from the `repro/` directory. For each test:

| Check | Pass criteria |
|-------|--------------|
| Activation status | Both fired or both didn't |
| Value | Within ±0.03 OR ±15% (whichever larger) |
| Token | Same `active_token` (if fired) |

### Score

| Matches | Status |
|---------|--------|
| 5/5 | PASS |
| 3-4/5 | WARN |
| <3/5 | FAIL |

**Record:** REPRO_STATUS = PASS, WARN, or FAIL.

---

## Step A.8: Determine Overall Status

| Schema | Content | Process | Provenance | Logic | Causal | Repro | Overall |
|--------|---------|---------|------------|-------|--------|-------|---------|
| FAIL | — | — | — | — | — | — | **FAIL** |
| PASS | FAIL | — | — | — | — | — | **FAIL** |
| PASS | PASS | FAIL | — | — | — | — | **FAIL** |
| PASS | PASS | PASS | FAIL | — | — | — | **FAIL** |
| PASS | PASS | PASS | PASS | — | FAIL | — | **FAIL** |
| PASS | PASS | PASS | PASS | — | — | FAIL | **FAIL** |
| WARN (provenance unavailable) | PASS | PASS | PASS | PASS | PASS | PASS/SKIP | **WARN** |
| PASS | PASS | PASS | PASS | WARN | any | any | **WARN** |
| PASS | PASS | PASS | PASS | any | WARN | any | **WARN** |
| PASS | PASS | PASS | PASS | any | any | WARN | **WARN** |
| PASS | PASS | PASS | PASS | PASS | PASS | PASS/SKIP | **PASS** |

Rule: Any FAIL in a hard layer (Schema, Content, Process, Provenance, Causal, Repro) → overall FAIL. Any WARN with no FAILs → overall WARN. All PASS (or SKIP) → overall PASS.

---

## Step A.9: Write Audit Report

**Valid `Overall` tokens (choose exactly one):** PASS, WARN, FAIL, MISSING_RESULTS, BROKEN_FILE, IO_ERROR, LEGACY_FORMAT, INCONSISTENT_FORMAT. This is the single canonical enumeration — the orchestrator (batch mode) reads the `**Overall:** <TOKEN>` line from this header and classifies against these tokens. Do not invent new tokens or emit the bracketed literal `[STATUS]` placeholder below into the final file — substitute the actual status.

Write to `output/interpretations/feature$ARGUMENTS/audit_report.md`:

```markdown
# Audit Report: Feature $ARGUMENTS

**Date:** YYYY-MM-DD | **Overall:** [STATUS]

## Pre-Flight Gate

[If a special status was emitted at Step A.0 or A.0.5, this file was already written at that step with `**Overall:** <STATUS>` and a matching Pre-Flight Gate body; Layers 1–7 were not run and their sections are not written.]

[If gate passed:] All defensive type guards passed; results.json is well-formed and uses the four-agent schema.

## Layer 1: Schema — [PASS/FAIL/WARN]

| Field | Status | Notes |
|-------|--------|-------|
| feature_idx | ✓/✗ | [value or MISSING] |
| status | ✓/✗ | [value or MISSING] |
... (all required fields)

Missing: [list or "none"]
Provenance: [PASS / WARN unavailable]

## Layer 2: Content — [PASS/FAIL]

[Issues found or "All content checks passed."]

## Layer 3: Process — [PASS/FAIL]

| Stage | Action | Phase | Status |
|-------|--------|-------|--------|
| observation | observation | theorist/phase1 | ✓/✗ |
| abduction | abduction | theorist/phase1 | ✓/✗ |
| deduction | deduction | tester/phase1 | ✓/✗ |
| testing | testing | tester/phase1 | ✓/✗ |
| evaluation | evaluation | tester/phase1 | ✓/✗ |
| challenge | challenge | phase2 | ✓/✗ |
| evaluation | evaluation | phase2 | ✓/✗ |

Ordering: [correct or violations]
Timestamps: [all valid or placeholders found]

## Layer 4: Provenance — [PASS/FAIL]

[If pre-Reporter: "Pre-Reporter output — judgment field provenance skipped."]

### Corpus Stats
| Source Field | Target Field | Source | Target | Match? |
|-------------|-------------|--------|--------|--------|
... (6 rows)

### Judgment Fields
| Field | Source | Target | Match? |
|-------|--------|--------|--------|
... (7 rows, or "skipped" if pre-Reporter)

### Count Checks
| Section | Source Count | Target Count | Match? |
|---------|-------------|--------------|--------|
... (6 rows)

## Layer 5: Logic — [PASS/WARN]

| Metric | Value |
|--------|-------|
| Total tests | N |
| TP | N |
| FP | N |
| TN | N |
| FN | N |
| Accuracy | X% |
| Precision | X% |
| Recall | X% |

**Verdict:** [from results.json] | **Confidence:** [from results.json]
[Flags or "Verdict and confidence align with test evidence."]

## Layer 6: Causal — [PASS/WARN/FAIL]

[If violations found:]
| Field | Quote | Issue |
|-------|-------|-------|
...

[If none:] All interpretive text correctly describes left-context patterns.

## Layer 7: Repro — [PASS/WARN/FAIL/SKIPPED]

[If SKIPPED:] Reproducibility check skipped (--skip-repro flag).

[If run:]
| # | Text (first 50 chars) | Original Act. | Repro Act. | Token Match | Status |
|---|----------------------|---------------|------------|-------------|--------|
...

**Matched:** N/5

## Summary

| Layer | Status |
|-------|--------|
| 1. Schema | [status] |
| 2. Content | [status] |
| 3. Process | [status] |
| 4. Provenance | [status] |
| 5. Logic | [status] |
| 6. Causal | [status] |
| 7. Repro | [status] |
| **Overall** | **[status]** |

[If PASS:] "Interpretation verified."
[If WARN:] "Interpretation verified with warnings: [summary]"
[If FAIL:] "Audit failed: [summary of failures]"
```

---

## Final Output

Display to console:

```
=== AUDIT: Feature $ARGUMENTS ===
Overall: [PASS/WARN/FAIL/MISSING_RESULTS/BROKEN_FILE/LEGACY_FORMAT/INCONSISTENT_FORMAT]

[If a special pre-flight status fired, just print the status and exit.]

[Otherwise:]
Schema:     [PASS/FAIL/WARN]
Content:    [PASS/FAIL]
Process:    [PASS/FAIL]
Provenance: [PASS/FAIL]
Logic:      [PASS/WARN]
Causal:     [PASS/WARN/FAIL]
Repro:      [PASS/WARN/FAIL/SKIPPED]

Report: output/interpretations/feature$ARGUMENTS/audit_report.md
```

If auditing multiple features, display the summary for each, then a final aggregate:

```
=== AUDIT SUMMARY ===
Features audited: N
PASS: X | WARN: Y | FAIL: Z | LEGACY: L | BROKEN: B
```

---

## Begin

1. Parse arguments to extract feature ID(s) and check for `--skip-repro` flag
2. For each feature, start with Step A.0
