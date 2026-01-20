# Batch Verify Feature Interpretations

Verify **N** total feature interpretations from Feature_output.csv, running **batch_size** in parallel at a time.

**Usage:** `/verify-batch <batch_size> [total] [sort_by]`

**Arguments:**
- `batch_size` (required) - How many verifications to run in parallel per batch
- `total` (optional) - Total features to verify, defaults to batch_size
- `sort_by` (optional) - Column to sort by: `rank_control` (default) or `rank_nocontrol`

**Examples:**
- `/verify-batch 3 10` - Verify 10 features, 3 at a time, sorted by rank_control
- `/verify-batch 5` - Verify 5 features, 5 at a time (single batch)
- `/verify-batch 3 10 rank_nocontrol` - Verify 10 features sorted by rank_nocontrol

## Instructions

### Step 1: Parse Arguments

Parse `$ARGUMENTS` to extract:
- `batch_size` = first number (how many to run in parallel)
- `total` = second number (total to verify), defaults to batch_size if not specified
- `sort_by` = third argument if present, must be `rank_control` or `rank_nocontrol`, defaults to `rank_control`

Display: "Will verify {total} features, {batch_size} at a time, sorted by {sort_by}"

### Step 2: Create CSV Backup

Run:
```bash
py -3.12 batch_utils.py backup-csv
```

Display: "Created backup: Feature_output.csv.backup"

### Step 3: Find Candidate Features

Run:
```bash
py -3.12 batch_utils.py find-unverified --sort-by {sort_by} --limit {total}
```

Parse the JSON output to extract the `features` array. This gives you the list of feature IDs with interpretations but no verification status, sorted by the specified column.

Display: "Candidate features: {features list}"

If no features are returned (features array is empty), display "All interpreted features have already been verified" and exit.

### Step 4: Verify Prerequisites

Run:
```bash
py -3.12 batch_utils.py check-existing --features {comma-separated feature IDs}
```

Parse the JSON output:
- **If any features are in `missing_output`:** These don't have results.json. Remove them from the processing list.
  Display: "Skipping N features without results.json: X, Y, Z"

The features to process are those in `has_output`.

Display: "Features to verify: A, B, C, ..."

If no features remain to process, display "No features ready for verification" and exit.

### Step 5: Process in Batches (LOOP)

Split the remaining feature list into chunks of size `batch_size`.

**For each batch:**

1. Display: "--- Batch {i}/{num_batches}: Verifying features {list} ---"

2. For EACH feature in this batch, spawn a sub-agent using the Task tool:
   - `subagent_type: "general-purpose"`
   - Use this prompt template (replace {ID} with the feature index):
     ```
     Verify SAE feature interpretation for feature {ID}.

     1. Read the full instructions from: .claude/commands/verify.md
     2. Follow ALL steps in that file for feature {ID} (replace $ARGUMENTS with {ID})
     3. Perform the full verification including reproducibility check.

     IMPORTANT: Use relative paths only (e.g., "output/interpretations/feature{ID}/"). Absolute paths will trigger permission prompts.
     ```
   - **CRITICAL:** Spawn ALL agents for this batch in a SINGLE message (parallel execution)

3. Wait for all agents in this batch to complete

4. **Collect results for this batch:**
   For each feature in this batch, read the verification status from the agent output or check the verification.md file:
   ```
   output/interpretations/feature{ID}/verification.md
   ```

   Look for the "Overall:" line to determine PASS/WARN/FAIL status.

   Note: The CSV is already updated by each verification agent via `batch_utils.py update-verify`.

5. Display: "Batch {i} complete. Verified {n} features."

6. Continue to next batch

### Step 6: Final Summary

After all batches complete:

1. **Record batch summary:**
   ```bash
   py -3.12 batch_utils.py batch-summary --features {comma-separated list of all verified features}
   ```
   This records the batch to `feature data/batch_history.csv` and outputs detailed JSON with reasons for any WARN/FAIL.

2. **Display summary table** (use the JSON output from batch-summary):
   ```
   === BATCH VERIFICATION COMPLETE ===
   Total features verified: {count}
   Batches completed: {num_batches}

   | Feature | Status | Logic | Causal | Repro | Reason |
   |---------|--------|-------|--------|-------|--------|
   | 7907    | PASS   | PASS  | PASS   | PASS  |        |
   | 11579   | WARN   | WARN  | PASS   | PASS  | High confidence (90%) with 70% accuracy |
   ...

   Feature_output.csv has been updated with verification results.
   Batch history saved to: feature data/batch_history.csv
   ```

The "Reason" column should only be populated for WARN/FAIL entries.

## Error Handling

- **Missing results.json:** Feature skipped (can't verify without interpretation)
- **Agent crash:** Partial batch results are still collected, failed features noted
- **Missing verification.md:** Mark as "FAILED" in summary

## Utility Script Reference

The `batch_utils.py` script provides these commands:

| Command | Purpose |
|---------|---------|
| `backup-csv` | Create Feature_output.csv.backup |
| `find-unverified --sort-by X --limit N` | Find interpreted features without verification |
| `check-existing --features 1,2,3` | Check for output files |
| `update-verify --feature N --status X` | Update verify_status in CSV (called by verify agents) |

All commands output JSON for easy parsing.

## Example Run

User runs: `/verify-batch 3 6`

```
Will verify 6 features, 3 at a time, sorted by rank_control

> py -3.12 batch_utils.py backup-csv
Created backup: Feature_output.csv.backup

> py -3.12 batch_utils.py find-unverified --sort-by rank_control --limit 6
Candidate features: 7907, 20276, 14292, 5005, 19056, 3080

> py -3.12 batch_utils.py check-existing --features 7907,20276,14292,5005,19056,3080
Features to verify: 7907, 20276, 14292, 5005, 19056, 3080

--- Batch 1/2: Verifying features 7907, 20276, 14292 ---
[3 agents run in parallel]
Batch 1 complete. Verified 3 features.

--- Batch 2/2: Verifying features 5005, 19056, 3080 ---
[3 agents run in parallel]
Batch 2 complete. Verified 3 features.

=== BATCH VERIFICATION COMPLETE ===
Total features verified: 6
Batches completed: 2

| Feature | Status | Logic | Causal | Repro |
|---------|--------|-------|--------|-------|
| 7907    | PASS   | PASS  | PASS   | PASS  |
| 20276   | PASS   | PASS  | PASS   | PASS  |
| 14292   | WARN   | WARN  | PASS   | PASS  |
| 5005    | PASS   | PASS  | PASS   | PASS  |
| 19056   | PASS   | PASS  | PASS   | PASS  |
| 3080    | PASS   | PASS  | PASS   | PASS  |

Feature_output.csv has been updated with verification results.
```
