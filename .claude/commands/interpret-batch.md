# Batch Interpret Features

Process **N** total features from Feature_output.csv, running **batch_size** in parallel at a time.

**Usage:** `/interpret-batch <batch_size> [total] [sort_by]`

**Arguments:**
- `batch_size` (required) - How many features to run in parallel per batch
- `total` (optional) - Total features to process, defaults to batch_size
- `sort_by` (optional) - Column to sort by: `rank_nocontrol` (default) or `rank_control`

**Examples:**
- `/interpret-batch 3 10` - Process 10 features, 3 at a time, sorted by rank_nocontrol
- `/interpret-batch 5` - Process 5 features, 5 at a time (single batch)
- `/interpret-batch 3 10 rank_control` - Process 10 features sorted by rank_control
- `/interpret-batch 5 5 rank_control` - Process 5 features sorted by rank_control

## Instructions

### Step 1: Parse Arguments

Parse `$ARGUMENTS` to extract:
- `batch_size` = first number (how many to run in parallel)
- `total` = second number (total to process), defaults to batch_size if not specified
- `sort_by` = third argument if present, must be `rank_nocontrol` or `rank_control`, defaults to `rank_nocontrol`

Display: "Will process {total} features, {batch_size} at a time, sorted by {sort_by}"

### Step 2: Create CSV Backup

Run:
```bash
py -3.12 batch_utils.py backup-csv
```

Display: "Created backup: Feature_output.csv.backup"

### Step 3: Find Candidate Features

Run:
```bash
py -3.12 batch_utils.py find-uninterpreted --sort-by {sort_by} --limit {total}
```

Parse the JSON output to extract the `features` array. This gives you the list of feature IDs with empty interpretations, sorted by the specified column.

Display: "Candidate features: {features list}"

### Step 4: Verify Pre-computed Data and Filter Existing

Run:
```bash
py -3.12 batch_utils.py check-existing --features {comma-separated feature IDs}
```

Parse the JSON output:
- **If `missing_precomputed` is non-empty:** Report which files are missing and stop. User must generate them first with:
  ```bash
  py -3.12 run_modal_utf8.py analyze_feature_json --feature-idx <ID>
  ```
- **If `has_output` is non-empty:** These features already have interpretations. Remove them from processing list.
  Display: "Skipping N features with existing output: X, Y, Z"

The features to process are those in `missing_output` that are also in `has_precomputed`.

Display: "Features to process: A, B, C, ..."

If no features remain to process, display "All features already have interpretations" and exit.

### Step 5: Process in Batches (LOOP)

Split the remaining feature list into chunks of size `batch_size`.

**For each batch:**

1. Display: "--- Batch {i}/{num_batches}: Processing features {list} ---"

2. For EACH feature in this batch, spawn a sub-agent using the Task tool:
   - `subagent_type: "general-purpose"`
   - Use this prompt template (replace {ID} with the feature index):
     ```
     Interpret SAE feature {ID}.

     1. Read the full instructions from: .claude/commands/interpret-and-challenge-existing.md
     2. Follow ALL steps in that file for feature {ID} (replace $ARGUMENTS with {ID})
     3. Produce outputs in: output/interpretations/feature{ID}/
     ```
   - **CRITICAL:** Spawn ALL agents for this batch in a SINGLE message (parallel execution)

3. Wait for all agents in this batch to complete

4. **Extract interpretations for this batch:**
   For each feature in this batch, run:
   ```bash
   py -3.12 batch_utils.py extract-interpretation --feature {ID}
   ```

   Parse the JSON output:
   - If `status` is "success": use the `interpretation` field
   - If `status` is "error": use "FAILED" as the interpretation

5. **Update CSV for each feature:**
   For each feature with a successful or failed interpretation, run:
   ```bash
   py -3.12 batch_utils.py update-csv --feature {ID} --interpretation "{interpretation text}"
   ```

   Display: "Batch {i} complete. Updated CSV with {n} interpretations."

6. Continue to next batch

### Step 6: Final Summary

After all batches complete, display:

```
=== BATCH PROCESSING COMPLETE ===
Total features processed: {count}
Batches completed: {num_batches}

| Feature | Verdict | Interpretation (truncated) |
|---------|---------|---------------------------|
| 21422   | CONFIRMED | Detects blank lines... |
| 483     | REFINED | Fires on specific token... |
...

Feature_output.csv has been updated.
```

## Error Handling

- **Missing results.json:** `extract-interpretation` returns error status, write "FAILED" in CSV
- **Malformed JSON:** `extract-interpretation` returns error status, write "FAILED" in CSV
- **Agent crash:** Partial batch results are still extracted, failed features get "FAILED"

## Utility Script Reference

The `batch_utils.py` script provides these commands:

| Command | Purpose |
|---------|---------|
| `backup-csv` | Create Feature_output.csv.backup |
| `find-uninterpreted --sort-by X --limit N` | Find features with empty interpretations |
| `check-existing --features 1,2,3` | Check for output files and pre-computed data |
| `extract-interpretation --feature N` | Pull label/verdict/summary from results.json |
| `update-csv --feature N --interpretation "..."` | Write interpretation back to CSV |

All commands output JSON for easy parsing.

## Example Run

User runs: `/interpret-batch 3 7`

```
Will process 7 features, 3 at a time, sorted by rank_nocontrol

> py -3.12 batch_utils.py backup-csv
Created backup: Feature_output.csv.backup

> py -3.12 batch_utils.py find-uninterpreted --sort-by rank_nocontrol --limit 7
Candidate features: 8134, 9404, 23933, 21422, 483, 13333, 17588

> py -3.12 batch_utils.py check-existing --features 8134,9404,23933,21422,483,13333,17588
Skipping 0 features with existing output
Features to process: 8134, 9404, 23933, 21422, 483, 13333, 17588

--- Batch 1/3: Processing features 8134, 9404, 23933 ---
[3 agents run in parallel]
> py -3.12 batch_utils.py extract-interpretation --feature 8134
> py -3.12 batch_utils.py update-csv --feature 8134 --interpretation "..."
[repeat for each feature]
Batch 1 complete. Updated CSV with 3 interpretations.

--- Batch 2/3: Processing features 21422, 483, 13333 ---
[3 agents run in parallel]
Batch 2 complete. Updated CSV with 3 interpretations.

--- Batch 3/3: Processing feature 17588 ---
[1 agent runs]
Batch 3 complete. Updated CSV with 1 interpretation.

=== BATCH PROCESSING COMPLETE ===
Total features processed: 7
Feature_output.csv has been updated.
```
