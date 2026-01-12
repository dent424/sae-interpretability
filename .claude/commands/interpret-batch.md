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

Create a backup of the CSV file:
1. Read `feature data/Feature_output.csv` using the Read tool
2. Write the contents to `feature data/Feature_output.csv.backup` using the Write tool

**Do NOT use Bash copy/cp commands** - use the Read → Write pattern instead.

Display: "Created backup: Feature_output.csv.backup"

### Step 3: Find Next Features

1. Read `feature data/Feature_output.csv`
2. Parse as CSV with columns: `feature_index`, `rank_control`, `rank_nocontrol`, `interpretation`
3. Filter rows where `interpretation` column is empty
4. Sort by `{sort_by}` ascending (use the column specified in Step 1)
5. Take first `total` rows
6. Extract the `feature_index` values into a list
7. Display: "Candidate features: 8134, 9404, 23933, ..."

### Step 4: Verify Pre-computed Data Exists

For each feature_index in the list, check that `feature data/feature_{ID}.json` exists (NOTE: underscore before ID).
- **If any are missing:** Report which files are missing and stop. User must generate them first with:
  ```bash
  py -3.12 run_modal_utf8.py analyze_feature_json --feature-idx <ID>
  ```
- Only proceed when ALL features have pre-computed data.

### Step 5: Filter Out Already-Interpreted Features

For each feature_index in the list:
1. Check if `output/interpretations/feature{ID}/results.json` exists (NOTE: no underscore in output folder)
2. If it exists: Remove from processing list (already has interpretation)

Display: "Skipping N features with existing output: X, Y, Z"
Display: "Features to process: A, B, C, ..."

If no features remain to process, display "All features already have interpretations" and exit.

### Step 6: Process in Batches (LOOP)

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
   - For each feature, read `output/interpretations/feature{ID}/results.json`
   - Extract `label`, `verdict`, and `executive_summary` fields
   - Format as: "{label} [{verdict}]: {executive_summary}"
   - **Truncate to 250 characters** if longer, adding "..." at end
   - **If results.json is missing or malformed:** Write "FAILED" as interpretation, log error, continue with others

5. **Update CSV immediately after each batch:**
   - Read `feature data/Feature_output.csv`
   - Update `interpretation` column for each feature in this batch
   - Use standard CSV escaping (wrap in quotes, double internal quotes)
   - Write back to CSV
   - Display: "Batch {i} complete. Updated CSV with {n} interpretations."

6. Continue to next batch

### Step 7: Final Summary

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

- **Missing results.json:** Write "FAILED" in CSV, continue with other features
- **Malformed JSON:** Write "FAILED" in CSV, log error, continue
- **Agent crash:** Partial batch results are still extracted, failed features get "FAILED"

## Example Run

User runs: `/interpret-batch 3 7`

```
Will process 7 features, 3 at a time, sorted by rank_nocontrol
Created backup: Feature_output.csv.backup
Candidate features: 8134, 9404, 23933, 21422, 483, 13333, 17588
Skipping 0 features with existing output
Features to process: 8134, 9404, 23933, 21422, 483, 13333, 17588

--- Batch 1/3: Processing features 8134, 9404, 23933 ---
[3 agents run in parallel]
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
