# Audit Schema Command

Validates results.json files have all required top-level fields populated. Optionally fixes issues using backup-compare-verify procedure.

## Arguments

$ARGUMENTS - Optional: `[feature_ids]` and/or `--fix`

**Usage:**
```
/audit-schema                         # Audit all features with results.json
/audit-schema 5804                    # Audit single feature
/audit-schema 5804, 22080, 14292      # Audit specific features (comma-separated)
/audit-schema --fix                   # Audit all and fix issues interactively
/audit-schema 5804, 22080 --fix       # Audit specific features and fix
```

## Required Schema Fields

| Field | Type | Notes |
|-------|------|-------|
| `feature_idx` | number | Feature ID |
| `status` | string | Should be "complete" |
| `label` | string | Human-readable name |
| `description` | string | What the feature detects |
| `confidence` | number | 0.0-1.0 |
| `verdict` | string | CONFIRMED, REFINED, REFUTED, or UNCERTAIN |
| `corpus_stats` | object | Activation statistics |
| `interpretation_phase` | object | Hypothesis testing data |
| `challenge_phase` | object | Challenge testing data |

## Procedure

### Step 1: Parse Arguments

Parse `$ARGUMENTS` to determine:
- **feature_ids**: List of feature IDs to audit (comma-separated numbers). If empty, audit all.
- **fix_mode**: If `--fix` is present, enable interactive fixing

**Parsing logic:**
1. Remove `--fix` flag from arguments (note if present)
2. Split remaining text by commas
3. Extract all numbers as feature IDs
4. If no numbers found, feature_ids = [] (means "audit all")

### Step 2: Find Results Files

**If feature_ids is empty (audit all):**
Use Glob to find all results files:
```
output/interpretations/feature*/results.json
```

**If feature_ids has specific IDs:**
For each feature_id in the list, check:
```
output/interpretations/feature{feature_id}/results.json
```
Skip any that don't exist (report as "file not found" in results).

### Step 3: Validate Each File

For each results.json file:

1. Read the file using the Read tool
2. Parse the JSON content
3. Check each required field:
   - `feature_idx` - must exist and be a number
   - `status` - must exist (preferably "complete")
   - `label` - must exist and be non-empty string
   - `description` - must exist and be non-empty string
   - `confidence` - must exist and be a number 0.0-1.0
   - `verdict` - must exist and be one of: CONFIRMED, REFINED, REFUTED, UNCERTAIN
   - `corpus_stats` - must exist and be an object
   - `interpretation_phase` - must exist and be an object
   - `challenge_phase` - must exist and be an object

4. Record validation results:
   - Feature ID
   - Which fields are present (✓) or missing (✗)
   - Overall status: OK or list of missing fields

### Step 4: Display Results Table

Output a markdown table:

```
=== Schema Audit Results ===

| Feature | label | description | confidence | verdict | Status |
|---------|-------|-------------|------------|---------|--------|
| 5804    | ✓     | ✓           | ✓          | ✓       | OK     |
| 22080   | ✗     | ✗           | ✓          | ✓       | MISSING: label, description |

Summary: X/Y files pass schema validation
Missing fields: Z files need fixes
```

### Step 5: Interactive Fix (if --fix flag present)

For each file that fails validation:

#### 5a. Display Issue
```
=== Feature {ID} needs fixes ===
Missing fields: label, description
File: output/interpretations/feature{ID}/results.json
```

#### 5b. Ask User
Use AskUserQuestion:
- Question: "Attempt fix for feature {ID}?"
- Options: "Yes", "No", "Skip all remaining"

#### 5c. If User Says Yes - Apply Fix

1. **Create Backup**:
   - Read the current results.json
   - Write contents to results.json.backup (using Read → Write, NOT bash copy)

2. **Determine Fix Values**: Look for missing data in nested structures:

   | Missing Field | Where to Find It |
   |---------------|------------------|
   | `label` | `interpretation_phase.initial_label` OR `challenge_phase.refined_label` |
   | `description` | `interpretation_phase.initial_conclusion` OR `interpretation_phase.executive_summary` |
   | `confidence` | `interpretation_phase.initial_confidence` OR `challenge_phase.final_confidence` |
   | `verdict` | `challenge_phase.verdict` OR `challenge_phase.challenge_verdict.outcome` |

   If not found in nested data, use placeholder: `"[MISSING - needs manual review]"`

3. **Normalize Verdicts**: If verdict exists but in wrong format:
   - "survived" → "CONFIRMED"
   - "refined" → "REFINED"
   - "refuted" → "REFUTED"
   - anything else → "UNCERTAIN"

4. **Apply Edits**: Use Edit tool to add missing fields to the results.json

#### 5d. Show Semantic Diff

After making changes, display what changed:

```
=== Diff for feature {ID} ===

ADDED fields:
+ "label": "Extracted label from interpretation_phase"
+ "description": "Extracted from initial_conclusion"

CHANGED fields:
~ "verdict": "survived" → "CONFIRMED"

Backup saved to: results.json.backup
```

**Implementation**: Compare the backup JSON and edited JSON programmatically:
- Parse both files
- For each required field, compare values
- Report ADDED (was missing/null, now has value), CHANGED (value different), or unchanged

#### 5e. Confirm Changes

Use AskUserQuestion:
- Question: "Keep these changes for feature {ID}?"
- Options: "Yes, keep changes", "No, restore backup"

**If Yes**:
- Use Bash to delete the backup file: `del "output/interpretations/feature{ID}/results.json.backup"`
- Report: "Changes saved. Backup removed."
- Continue to next file

**If No**:
- Read results.json.backup
- Write contents back to results.json (overwriting the edits)
- Delete backup file
- Report: "Changes reverted from backup."
- Continue to next file

### Step 6: Update CSV

After auditing all files, update the `schema_status` column in `feature data/Feature_output.csv`:

1. **Read the CSV** using the Read tool
2. **Add `schema_status` column** if it doesn't exist (append after last column in header)
3. **For each audited feature**, find its row and set `schema_status` to:
   - `OK` - all required fields present
   - `MISSING:field1,field2` - list missing fields (truncate to 50 chars)
   - `MODIFIED:field1,field2` - fields were fixed during this run (with --fix)
   - `NO_FILE` - results.json not found
4. **Write the updated CSV** using the Write tool

**Only update rows for features that were audited.** Leave other rows unchanged.

### Step 7: Final Summary

After processing all files (or after --fix completes):

```
=== Audit Complete ===

Files audited: X
Passed: Y
Fixed: Z
Still need attention: W

CSV updated: feature data/Feature_output.csv (schema_status column)

Files needing manual review:
- feature 12345: missing challenge_phase (cannot auto-fix)
- feature 67890: confidence value invalid (0.95abc)
```

## Example Output (No Fix Mode)

**Audit all features:**
```
=== Schema Audit Results ===

| Feature | label | desc | conf | verdict | corpus | interp | challenge | Status |
|---------|-------|------|------|---------|--------|--------|-----------|--------|
| 5804    | ✓     | ✓    | ✓    | ✓       | ✓      | ✓      | ✓         | OK     |
| 22080   | ✗     | ✗    | ✓    | ✓       | ✓      | ✓      | ✓         | MISSING |
| 14292   | ✓     | ✓    | ✓    | ✓       | ✓      | ✓      | ✓         | OK     |

Summary: 121/123 files pass schema validation
3 files need fixes (run with --fix to repair)
```

**Audit specific features (`/audit-schema 5804, 22080, 99999`):**
```
=== Schema Audit Results ===

| Feature | label | desc | conf | verdict | corpus | interp | challenge | Status |
|---------|-------|------|------|---------|--------|--------|-----------|--------|
| 5804    | ✓     | ✓    | ✓    | ✓       | ✓      | ✓      | ✓         | OK     |
| 22080   | ✗     | ✗    | ✓    | ✓       | ✓      | ✓      | ✓         | MISSING |
| 99999   | -     | -    | -    | -       | -      | -      | -         | FILE NOT FOUND |

Summary: 1/3 files pass schema validation
1 file needs fixes, 1 file not found
```

## Notes

- This command uses Read/Write tools for file operations, not bash copy/cat
- Backups are created before any modification
- User confirms every change before it's finalized
- Verdict normalization handles common variations (lowercase, "survived" etc.)
- Missing nested phases (corpus_stats, interpretation_phase, challenge_phase) cannot be auto-fixed - report for manual review
