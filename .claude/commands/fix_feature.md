# Fix Feature Command

Repair salvageable SAE feature interpretations by extracting fields from nested locations and generating synthesis fields.

## Arguments

$ARGUMENTS - Required: `<feature_id>` or `<id1>, <id2>, ...` or `--salvageable` and optional `--dry-run`

**Usage:**
```
/fix_feature <feature_id>              # Fix single feature
/fix_feature <id1>, <id2>, ...         # Fix specific features (comma-separated)
/fix_feature --salvageable             # Fix all with INCOMPLETE/SCHEMA status
/fix_feature --dry-run [ids]           # Preview what would be fixed
```

---

## What Can Be Fixed

### Type 1: Field Extraction (schema_fix)
Move fields from nested structures to top-level:

| Missing Field | Extract From |
|---------------|--------------|
| `label` | `interpretation_phase.initial_label` OR `final_interpretation.label` |
| `description` | `interpretation_phase.initial_conclusion` OR `final_interpretation.description` |
| `confidence` | `interpretation_phase.initial_confidence` OR `final_interpretation.confidence` |
| `verdict` | `challenge_phase.verdict` |
| `necessary_conditions` | `final_interpretation.necessary_conditions` |

**Verdict Normalization:**
- "survived" → "CONFIRMED"
- "refined" → "REFINED"
- "refuted" → "REFUTED"
- "PASS_WITH_REVISION" → "REFINED"
- "passed" → "CONFIRMED"

### Type 2: Synthesis Generation (synthesis_fix)
Generate missing synthesis fields from interpretation data:

| Field | Generated From |
|-------|----------------|
| `key_examples` | Top activations with highest values (3-5 examples) |
| `executive_summary` | Label + description + verdict combined |
| `linguistic_function` | Hypothesis winner description |
| `potential_applications` | Feature detection pattern |
| `category` | Inferred from label (lexical/syntactic/discourse/etc) |
| `boundary_conditions` | Challenge phase counterexamples |
| `does_not_detect` | Failed tests or challenge findings |

### What Cannot Be Fixed
- Missing `interpretation_phase` (no hypotheses) → needs re-interpretation
- Missing `challenge_phase` (no tests) → needs re-interpretation
- Missing `results.json` → needs full interpretation

---

## CSV Column: fix_status

Records what type of fix was applied:

| Value | Meaning |
|-------|---------|
| `schema_fix` | Only extracted fields from nested locations |
| `synthesis_fix` | Only generated synthesis fields |
| `schema+synthesis` | Both extraction and generation applied |
| `unsalvageable` | Cannot be fixed - needs re-interpretation |
| *(empty)* | Not yet processed by /fix_feature |

---

## Procedure

### Step 1: Parse Arguments

Parse `$ARGUMENTS` to determine:
- **feature_ids**: List of specific feature IDs (comma-separated numbers), or empty if using --salvageable
- **salvageable_mode**: If `--salvageable` is present, find all fixable features from CSV
- **dry_run**: If `--dry-run` is present, only preview without modifications

### Step 2: Find Features to Fix

**If specific feature_ids provided:**
- Validate each ID exists in CSV
- Verify `output/interpretations/feature{ID}/results.json` exists
- Skip any that don't exist (with warning)

**If --salvageable mode:**
Run batch_utils to find features with non-COMPLETE audit status:
```bash
py -3.12 batch_utils.py find-fixable --limit 100
```

This returns features from Feature_output.csv where:
- `audit_status` starts with `INCOMPLETE` or `SCHEMA`
- `results.json` exists in `output/interpretations/feature{ID}/`
- `fix_status` is empty or not `unsalvageable`

Parse the JSON output to get the feature list.

### Step 3: Assess Salvageability (Per Feature)

For each feature, spawn an assessment sub-agent to determine what can be fixed:

**Spawn assessment sub-agent** using Task tool:
- `subagent_type`: "general-purpose"
- `model`: "haiku" (fast assessment)
- `description`: "Assess feature {ID}"
- `prompt`: Use the Assessment Prompt Template below

**Assessment Prompt Template:**
```
Assess salvageability of SAE feature interpretation.

**Feature ID:** {feature_id}
**Results Path:** output/interpretations/feature{feature_id}/results.json
**Audit Report:** output/interpretations/feature{feature_id}/audit_report.md

## Task

Read both files and assess what can be fixed.

### Check 1: Core Data Exists
Verify these exist in results.json:
- `interpretation_phase` with `hypotheses` array (at least 1)
- `interpretation_phase.test_results` (at least 1 test)
- `challenge_phase` with `verdict` or `counterexamples`

If missing: NOT salvageable (needs re-interpretation)

### Check 2: Field Extraction Possible
Check if missing top-level fields exist in nested locations:
- `label` in `interpretation_phase.initial_label` or `final_interpretation.label`
- `description` in `interpretation_phase.initial_conclusion` or `final_interpretation.description`
- `confidence` in `interpretation_phase.initial_confidence` or `final_interpretation.confidence`
- `verdict` in `challenge_phase.verdict`
- `necessary_conditions` in `final_interpretation.necessary_conditions`

List which fields can be extracted.

### Check 3: Synthesis Generation Possible
Check if synthesis fields are missing but can be generated:
- `key_examples` (can generate if `top_activations` exists)
- `executive_summary` (can generate if label + description exist)
- `linguistic_function` (can generate if winner exists)
- `potential_applications` (can generate if label exists)
- `category` (can infer if label exists)
- `boundary_conditions` (can extract if counterexamples exist)
- `does_not_detect` (can extract if test failures exist)

List which fields can be generated.

## Response Format
---ASSESS-RESULT---
FEATURE_ID: {number}
SALVAGEABLE: {FULLY|PARTIALLY|NO}
SALVAGE_REASON: {brief explanation}
CAN_EXTRACT: {comma-separated field names or "none"}
CAN_GENERATE: {comma-separated field names or "none"}
STILL_MISSING_AFTER_FIX: {fields that will still be missing or "none"}
---END-ASSESS---
```

### Step 4a: Display Assessment (If --dry-run)

If `--dry-run` mode:
1. Display assessment results for each feature
2. Show what would be fixed vs what needs re-interpretation
3. **EXIT without modifying any files**

Display format:
```
=== Dry Run: Fix Feature Assessment ===

Feature 1219:
  Salvageable: FULLY
  Can extract: label, description, confidence, verdict
  Can generate: key_examples, executive_summary
  Would become: COMPLETE

Feature 19742:
  Salvageable: NO
  Reason: Missing interpretation_phase.hypotheses
  Action needed: Re-run /interpret-and-challenge

Summary:
- Fully salvageable: 3
- Partially salvageable: 2
- Not salvageable: 1
```

### Step 4b: Apply Fixes (If NOT --dry-run)

For each salvageable feature, spawn a fix sub-agent:

**Spawn fix sub-agent** using Task tool:
- `subagent_type`: "general-purpose"
- `model`: "opus" (careful file modifications)
- `description`: "Fix feature {ID}"
- `prompt`: Use the Fix Prompt Template below

**IMPORTANT:** Process features sequentially to avoid conflicts.

**Fix Prompt Template:**
```
Fix incomplete SAE feature interpretation.

**Feature ID:** {feature_id}
**Results Path:** output/interpretations/feature{feature_id}/results.json
**Assessment:** {paste assessment results}

## Task

### Step 1: Create Backup
- Read the current results.json
- Write exact copy to results.json.bak using Write tool

### Step 2: Extract Fields (If CAN_EXTRACT is not "none")

Extract missing top-level fields from nested structures:

For each field in CAN_EXTRACT:
- `label` ← `interpretation_phase.initial_label` OR `final_interpretation.label`
- `description` ← `interpretation_phase.initial_conclusion` OR `final_interpretation.description`
- `confidence` ← `interpretation_phase.initial_confidence` OR `final_interpretation.confidence`
- `verdict` ← `challenge_phase.verdict`
- `necessary_conditions` ← `final_interpretation.necessary_conditions`

Normalize verdict values:
- "survived" → "CONFIRMED"
- "refined" → "REFINED"
- "refuted" → "REFUTED"
- "PASS_WITH_REVISION" → "REFINED"
- "passed" → "CONFIRMED"

### Step 3: Generate Synthesis Fields (If CAN_GENERATE is not "none")

For each field in CAN_GENERATE:

**key_examples** (array of 3-5):
- Select from `top_activations` with highest activation values
- Format each as: { "context": "...", "token": "...", "activation": N, "meaning": "brief explanation" }

**executive_summary** (string, 50-150 chars):
- Combine label + description + challenge verdict
- Example: "Detects emphatic expressions like 'why in the world' [CONFIRMED with refinements for rhetorical questions]"

**linguistic_function** (string):
- Describe linguistic role based on hypothesis winner
- Example: "Marks discourse-level emphasis and speaker surprise"

**potential_applications** (string):
- Suggest applications based on detection pattern
- Example: "Sentiment intensity detection, rhetorical question identification"

**category** (string, if missing):
- Infer from label: "lexical", "syntactic", "discourse", "formatting", "semantic"

**boundary_conditions** (array, if missing):
- Extract from `challenge_phase.counterexamples`
- List conditions where feature may NOT activate as expected

**does_not_detect** (array, if missing):
- Extract from failed tests or challenge findings
- List patterns that look similar but don't activate the feature

### Step 4: Determine Fix Type

Based on what was done:
- Step 2 only → `schema_fix`
- Step 3 only → `synthesis_fix`
- Both Step 2 and Step 3 → `schema+synthesis`

### Step 5: Write Updated File

Write the complete updated JSON to results.json using Write tool.

### Step 6: Verify Changes

- Read both results.json.bak and results.json
- Confirm original data preserved (no deletions)
- Confirm new fields added correctly
- Confirm extracted values match source locations

## Response Format
---FIX-RESULT---
FEATURE_ID: {number}
SALVAGEABLE: {FULLY|PARTIALLY|NO}
SALVAGE_REASON: {why salvageable or not}
BACKUP_CREATED: {yes|no}
FIELDS_EXTRACTED: {list or "none"}
FIELDS_GENERATED: {list or "none"}
FIX_TYPE: {schema_fix|synthesis_fix|schema+synthesis|unsalvageable}
STILL_MISSING: {list or "none"}
VERIFICATION: {PASSED|FAILED}
---END-FIX---
```

### Step 5: Re-Audit and Update CSV (If Fixes Applied)

For each fixed feature:

1. **Re-audit the feature** to determine new status:
   - Read the updated results.json
   - Validate against schema requirements
   - Determine new audit_status (COMPLETE, INCOMPLETE, etc.)

2. **Update CSV with both statuses:**

   First, force-update audit_status (allow overwrite since we just fixed it):
   ```bash
   py -3.12 batch_utils.py force-update-audit --feature {feature_id} --status {new_audit_status}
   ```

   Then update fix_status:
   ```bash
   py -3.12 batch_utils.py update-fix-status --feature {feature_id} --status {fix_type}
   ```

### Step 6: Display Summary

```
=== Fix Feature Complete ===

Features processed: 8
- schema_fix: 2 (1219, 2597)
- synthesis_fix: 1 (7327)
- schema+synthesis: 3 (23472, 7907, 1022)
- unsalvageable: 2 (19742, 16686)

CSV updated: feature data/Feature_output.csv
  - audit_status: re-validated after fixes
  - fix_status: records fix type applied

Features now COMPLETE after fix: 5
Features still INCOMPLETE: 1
Features unsalvageable (need re-interpretation): 2
  - 19742: Missing hypotheses
  - 16686: Missing test results
```

---

## Begin

1. Parse arguments from $ARGUMENTS
2. Find features to fix (specific IDs or --salvageable)
3. Assess salvageability for each feature
4. If --dry-run: display assessment and exit
5. If not --dry-run: apply fixes for salvageable features
6. Re-audit fixed features and update CSV
7. Display summary
