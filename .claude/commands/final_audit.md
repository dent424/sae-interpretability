# Final Audit Command

Comprehensive validation combining schema validation, content completeness, and process verification for SAE feature interpretations.

**Model:** Sub-agents inherit **Opus 4.5** from parent (do NOT specify model parameter).

## Arguments

$ARGUMENTS - Optional: `[feature_ids]` and/or `--limit N`

**Usage:**
```
/final_audit                           # Audit all features with results.json
/final_audit <feature_id>              # Audit single feature
/final_audit <id1>, <id2>, ...         # Audit specific features (comma-separated)
/final_audit --limit N                 # Only audit first N features
```

---

## Three-Layer Validation

### Layer 1: Schema Validation
Required fields exist with correct types:
- `feature_idx` (number)
- `status` (string)
- `label` (non-empty string)
- `category` (string)
- `description` (non-empty string)
- `confidence` (number 0.0-1.0)
- `verdict` (one of: CONFIRMED, REFINED, REFUTED, UNCERTAIN)
- `necessary_conditions` (array)
- `boundary_conditions` (array)
- `does_not_detect` (array)
- `corpus_stats` (object)
- `top_tokens` (array)
- `top_activations` (array)
- `ngram_analysis` (object)
- `interpretation_phase` (object with: hypotheses[3], test_results, hypothesis_scores, winner)
- `challenge_phase` (object with: counterexamples, verdict, verdict_justification)
- `key_examples` (array)
- `executive_summary` (non-empty string)
- `linguistic_function` (non-empty string)
- `potential_applications` (non-empty string)

### Layer 2: Content Validation
Substantive content checks (not just field existence):

**Interpretation Phase:**
- Exactly 3 hypotheses with distinct descriptions
- Test results with actual activation values
- Hypothesis scores derived from test evidence
- Winner selection with justification

**Challenge Phase:**
- Counterexamples that were actually tested (with activation values)
- Verdict (CONFIRMED/REFINED/REFUTED/UNCERTAIN) with justification

**Synthesis:**
- Key examples with meaningful annotations
- Executive summary that describes the feature

### Layer 3: Process Validation
Check `audit.jsonl` for required steps:

| Step | Name | Required |
|------|------|----------|
| 1.1 | Load Existing Data | ✓ |
| 1.2 | Generate Hypotheses | ✓ |
| 1.3 | Design Tests | ✓ |
| 1.4 | Context Ablation | optional |
| 1.5 | Evaluate Hypothesis Support | ✓ |
| 2.1 | Counterexample Hunt | ✓ |
| 2.2 | Alternative Explanation Tests | optional |
| 2.3 | Minimal Pair Grid | optional |
| 2.4 | Surprising Predictions | optional |
| 2.5 | Challenge Verdict | ✓ |
| 3.1 | Finalize JSON | ✓ |
| 3.2 | Write Report | ✓ |

---

## Output Status Values

| Schema | Content | Process | Status |
|--------|---------|---------|--------|
| PASS | PASS | PASS | `COMPLETE` |
| PASS | PASS | FAIL | `PROCESS_INCOMPLETE:<missing steps>` |
| PASS | FAIL | - | `INCOMPLETE:<reason>` |
| FAIL | - | - | `SCHEMA:<missing fields>` |

---

## Procedure

### Step 1: Parse Arguments

Parse `$ARGUMENTS` to determine:
- **feature_ids**: List of feature IDs (comma-separated numbers). If empty, audit all.
- **limit**: If `--limit N` is present, only audit first N features

### Step 2: Find Features to Audit

**If feature_ids is empty (audit all):**
Run batch_utils to find features with interpretations that need auditing:
```bash
py -3.12 batch_utils.py find-auditable --limit {N if specified, else 100}
```

This returns features from Feature_output.csv where:
- `interpretation` column is non-empty (feature has been interpreted)
- `audit_status` column is empty (not yet audited)
- `results.json` exists in `output/interpretations/feature{ID}/`

Parse the JSON output to get the feature list.

**If feature_ids has specific IDs:**
Use the provided IDs directly. Verify each has:
- A non-empty interpretation in the CSV
- A results.json file at `output/interpretations/feature{ID}/results.json`

Apply `--limit N` if specified.

### Step 3: Audit Each Feature (Sequential)

**IMPORTANT:** Process ONE feature at a time to manage context.

For each feature_id:

1. **Spawn validation sub-agent** using Task tool:
   - `subagent_type`: "general-purpose"
   - `description`: "Audit feature {ID}"
   - `prompt`: Use the Sub-Agent Prompt Template below

2. **Parse sub-agent response** to extract the `---AUDIT-RESULT---` block

3. **Display result** after each feature:
   ```
   === [3/40] Feature 8134 ===
   Status: COMPLETE
   Schema: ✓ (28/28 required fields)
   Content: ✓ (3 hypotheses, 4 tests, verdict REFINED)
   Process: ✓ (8/8 required steps)
   ```

### Sub-Agent Prompt Template

```
Perform comprehensive audit of SAE feature interpretation.

**Feature ID:** {feature_id}
**Results Path:** output/interpretations/feature{feature_id}/results.json
**Audit Path:** output/interpretations/feature{feature_id}/audit.jsonl
**Report Path:** output/interpretations/feature{feature_id}/audit_report.md

## Task

Read both files, validate all three layers, write the audit report, and update the CSV.

### Layer 1: Schema Validation

Check these REQUIRED top-level fields exist with correct types:

| Field | Type | Validation |
|-------|------|------------|
| feature_idx | number | Must exist |
| status | string | Should be "complete" |
| label | string | Non-empty |
| category | string | Must exist |
| description | string | Non-empty |
| confidence | number | 0.0-1.0 |
| verdict | string | CONFIRMED, REFINED, REFUTED, or UNCERTAIN |
| necessary_conditions | array | Must exist |
| boundary_conditions | array | Must exist |
| does_not_detect | array | Must exist |
| corpus_stats | object | Must exist |
| top_tokens | array | Must exist |
| top_activations | array | Must exist |
| ngram_analysis | object | Must exist |
| interpretation_phase | object | Must exist |
| interpretation_phase.hypotheses | array | Exactly 3 items |
| interpretation_phase.test_results | array | Non-empty |
| interpretation_phase.hypothesis_scores | array | Non-empty |
| interpretation_phase.winner | object | Must exist |
| challenge_phase | object | Must exist |
| challenge_phase.counterexamples | array | Non-empty |
| challenge_phase.verdict | string | CONFIRMED, REFINED, REFUTED, or UNCERTAIN |
| challenge_phase.verdict_justification | string | Non-empty |
| key_examples | array | Non-empty |
| executive_summary | string | Non-empty |
| linguistic_function | string | Non-empty |
| potential_applications | string | Non-empty |

**Check for alternative field locations if missing:**
- `verdict` → also check `challenge_phase.verdict` or `final_interpretation.verdict`
- `label` → also check `interpretation_phase.initial_label` or `final_interpretation.label`
- `description` → also check `interpretation_phase.initial_conclusion` or `final_interpretation.description`
- `confidence` → also check `interpretation_phase.initial_confidence` or `final_interpretation.confidence`

### Layer 2: Content Validation

Verify substantive content (not placeholders):

**Interpretation Phase:**
- hypotheses: Exactly 3 with DISTINCT descriptions (not duplicates)
- test_results: At least 1 test with actual activation values
- hypothesis_scores: Scores that reflect test outcomes
- winner: Has id and justification

**Challenge Phase:**
- counterexamples: At least 1 with activation value
- verdict: One of CONFIRMED/REFINED/REFUTED/UNCERTAIN
- verdict_justification: Non-empty explanation

**Synthesis:**
- key_examples: At least 1 with context, token, and meaning
- executive_summary: At least 50 characters describing the feature
- linguistic_function: Non-empty
- potential_applications: Non-empty

### Layer 3: Process Validation

Read audit.jsonl and check for required steps:

**Required steps (must be present):**
- 1.1 Load Existing Data
- 1.2 Generate Hypotheses
- 1.3 Design Tests (may also be "Design and Run Discriminating Tests")
- 1.5 Evaluate Hypothesis Support
- 2.1 Counterexample Hunt
- 2.5 Challenge Verdict
- 3.1 Finalize JSON
- 3.2 Write Report

**Optional steps (note if present):**
- 1.4 Context Ablation
- 2.2 Alternative Explanation Tests
- 2.3 Minimal Pair Grid
- 2.4 Surprising Predictions

If audit.jsonl doesn't exist, mark process as INCOMPLETE.

### Step 4: Write Audit Report

After validation, write the audit report to the Report Path using the Write tool:

```markdown
# Audit Report: Feature {ID}

**Date:** {YYYY-MM-DD} | **Status:** {OVERALL_STATUS}

## Required Fields

| Field | Status | Notes |
|-------|--------|-------|
| feature_idx | {checkmark or X} | {value or MISSING} |
| status | {checkmark or X} | {value or MISSING} |
| label | {checkmark or X} | {truncated value or MISSING} |
... (all 27 required fields with actual values from validation)

## Optional Fields

| Field | Status |
|-------|--------|
| interpretation_phase.test_design | {checkmark or dash} |
... (all 5 optional fields)

## Process Validation (audit.jsonl)

| Step | Name | Status |
|------|------|--------|
| 1.1 | Load Existing Data | {checkmark or X} |
... (all 12 steps)

## Summary

- **Required fields present:** {N}/27
- **Missing required:** {list or "none"}
- **Optional fields present:** {N}/5
- **Required steps logged:** {N}/8
- **Optional steps logged:** {N}/4
```

### Step 5: Update CSV

Run this command to update the CSV with the audit status:

```bash
py -3.12 batch_utils.py update-audit --feature {feature_id} --status {OVERALL_STATUS}
```

Status values:
- `COMPLETE` - all validations passed
- `SCHEMA:<missing fields>` - schema validation failed
- `INCOMPLETE:<reason>` - content validation failed
- `PROCESS_INCOMPLETE:<missing steps>` - process validation failed

## STRICT STATUS RULES (MANDATORY - NO EXCEPTIONS)

These rules are MANDATORY. Do NOT rationalize, excuse, or override them for any reason:

1. If ANY required schema field is missing/invalid → OVERALL_STATUS = SCHEMA
2. If ANY content validation fails → OVERALL_STATUS = INCOMPLETE
3. If ANY required process step is missing from audit.jsonl → OVERALL_STATUS = PROCESS_INCOMPLETE
4. ONLY if ALL THREE layers fully PASS → OVERALL_STATUS = COMPLETE

**There are NO partial passes. There are NO "minor issues" or "documentation gaps."**
A missing step IS a failure. An empty field IS a failure.
Follow these rules LITERALLY and EXACTLY.

Your OVERALL_STATUS must be MECHANICALLY derived from the three layer statuses:
- SCHEMA_STATUS=FAIL → OVERALL_STATUS starts with "SCHEMA"
- CONTENT_STATUS=FAIL → OVERALL_STATUS starts with "INCOMPLETE"
- PROCESS_STATUS=FAIL → OVERALL_STATUS starts with "PROCESS_INCOMPLETE"
- All PASS → OVERALL_STATUS = "COMPLETE"

## Response Format

After writing the report and updating the CSV, return EXACTLY this format:

```
---AUDIT-RESULT---
FEATURE_ID: {number}
SCHEMA_STATUS: {PASS|FAIL}
SCHEMA_MISSING: {comma-separated list or "none"}
SCHEMA_ALTERNATIVES: {fields found in alt locations or "none"}
CONTENT_STATUS: {PASS|FAIL}
CONTENT_ISSUES: {comma-separated issues or "none"}
PROCESS_STATUS: {PASS|FAIL|NO_FILE}
PROCESS_MISSING: {comma-separated missing required steps or "none"}
PROCESS_OPTIONAL: {comma-separated present optional steps or "none"}
OVERALL_STATUS: {COMPLETE|INCOMPLETE|SCHEMA|PROCESS_INCOMPLETE}
OVERALL_REASON: {brief reason if not COMPLETE}
REPORT_WRITTEN: {yes|no}
CSV_UPDATED: {yes|no}
---END-RESULT---
```

**Status determination:**
- If SCHEMA_STATUS=FAIL: OVERALL_STATUS=SCHEMA
- Else if CONTENT_STATUS=FAIL: OVERALL_STATUS=INCOMPLETE
- Else if PROCESS_STATUS=FAIL: OVERALL_STATUS=PROCESS_INCOMPLETE
- Else: OVERALL_STATUS=COMPLETE
```

### Step 4: Display Summary (Main Procedure)

```
=== Final Audit Complete ===

Features audited: X
- COMPLETE: Y
- INCOMPLETE: Z (content issues)
- SCHEMA: W (missing fields)
- PROCESS_INCOMPLETE: V (missing steps)

CSV updated: feature data/Feature_output.csv (audit_status column)

Reports written: output/interpretations/feature*/audit_report.md
```

If any features need attention, list them:
```
Features needing manual review:
- Feature 12345: SCHEMA:label;description (cannot auto-fix synthesis fields)
- Feature 67890: INCOMPLETE:no counterexamples tested
```

---

## Fixing Issues

To repair features with INCOMPLETE or SCHEMA status, use the `/fix_feature` command:

```
/fix_feature <feature_id>        # Fix specific feature
/fix_feature --salvageable       # Fix all salvageable features
/fix_feature --dry-run           # Preview what would be fixed
```

See `/fix_feature` for details on what can be automatically repaired.

---

## Begin

1. Parse arguments from $ARGUMENTS
2. Find features to audit
3. For each feature, spawn validation sub-agent (which writes report and updates CSV)
4. Display summary
