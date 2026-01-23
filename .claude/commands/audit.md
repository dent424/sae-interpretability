# Audit Feature Completeness

Audit interpreted features to verify they completed all major interpretation phases.

**Note:** Only features with a non-empty `interpretation` column in the CSV are eligible for audit.

**Usage:**
- `/audit` - Audit all pending features (those with interpretation but no completeness_status)
- `/audit <feature_id>` - Audit a single feature
- `/audit --limit N` - Only audit first N pending features

**Parse arguments:** Extract optional feature ID and `--limit N` from: **$ARGUMENTS**

---

## Completeness Criteria

A feature interpretation is **COMPLETE** if it has substantively gone through:

### 1. Interpretation Phase
- Generated multiple hypotheses about what the feature detects
- Ran discriminating tests to distinguish between hypotheses
- Scored hypotheses based on test results
- Selected a winning hypothesis with justification

### 2. Challenge Phase
- Tested counterexamples to stress-test the interpretation
- Ran minimal pairs or alternative tests
- Reached a verdict (CONFIRMED, REFINED, or REJECTED)

### 3. Final Output
- Has a clear label/name for the feature
- Has a description explaining what it detects
- Has a confidence score

**INCOMPLETE** if any major phase is missing or clearly placeholder/empty.

---

## Execution

### Step 1: Find Features to Audit

Read `feature data/Feature_output.csv` and identify features where:
- `interpretation` column is NOT empty
- `completeness_status` column IS empty (or doesn't exist)

If a specific feature ID was provided in arguments, audit only that feature.
If `--limit N` was provided, only audit the first N pending features.

### Step 2: Audit Each Feature

For each feature to audit, spawn an **opus** subagent to read and verify the content.

The subagent should use the Read tool to examine `output/interpretations/feature<ID>/results.json` and verify that actual work was done (not just that fields exist, but that they contain substantive content).

**Subagent prompt:**

```
Audit feature <ID>'s interpretation for completeness.

Read: output/interpretations/feature<ID>/results.json

Verify that actual work was done in each phase by checking for SUBSTANTIVE CONTENT (not just field existence):

1. INTERPRETATION PHASE - verify these actually happened:
   - Multiple distinct hypotheses (at least 2-3 with different descriptions)
   - Test results with actual activation values or outcomes
   - Hypothesis scores derived from test evidence
   - Winner selection with reasoning

2. CHALLENGE PHASE - verify these actually happened:
   - Counterexamples that were actually tested (not placeholder text)
   - Minimal pairs or alternative tests with results
   - A verdict (CONFIRMED/REFINED/REJECTED) with justification

3. FINAL OUTPUT - verify these exist:
   - A label/name for the feature
   - A description of what it detects
   - A confidence score (numeric)

Mark as INCOMPLETE if any phase has only placeholder content, empty arrays, or missing sections.
Mark as COMPLETE if all phases have substantive content showing the work was done.

Return ONLY a JSON object:
{"feature": <ID>, "status": "COMPLETE" or "INCOMPLETE", "reason": "<brief reason if INCOMPLETE>"}

If results.json doesn't exist:
{"feature": <ID>, "status": "INCOMPLETE", "reason": "no results.json file"}
```

### Step 3: Update CSV

For each audited feature, update the `completeness_status` column in the CSV:
- If COMPLETE: set to `COMPLETE`
- If INCOMPLETE: set to `INCOMPLETE:<reason>` (truncate reason to 50 chars)

Use the Read tool to get current CSV contents, then Write tool to update it.
Only update rows for features that were audited.
Add `completeness_status` column if it doesn't exist.

### Step 4: Report Results

Print summary:
```
Audit complete.
Total audited: N
- Complete: X
- Incomplete: Y

CSV updated: feature data/Feature_output.csv
```

---

## Example Subagent Response

For a complete feature:
```json
{"feature": 8134, "status": "COMPLETE", "reason": ""}
```

For an incomplete feature:
```json
{"feature": 6162, "status": "INCOMPLETE", "reason": "no challenge phase"}
```

---

## Begin

1. Parse arguments from $ARGUMENTS
2. Read CSV to find features needing audit
3. For each feature, spawn opus subagent to verify content completeness
4. Collect results and update CSV
5. Report summary
