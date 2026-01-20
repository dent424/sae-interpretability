# Verify SAE Feature Interpretation

Verify the interpretation for feature with logic, causal masking, and reproducibility checks.

**Usage:**
- `/verify <feature_id>` - Full verification including repro check
- `/verify <feature_id> --skip-repro` - Skip reproducibility check

**Parse arguments:** Extract feature ID and check for `--skip-repro` flag from: **$ARGUMENTS**

---

## FILE SAFETY RULES (CRITICAL)

**This command must NEVER modify existing data.**

| File | Allowed Operation |
|------|-------------------|
| `results.json` | READ ONLY |
| `batch_test_*.json` | READ ONLY |
| `report.md`, `audit.jsonl` | READ ONLY |
| `verification.md` | CREATE or OVERWRITE (this file only) |
| `Feature_output.csv` | ADD columns + FILL empty cells only |

**CSV Rules:**
- May add new columns (`verify_status`, `verify_date`) if they don't exist
- May fill empty cells in verification columns
- **NEVER overwrite existing cell values** - if a cell has data, leave it unchanged
- **NEVER modify existing columns** (feature_index, interpretation, etc.)

---

## Pre-flight Check

Check if `output/interpretations/feature<ID>/results.json` exists. If not, stop:
> "No interpretation found for feature <ID>. Run /interpret <ID> first."

---

## Step 1: Load Data

Read `output/interpretations/feature<ID>/results.json`

Extract:
- `verdict` and `confidence`
- `interpretation_phase.test_results` (array of test outcomes)
- `interpretation_phase.hypotheses` (if exists)
- All interpretive text fields:
  - `label`, `description`, `category`
  - `necessary_conditions[]`, `boundary_conditions[]`, `does_not_detect[]`
  - `key_examples[].meaning`
  - `interpretation_phase.hypotheses[].description`
  - `interpretation_phase.initial_conclusion`
  - `challenge_phase.verdict_justification` (if exists)
  - `executive_summary`, `linguistic_function`

---

## Step 2: Logic Check

### 2a: Calculate Test Metrics

From `interpretation_phase.test_results`, count:
```
TP = expected=true AND actual=true
FP = expected=false AND actual=true
TN = expected=false AND actual=false
FN = expected=true AND actual=false

accuracy = (TP + TN) / total
precision = TP / (TP + FP) if (TP + FP) > 0 else N/A
recall = TP / (TP + FN) if (TP + FN) > 0 else N/A
```

### 2b: Assess Verdict Alignment

Check for mismatches (flag as WARN, not FAIL):

| Observation | Flag |
|-------------|------|
| SUPPORTED verdict with accuracy < 70% | WARN: Low accuracy for SUPPORTED |
| REJECTED verdict with accuracy > 65% | WARN: High accuracy for REJECTED |
| Confidence > 90% with FP > 2 | WARN: High confidence with multiple false positives |
| Confidence > 90% with accuracy < 80% | WARN: Confidence may be high for evidence |

### 2c: Logic Status
- **PASS** if no flags
- **WARN** if any flags

---

## Step 3: Causal Check

### 3a: Causal Masking Rule

GPT-2 uses causal attention. At position N, the model only sees tokens 0 to N-1. A feature firing on token X:
- CAN see: Everything to the LEFT of X
- CANNOT see: Anything to the RIGHT of X

### 3b: Review Interpretive Text

**Violations** (claims right-context dependency): "fires when followed by X", "detects tokens before [punctuation]", "anticipates/predicts future tokens", "depends on what comes after", "fires when next token is"

**Valid patterns** (left-context): "fires when preceded by X", "fires on [token] following [context]", "left context contains X", "after seeing X, fires on Y"

### 3c: Assess Each Field
- **PASS**: Correctly describes left-context patterns
- **WARN**: Ambiguous phrasing
- **FAIL**: Clear violation claiming right-context dependency

### 3d: Causal Status
PASS (all pass) | WARN (any ambiguous) | FAIL (any violations)

---

## Step 4: Repro Check

**Skip this step if `--skip-repro` flag was provided.** Set repro_status = "SKIPPED_BY_FLAG" and proceed to Step 5.

### 4a: Load Original Tests

**Primary Source:** Read test data from `output/interpretations/feature<ID>/results.json`
- Look for `interpretation_phase.test_results[]`
- Also check `challenge_phase.*` arrays if present (counterexamples, alternative_tests, minimal_pairs.grid, surprising_predictions)

**Check for required fields:** Each test should have `token_idx` and `all_tokens` fields.

**Fallback:** If results.json tests lack `token_idx` or `all_tokens` fields:
1. Try reading `output/interpretations/feature<ID>/batch_test_<ID>.json`
2. Match tests by `text` field to get the missing data
3. Note in report: "Used batch_test fallback for detailed test data"

**Skip repro if:**
- results.json has no test_results AND batch_test file doesn't exist
- Set repro_status = "SKIPPED" with message: "Insufficient test data for reproducibility check"

### 4b: Select 5 Tests
Choose:
- 2 tests that activated (highest activation values)
- 3 tests that did NOT activate

If fewer than 5 tests available, use all.

### 4c: Re-run Tests

**Shell Escaping Handling:**

Before running tests, check if any test text contains shell-sensitive characters: dollar sign ($), backtick, exclamation mark (!), or backslash.

If special characters are present:
1. First attempt: Run the test as-is
2. Check results: If the returned text differs from the input (characters stripped/modified), this indicates shell escaping corruption
3. Retry with escaping: For corrupted tests, apply minimal modifications:
   - Dollar signs: escape with backslash
   - Backticks: escape with backslash
   - Track which tests were modified for the report

**Run command:**
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx <ID> --output-dir output/interpretations/feature<ID>/repro --fresh --texts "text1|text2|text3|text4|text5"
```

Note: Uses a separate `repro/` subdirectory to avoid modifying the original batch_test file.

**After running, verify text integrity:**
- Read the output JSON file at `output/interpretations/feature<ID>/repro/batch_test_<ID>.json`
- Compare each result's `text` field to the original input
- If texts don't match (shell corruption detected):
  1. Identify which texts were corrupted
  2. Re-run ONLY those texts with escaped special characters
  3. Merge results

### 4d: Compare Results

For each test, check:
| Check | Pass Criteria |
|-------|--------------|
| Activation status | Both fired or both didn't |
| Value | Within ±0.03 OR ±15% (whichever larger) |
| Token | Same active_token (if fired) |

**For retried tests:** Compare against the original expected result, noting that escaping was required.

### 4e: Score
| Matches | Status |
|---------|--------|
| 5/5 | PASS |
| 4/5 | WARN |
| 3/5 | WARN |
| <3/5 | FAIL |

**Note in report if any tests required shell escaping workaround.**

---

## Step 5: Determine Overall Status

| Logic | Causal | Repro | Overall |
|-------|--------|-------|---------|
| PASS | PASS | PASS/SKIPPED* | PASS |
| WARN | PASS | PASS/SKIPPED* | WARN |
| PASS | WARN | PASS/SKIPPED* | WARN |
| Any | FAIL | Any | FAIL |
| FAIL | Any | Any | FAIL |
| Any | Any | FAIL | FAIL |

*SKIPPED includes SKIPPED_BY_FLAG and SKIPPED (file not found)

---

## Step 6: Generate Report

Create `output/interpretations/feature<ID>/verification.md`:

```markdown
# Verification: Feature <ID>

**Date:** [today] | **Overall:** [PASS/WARN/FAIL]

## Logic Check — **Status:** [PASS/WARN]
| Metric | Value |
|--------|-------|
| Total/TP/FP/TN/FN | N |
| Accuracy/Precision/Recall | X% |

**Verdict:** [from results.json] | **Confidence:** [from results.json]
[Flags or "Verdict and confidence align with test evidence."]

## Causal Check — **Status:** [PASS/WARN/FAIL]
[If violations:] | Field | Quote | Issue |
[If none:] All text correctly describes left-context patterns.

## Repro Check — **Status:** [PASS/WARN/FAIL/SKIPPED]
| # | Text (40 chars) | Original | Reproduced | Match | Notes |
**Matched:** N/5
[Notes on escaping if needed, or SKIPPED reason]

## Summary
| Check | Status |
|-------|--------|
| Logic/Causal/Repro/**Overall** | [status] |

[Recommendations if issues, or "Interpretation verified."]
```

---

## Step 7: Update CSV

Use the `batch_utils.py` utility to update verification status:

```bash
py -3.12 batch_utils.py update-verify --feature <ID> --status [PASS|WARN|FAIL]
```

This command:
- Adds `verify_status` and `verify_date` columns if they don't exist
- Only fills empty cells - never overwrites existing values
- Returns JSON with status (success, skipped, or error)

If the response shows `"status": "skipped"`, report to user:
> "Note: CSV already has verification data for this feature. Keeping existing values."

---

## Final Output

Report to user:
```
Verification complete for feature <ID>.

Overall: [PASS/WARN/FAIL]
- Logic: [status]
- Causal: [status]
- Repro: [status]

Report: output/interpretations/feature<ID>/verification.md
CSV updated: feature data/Feature_output.csv
```

---

## Begin

1. Parse arguments to extract feature ID and check for `--skip-repro` flag
2. Start with Step 1: Load the results.json file for the feature
