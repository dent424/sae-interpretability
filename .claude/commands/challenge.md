# Challenge SAE Feature Interpretation

Challenge and stress-test the interpretation for feature **$ARGUMENTS**.

## Pre-flight Check

**First**, check if `output/interpretations/feature$ARGUMENTS/results.json` contains `challenge_results` or `challenge_phase`. If so, ask the user: "Feature $ARGUMENTS already has challenge results. Re-challenge or Abort?"

---

## Your Task

You are a **devil's advocate** for SAE feature interpretations. Your job is to **break** the current hypothesis, not confirm it. A good interpretation should survive adversarial testing.

## Step 1: Load Current Interpretation

Read the existing interpretation:
- `output/interpretations/feature$ARGUMENTS/results.json` (structured data)
- `output/interpretations/feature$ARGUMENTS/report.md` (full report)

Extract:
1. **The hypothesis**: What pattern does the interpretation claim?
2. **The evidence**: What tests supported it?
3. **Confidence level**: How certain was the original interpretation?

If no interpretation exists, stop and tell the user to run `/interpret $ARGUMENTS` first.

## Step 2: Generate Adversarial Tests

Design tests specifically intended to **FALSIFY** the hypothesis. Think like a skeptic.

### 2a: Counterexample Hunt (5-8 tests)
Texts that **fit pattern but you suspect WON'T fire**: edge cases, unusual contexts, domain transfers, archaic/formal/slang variants.

### 2b: Alternative Explanation Tests (5-8 tests)
Test for confounds: tokenization artifacts, position effects, frequency confounds, spurious correlations (detecting something NEARBY).

### 2c: Minimal Pair Grid (9-16 tests)
Vary ONE element at a time. Example grid for "Imperative + your":
| | "your" | "my" | "the" | "our" |
|---|---|---|---|---|
| "Save/Keep/Take/Get" | test | test | test | test |
Isolates whether BOTH elements are necessary.

### 2d: Surprising Predictions (2-3 tests)
If hypothesis TRUE, what unexpected text SHOULD activate? Strong evidence if it fires.

## Step 3: Run All Challenge Tests

Batch all tests in a single call (appends to existing batch_test file):
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --output-dir output/interpretations/feature$ARGUMENTS --texts "counterexample1|counterexample2|...|grid_test1|grid_test2|..."
```

**SAFEGUARD:** If ALL activations are 0.0:
1. Try a corpus context from `top_activations` (e.g., reconstruct "...I'm happy** to** say..." as full text)
2. If corpus context also fails → **STOP.** Flag: "Feature [N] won't activate. Check feature index or pipeline."
3. If corpus works but synthetic fails → redesign test examples based on corpus patterns

## Step 4: Analyze Results

For each test category, evaluate:

### Counterexamples
- Did any "should work but won't" texts actually fire? → Hypothesis is **more robust** than expected
- Did any fail as predicted? → Hypothesis has **boundary conditions** to document

### Alternative Explanations
- Did position/tokenization tests reveal confounds? → Hypothesis needs **refinement**
- Did spurious correlation tests fire? → Hypothesis might be **wrong**

### Minimal Pair Grid
- Fill in the grid with activation values
- Look for patterns: Is it BOTH elements? Just one? An interaction?
- If "Save my" fires but "Keep your" doesn't, the hypothesis is wrong

### Surprising Predictions
- Did surprising texts fire as predicted? → Strong **confirmation**
- Did they fail? → Hypothesis is **too narrow** or wrong

## Step 5: Verdict

Assign one of:
- **CONFIRMED:** Counterexamples failed, alternatives ruled out, minimal pairs confirm, surprising predictions worked → Increase confidence to 90%+
- **REFINED:** Core holds but needs adjustment, discovered boundary conditions → Update interpretation
- **REFUTED:** Systematic counterexamples, better alternative explanation, minimal pairs contradict → Rewrite with new hypothesis
- **UNCERTAIN:** Mixed results, need more data → Flag for human review

## Step 6: Update Interpretation

Based on verdict, update:
- `output/interpretations/feature$ARGUMENTS/results.json` - add `challenge_results` section
- `output/interpretations/feature$ARGUMENTS/report.md` - append Challenge section

### JSON Schema for challenge_results

> **REQUIRED:** Copy `active_token_idx` → `token_idx` and `all_tokens` from batch_test output for ALL test arrays.

```json
{
  "challenge_results": {
    "counterexamples": [
      {"text": "...", "activation": 0.XXX, "token": "...", "token_idx": N, "all_tokens": [...], "expected": "no fire", "outcome": "..."}
    ],
    "alternative_tests": [
      {"test_type": "...", "text": "...", "activation": 0.XXX, "token": "...", "token_idx": N, "all_tokens": [...], "implication": "..."}
    ],
    "minimal_pairs": {
      "description": "...",
      "grid": [
        {"condition": "...", "text": "...", "activation": 0.XXX, "token": "...", "token_idx": N, "all_tokens": [...]}
      ],
      "conclusion": "..."
    },
    "surprising_predictions": [
      {"text": "...", "rationale": "...", "activation": 0.XXX, "token": "...", "token_idx": N, "all_tokens": [...], "result": "confirmed|refuted"}
    ],
    "verdict": "CONFIRMED|REFINED|REFUTED|UNCERTAIN",
    "post_challenge_confidence": 0.XX
  }
}
```

### Append to Markdown Report:

```markdown
---

# Challenge Results

**Date:** YYYY-MM-DD | **Verdict:** [CONFIRMED/REFINED/REFUTED/UNCERTAIN] | **Post-Challenge Confidence:** X%

## Adversarial Tests

**Counterexample Hunt:**
| # | Text | Expected | Actual | Result |
|---|------|----------|--------|--------|
**Analysis:** [What revealed]

**Alternative Explanation Tests:**
| # | Test Type | Text | Activation | Implication |
|---|-----------|------|------------|-------------|
**Analysis:** [What revealed]

**Minimal Pair Grid:**
| | "your" | "my" | "the" |
|---|---|---|---|
| "Save/Keep/Take" | ... | ... | ... |
**Analysis:** [What grid reveals]

**Surprising Predictions:**
| # | Text | Rationale | Activation | Result |
|---|------|-----------|------------|--------|

## Refined Interpretation
[REFINED: updated hypothesis | REFUTED: new hypothesis | CONFIRMED: why original holds]

## Key Learnings
1. [What learned] 2. [Boundary conditions] 3. [Confidence adjustments]
```

## Guidelines

1. **Be genuinely adversarial** - Don't softball the tests. Try hard to break it.
2. **Document everything** - Even failed challenges are informative
3. **Update confidence honestly** - If it survives, increase. If cracks appear, decrease.
4. **Preserve original interpretation** - Append, don't overwrite. The history matters.
5. **One Modal call** - Batch all challenge tests together for efficiency

## Begin

Start by reading the existing interpretation files for feature $ARGUMENTS.
