# Challenge SAE Feature Interpretation

Challenge and stress-test the interpretation for feature **$ARGUMENTS**.

## Pre-flight Check

**First**, check if `output/interpretations/feature_$ARGUMENTS.json` contains `challenge_results` or `challenge_phase`. If so, ask the user: "Feature $ARGUMENTS already has challenge results. Re-challenge or Abort?"

---

## Your Task

You are a **devil's advocate** for SAE feature interpretations. Your job is to **break** the current hypothesis, not confirm it. A good interpretation should survive adversarial testing.

## Step 1: Load Current Interpretation

Read the existing interpretation:
- `output/interpretations/feature_$ARGUMENTS.json` (structured data)
- `output/interpretations/feature_$ARGUMENTS.md` (full report)

Extract:
1. **The hypothesis**: What pattern does the interpretation claim?
2. **The evidence**: What tests supported it?
3. **Confidence level**: How certain was the original interpretation?

If no interpretation exists, stop and tell the user to run `/interpret $ARGUMENTS` first.

## Step 2: Generate Adversarial Tests

Design tests specifically intended to **FALSIFY** the hypothesis. Think like a skeptic.

### 2a: Counterexample Hunt (5-8 tests)
Create texts that **fit the stated pattern but you suspect WON'T fire**:
- Edge cases the original tests missed
- Unusual contexts for the same syntactic structure
- Domain transfers (if pattern is "Save your money", try "Save your game")
- Archaic/formal/slang variants

### 2b: Alternative Explanation Tests (5-8 tests)
What ELSE could explain the activations? Test for:
- **Tokenization artifacts**: Does activation depend on how GPT-2 tokenizes, not meaning?
- **Position effects**: Same phrase at sentence start vs middle vs end
- **Frequency confounds**: Is it just detecting common words in common positions?
- **Spurious correlation**: What if it's actually detecting something NEARBY the claimed pattern?

### 2c: Minimal Pair Grid (9-16 tests)
Systematically vary ONE element at a time. Create a grid like:

If hypothesis is "Imperative + your":
| | "your" | "my" | "the" | "our" |
|---|---|---|---|---|
| "Save" | test | test | test | test |
| "Keep" | test | test | test | test |
| "Take" | test | test | test | test |
| "Get" | test | test | test | test |

This isolates whether BOTH elements are necessary or just one.

### 2d: Surprising Prediction (2-3 tests)
If the hypothesis is TRUE, what **unexpected** text SHOULD activate?
- Something the original tester probably didn't think of
- Something that sounds weird but fits the pattern
- Test it. If it fires, that's strong evidence FOR the hypothesis.

## Step 3: Run All Challenge Tests

Batch all tests in a single call:
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --texts "counterexample1|counterexample2|...|grid_test1|grid_test2|..."
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

Assign one of these outcomes:

### CONFIRMED
- Counterexamples failed to break it
- Alternative explanations ruled out
- Minimal pairs show the claimed pattern
- Surprising predictions worked
→ **Increase confidence** to 90%+

### REFINED
- Core hypothesis holds but needs adjustment
- Discovered boundary conditions
- Found additional sub-patterns
→ **Update the interpretation** with refinements

### REFUTED
- Found systematic counterexamples
- Alternative explanation fits better
- Minimal pairs contradict the claim
→ **Rewrite the interpretation** with new hypothesis

### UNCERTAIN
- Mixed results, no clear pattern
- Need more data or different approach
→ **Flag for human review**

## Step 6: Update Interpretation

Based on verdict, update:
- `output/interpretations/feature_$ARGUMENTS.json` - add `challenge_results` section
- `output/interpretations/feature_$ARGUMENTS.md` - append Challenge section

### Append to Markdown Report:

```markdown
---

# Challenge Results

**Date:** YYYY-MM-DD
**Verdict:** [CONFIRMED/REFINED/REFUTED/UNCERTAIN]
**Post-Challenge Confidence:** X%

## Adversarial Tests

### Counterexample Hunt
| # | Text | Expected | Actual | Result |
|---|------|----------|--------|--------|
| 1 | ... | No fire | 0.000 | As expected |
| 2 | ... | No fire | 0.085 | SURPRISE - fires! |

**Analysis:** [What counterexamples reveal]

### Alternative Explanation Tests
| # | Test Type | Text | Activation | Implication |
|---|-----------|------|------------|-------------|
| 1 | Position | ... | ... | ... |
| 2 | Tokenization | ... | ... | ... |

**Analysis:** [What alternative tests reveal]

### Minimal Pair Grid
| | "your" | "my" | "the" |
|---|---|---|---|
| "Save" | 0.18 | 0.00 | 0.00 |
| "Keep" | 0.11 | 0.00 | 0.00 |
| "Take" | 0.00 | 0.00 | 0.00 |

**Analysis:** [What the grid reveals about necessary conditions]

### Surprising Predictions
| # | Text | Rationale | Activation | Result |
|---|------|-----------|------------|--------|
| 1 | ... | Should fire because... | 0.09 | CONFIRMED |

## Refined Interpretation

[If REFINED: State the updated hypothesis]
[If REFUTED: State the new hypothesis]
[If CONFIRMED: State why original holds]

## Key Learnings

1. [What we learned from challenging]
2. [Boundary conditions discovered]
3. [Confidence adjustments and why]
```

## Guidelines

1. **Be genuinely adversarial** - Don't softball the tests. Try hard to break it.
2. **Document everything** - Even failed challenges are informative
3. **Update confidence honestly** - If it survives, increase. If cracks appear, decrease.
4. **Preserve original interpretation** - Append, don't overwrite. The history matters.
5. **One Modal call** - Batch all challenge tests together for efficiency

## Begin

Start by reading the existing interpretation files for feature $ARGUMENTS.
