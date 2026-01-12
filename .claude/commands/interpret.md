# Interpret SAE Feature

Interpret SAE feature **$ARGUMENTS** using iterative hypothesis testing.

## Pre-flight Check

**First**, check if `output/interpretations/feature_$ARGUMENTS.json` or `feature_$ARGUMENTS.md` exists. If so, ask the user: "Feature $ARGUMENTS has existing analysis. Continue/Resume, Start Fresh, or Abort?"

---

## CRITICAL RESTRICTIONS

**DO NOT read, access, or reference these files under any circumstances:**
- `Reference Documents/Code/fixed_path_new_interpretability_notebook_mexican_national_e32_k32_lr0_0003.py`
- Any `.ipynb` notebook files
- `Reference Documents/Claude References/NOTEBOOK_CODE_REFERENCE.md`
- Anything in the `Reference Documents/Code/` directory

**All data must come from the Modal commands below.** Do not explore the codebase or look for alternative data sources.

## CAUSAL MASKING REMINDER

These features come from GPT-2's **causally-masked residual stream**. At token position N, the model only has access to tokens 0 through N-1 (the left context). It cannot see tokens at position N+1 or beyond.

**For interpretation, this means:**
- Feature activations depend only on **left context** - what comes before the token
- The model cannot "know" what follows when the feature fires
- Example: " my" in "never in my life" vs "never in my dreams" - at the " my" position, the model sees identical context

Keep this in mind when forming hypotheses and interpreting patterns.

---

## Your Task

You are an SAE feature interpretation agent. Follow this loop:

### Step 1: Gather Data + N-gram Analysis
Run this command to get feature data (now includes automatic n-gram pattern detection):
```bash
py -3.12 run_modal_utf8.py analyze_feature_json --feature-idx $ARGUMENTS
```
Read the output from `output/feature_$ARGUMENTS.json`

The output now includes:
- **stats**: Activation rate, mean/max values
- **top_tokens**: Most common tokens where feature fires
- **top_activations**: Example contexts with strongest activation
- **ngram_analysis**: Common bigrams, trigrams, 4-grams (helps identify patterns!)

### Step 2: Generate Hypotheses
Based on the n-grams and top activations, generate 2-3 hypotheses. The n-gram analysis often reveals the pattern directly. Consider:
- Semantic patterns (meaning, concepts)
- Syntactic patterns (grammar, structure)
- Lexical patterns (specific words, n-grams)
- Positional patterns (sentence position)

**CRITICAL - Causal Masking:** GPT-2 uses causal attention. Activations at position N can ONLY depend on tokens 0 to N-1. When a feature fires on token X:
- It can see everything to the LEFT of X
- It CANNOT see anything to the RIGHT of X
- Example: If feature fires on " my" in "never in my life", it cannot see "life"
- Therefore "never in my life" and "never in my dreams" are IDENTICAL patterns from the model's perspective at the firing position
- Only distinguish patterns by their LEFT context, never by what follows

### Step 3: Design Test Examples
For each hypothesis, create:
- **10 POSITIVE examples** (text that SHOULD activate this feature)
- **10 NEGATIVE examples** (text that should NOT activate, testing boundaries)

Critical for negatives: Use similar words in DIFFERENT contexts to test if it's semantic vs lexical.

### Step 4: Run Batch Tests
Test ALL examples in a single call:
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --texts "positive 1|positive 2|...|negative 1|negative 2|..."
```

**Important:**
- Texts are separated by `|` (pipe character)
- All texts are processed in ONE Modal call (~10x faster)
- Results: `output/batch_test_$ARGUMENTS.json`

Output shows visual activation bars:
```
[+] 0.220 |██████████████████████████████| @ ' my'    <- ACTIVATED
    I have never in my life...

[-] 0.000 |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|            <- NOT activated
    I live in my house...
```

**SAFEGUARD:** If ALL activations are 0.0:
1. Try a corpus context from `top_activations` (e.g., reconstruct "...I'm happy** to** say..." as full text)
2. If corpus context also fails → **STOP.** Flag: "Feature [N] won't activate. Check feature index or pipeline."
3. If corpus works but synthetic fails → redesign test examples based on corpus patterns

### Step 4a: Context Ablation (Recommended)
For a text where the feature fires, run ablation to find the **causally necessary** context:
```bash
py -3.12 run_modal_utf8.py ablate_context --feature-idx $ARGUMENTS --text "I have never in my life tasted such amazing tacos."
```

This progressively removes left context to show exactly which tokens matter:
```
Depth  Left Context                   Activation
------ ------------------------------ ----------
  0    "I have never in my"           0.220  ████████████████████████
  1    "have never in my"             0.218  ████████████████████████
  2    "never in my"                  0.215  ███████████████████████
  3    "in my"                        0.089  ████████  <- CLIFF
  4    "my"                           0.012  █

Critical token: "never" (removing it drops activation by 58%)
```

### Step 5: Evaluate
- Which hypotheses are supported by the evidence?
- Which are falsified?
- Does the ablation confirm which tokens are causally necessary?
- Are you confident enough to stop, or should you refine and iterate?

### Step 6: Output
Write final results to:
- `output/interpretations/feature_$ARGUMENTS.json` (structured data)
- `output/interpretations/feature_$ARGUMENTS.md` (human-readable report)

**IMPORTANT: The markdown report must document EVERYTHING for full transparency.** Use this structure:

```markdown
# Feature [N] Interpretation Report

**Date:** YYYY-MM-DD
**Iterations:** N
**Final Confidence:** X%

---

## Final Interpretation

[1-2 sentence summary of what the feature detects]

**Label:** [Short label]
**Category:** [Category > Subcategory]

> **Causal Masking Note:** [Always include this reminder about left-context only]

---

# Interpretation Process

## Step 1: Initial Data Gathering

**Command:**
[exact command used]

### Corpus Statistics
[Table with tokens scanned, activation rate, mean, max]

### Top Tokens by Count
[Full table of top 20 tokens with count and mean activation]

### Top 10 Corpus Activations
[Table showing rank, activation, token, and full context for each]

---

## Step 2: N-gram Analysis

### Bigrams (2-grams)
[Full table from JSON output]

### Trigrams (3-grams)
[Full table from JSON output]

### 4-grams
[Full table from JSON output]

---

## Step 3: Hypothesis Generation

[List each hypothesis with description]

---

## Step 4: Test Example Design

### Positive Examples (Expected to Activate)
[Table with #, Text, Rationale for ALL 10 positive examples]

### Negative Examples (Expected NOT to Activate)
[Table with #, Text, Rationale for ALL 10 negative examples]

---

## Step 5: Batch Test Execution

**Command:**
[exact command used including all test texts]

### Results
[Table with #, Type (POS/NEG), Activation, Token, Text for ALL 20 tests]

### Test Summary
[Table with total tests, TP, TN, FP, FN, accuracy]

---

## Step 6: Context Ablation Analysis

**Command:**
[exact command used]

**Target:** [Token and position]

### Ablation Results
[Table showing Depth, Tokens Removed, Left Context, Activation, Status for each step]

### Ablation Analysis
[Table with cliff depth, cliff drop, critical token, minimum context]

**Interpretation:** [1-2 sentences explaining what ablation reveals]

---

## Step 7: Hypothesis Evaluation

[Table with Hypothesis, Result (SUPPORTED/PARTIAL/REFUTED), Evidence]

---

## Final Pattern Summary

### Pattern 1: [name] (Strength range)
[Table showing Left Context, Fires On, Activation for examples]

### Pattern 2: [name] (if applicable)
[Same format]

### Does NOT Fire On
[Bullet list of negative conditions]

---

## Conclusion

[Summary paragraph with key findings numbered]

**Linguistic function:** [What this pattern means in natural language]
```

This format ensures:
1. All data is preserved (top tokens, n-grams, corpus activations)
2. All test examples are documented with rationale
3. All test results are shown with exact activations
4. The exact commands are recorded for reproducibility
5. The reasoning process is transparent

## Iteration
Default: Run up to 3 iterations. Stop early if confident.

## Tips for Effective Testing
1. **Check n-grams first**: The top trigrams often reveal the pattern directly
2. **Use ablation**: It tells you which context is causally necessary vs just correlated
3. **Minimal pairs**: Test "never in my life" vs "always in my life" to isolate key elements
4. **Batch strategically**: All 20 tests fit easily in one batch_test call
5. **Remember causal masking**: Only LEFT context matters at the firing position. Don't create separate pattern categories based on what comes AFTER the token where the feature fires.

## Advanced: Full Analysis with Ablation
For deep analysis, run with automatic ablation on top activations:
```bash
py -3.12 run_modal_utf8.py analyze_feature_json --feature-idx $ARGUMENTS --run-ablation
```

## Collect Intermediate Files

After writing outputs, move working files into the interpretations folder:

```bash
mv output/batch_test_$ARGUMENTS.json output/interpretations/ 2>/dev/null || true
mv output/ablation_$ARGUMENTS_*.json output/interpretations/ 2>/dev/null || true
```

This preserves raw test results for auditing while keeping the main `output/` folder clean.

## Begin
Start with Step 1 now.
