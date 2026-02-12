# Interpret SAE Feature

Interpret SAE feature **$ARGUMENTS** using iterative hypothesis testing.

## Pre-flight Check

**First**, check if `output/interpretations/feature$ARGUMENTS/results.json` or `report.md` exists. If so, ask the user: "Feature $ARGUMENTS has existing analysis. Continue/Resume, Start Fresh, or Abort?"

---

## CRITICAL RESTRICTIONS

**DO NOT read, access, or reference these files under any circumstances:**
- `Reference Documents/Code/fixed_path_new_interpretability_notebook_mexican_national_e32_k32_lr0_0003.py`
- Any `.ipynb` notebook files
- `Reference Documents/Claude References/NOTEBOOK_CODE_REFERENCE.md`
- Anything in the `Reference Documents/Code/` directory

**All data must come from the Modal commands below.** Do not explore the codebase or look for alternative data sources.

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
Based on the n-grams and top activations, generate exactly 3 hypotheses. The n-gram analysis often reveals the pattern directly. Consider:
- Semantic patterns (meaning, concepts)
- Syntactic patterns (grammar, structure)
- Lexical patterns (specific words, n-grams)
- Positional patterns (sentence position)
- Structural patterns (formatting, lists, discourse markers)

**CRITICAL - Causal Masking:** GPT-2 uses causal attention. Activations at position N can ONLY depend on tokens 0 to N-1. When a feature fires on token X:
- It can see everything to the LEFT of X
- It CANNOT see anything to the RIGHT of X
- Example: If feature fires on " my" in "never in my life", it cannot see "life"
- Therefore "never in my life" and "never in my dreams" are IDENTICAL patterns from the model's perspective at the firing position
- Only distinguish patterns by their LEFT context, never by what follows

**ALL hypotheses must account for causal masking.** Nothing that comes after an active token can matter for that activation. (Exception: if the feature fires on multiple tokens in sequence, later active tokens may provide information about the pattern.)

### Step 3: Design Hypothesis-Discriminating Tests

Design THREE test categories:
1. **Baseline (3-5):** All hypotheses predict FIRE. Confirms feature works. If these fail, check feature/hypotheses.
2. **Boundary (3-5):** All hypotheses predict NO FIRE. Confirms boundaries. If these fire, hypotheses too narrow.
3. **Discriminating (8-12) ← MOST IMPORTANT:** Hypotheses DISAGREE. For each hypothesis pair, create 2-3 tests where one predicts FIRE and other predicts NO FIRE.

**Document predictions BEFORE running:**
| Text | H1 Predicts | H2 Predicts | H3 Predicts | Actual | Supports |
|------|-------------|-------------|-------------|--------|----------|
| "..." | fire | no fire | no fire | ? | ? |

### Step 4: Run Batch Tests
Test ALL examples in a single call:
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --output-dir output/interpretations/feature$ARGUMENTS --fresh --texts "baseline1|baseline2|...|boundary1|...|discriminating1|..."
```

**Important:**
- Texts are separated by `|` (pipe character)
- All texts are processed in ONE Modal call (~10x faster)
- Results saved to: `output/interpretations/feature$ARGUMENTS/batch_test_$ARGUMENTS.json`
- The `--fresh` flag ensures a clean start; subsequent batch_test calls will append

> **Field Mapping:** Copy from batch_test output: `max_activation` → `activation`, `active_token` → `token`, `active_token_idx` → `token_idx`, `all_tokens` → `all_tokens`. The `token_idx` and `all_tokens` fields are REQUIRED in results.json.

> **NEVER TRUNCATE:** Always copy the complete `all_tokens` array from batch_test output. Do NOT abbreviate with `["truncated"]` or similar - the full token list is required for reproducibility.

**SAFEGUARD:** If ALL activations are 0.0:
1. Try a corpus context from `top_activations` (e.g., reconstruct "...I'm happy** to** say..." as full text)
2. If corpus context also fails → **STOP.** Flag: "Feature [N] won't activate. Check feature index or pipeline."
3. If corpus works but synthetic fails → redesign test examples based on corpus patterns

### Step 4a: Context Ablation (Recommended)
For a text where the feature fires, run ablation to find the **causally necessary** context:
```bash
py -3.12 run_modal_utf8.py ablate_context --feature-idx $ARGUMENTS --output-dir output/interpretations/feature$ARGUMENTS --text "I have never in my life tasted such amazing tacos."
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

### Step 5: Evaluate Hypothesis Support

#### Score Each Hypothesis

For **discriminating tests only**, count how well each hypothesis predicted:
- **Supported**: Hypothesis predicted correctly (predicted fire and fired, OR predicted no fire and didn't fire)
- **Refuted**: Hypothesis predicted incorrectly

| Hypothesis | Supported | Refuted | Score |
|------------|-----------|---------|-------|
| H1 | 6 | 2 | 6/8 = 75% |
| H2 | 3 | 5 | 3/8 = 38% |
| H3 | 2 | 6 | 2/8 = 25% |

#### Decision Rules
- **Clear Winner:** >70% accuracy AND >2x runner-up → Select and proceed to output
- **Mixed Evidence:** Scores within 20% → Design more tests or merge hypotheses
- **No Winner:** All <50% → Review what actually fired, generate new hypotheses, return to Step 2

State: "H[N] selected as final interpretation with X/Y discriminating test accuracy because [justification]"

### Step 6: Output
Write final results to:
- `output/interpretations/feature$ARGUMENTS/results.json` (structured data)
- `output/interpretations/feature$ARGUMENTS/report.md` (human-readable report)

**REQUIRED:** Verify results.json includes ALL required fields (feature_idx, status, label, category, description, confidence, corpus_stats, top_tokens, top_activations, ngram_analysis, hypotheses, test_results, key_examples) before proceeding. Do not skip any fields.

**IMPORTANT: The markdown report must document EVERYTHING for full transparency.** Use this structure:

```markdown
# Feature [N] Interpretation Report

**Date:** YYYY-MM-DD | **Iterations:** N | **Confidence:** X%

## Final Interpretation
[1-2 sentence summary] **Label:** [Short label] | **Category:** [Category > Subcategory]
> **Causal Masking Note:** [Reminder about left-context only]

---

# Interpretation Process

## Step 1: Data Gathering
**Command:** [exact command]
### Corpus Statistics
[Table: tokens scanned, activation rate, mean, max]
### Top Tokens by Count
[Table: top 20 tokens with count and mean activation]
### Top 10 Corpus Activations
[Table: rank, activation, token, full context]

## Step 2: N-gram Analysis
**Bigrams:** [Table] | **Trigrams:** [Table] | **4-grams:** [Table]

## Step 3: Hypothesis Generation
[List each hypothesis]

## Step 4: Test Design
**Baseline (all predict fire):** [Table: #, Text, Rationale]
**Boundary (all predict no fire):** [Table: #, Text, Rationale]
**Discriminating (disagree):** [Table: #, Text, H1/H2/H3 Predicts, Rationale]

## Step 5: Batch Test Results
**Command:** [exact command with all texts]

| Type | # | Text | Activation | Token | H1 | H2 | H3 | Supports |
|------|---|------|------------|-------|----|----|---------|
[All test results in single table]

### Hypothesis Scoring
| Hypothesis | Supported | Refuted | Accuracy |
|------------|-----------|---------|----------|
**Winner:** H[N] with X% accuracy

## Step 6: Ablation Analysis
**Command:** [exact command] | **Target:** [Token]
[Table: Depth, Left Context, Activation, Status]
**Critical token:** [token] | **Minimum context:** [context]

## Step 7: Final Evaluation
[Table: Hypothesis, Result (SUPPORTED/PARTIAL/REFUTED), Evidence]

## Pattern Summary
**Pattern 1:** [name] - [Table: Left Context, Fires On, Activation]
**Does NOT Fire On:** [Bullet list]

## Conclusion
[Summary paragraph] **Linguistic function:** [description]
```

## Iteration
Default: Run up to 3 iterations. Stop early if confident.

## Tips
1. **N-grams first**: Top trigrams often reveal the pattern directly
2. **Use ablation**: Distinguishes causally necessary vs correlated context
3. **Document predictions BEFORE running**: Commit to what each hypothesis predicts
4. **Remember causal masking**: Only LEFT context matters at firing position

## Advanced: Full Analysis with Ablation
For deep analysis, run with automatic ablation on top activations:
```bash
py -3.12 run_modal_utf8.py analyze_feature_json --feature-idx $ARGUMENTS --run-ablation
```

## Collect Intermediate Files

After writing outputs, move working files into the interpretations folder:

```bash
mv output/batch_test_$ARGUMENTS.json output/interpretations/feature$ARGUMENTS/ 2>/dev/null || true
mv output/ablation_$ARGUMENTS_*.json output/interpretations/feature$ARGUMENTS/ 2>/dev/null || true
```

This preserves raw test results for auditing while keeping the main `output/` folder clean.

## Begin
Start with Step 1 now.
