# Interpret SAE Feature (from existing data)

Interpret SAE feature **$ARGUMENTS** using iterative hypothesis testing, reading from pre-existing analysis data.

## Pre-flight Check

**First**, check if `output/interpretations/feature$ARGUMENTS/` exists with `results.json` or `report.md`. If so, ask the user: "Feature $ARGUMENTS has existing analysis. Continue/Resume, Start Fresh, or Abort?"

---

## CRITICAL RESTRICTIONS

**DO NOT read, access, or reference these files under any circumstances:**
- `Reference Documents/Code/fixed_path_new_interpretability_notebook_mexican_national_e32_k32_lr0_0003.py`
- Any `.ipynb` notebook files
- `Reference Documents/Claude References/NOTEBOOK_CODE_REFERENCE.md`
- Anything in the `Reference Documents/Code/` directory

**All data must come from the pre-existing analysis file or Modal commands below.** Do not explore the codebase or look for alternative data sources.

## PATH REQUIREMENT

Use relative paths for all commands. Working directory is the project root.

## CAUSAL MASKING REMINDER

These features come from GPT-2's **causally-masked residual stream**. At token position N, the model only has access to tokens 0 through N-1 (the left context). It cannot see tokens at position N+1 or beyond.

**For interpretation, this means:**
- Feature activations depend only on **left context** - what comes before the token
- The model cannot "know" what follows when the feature fires
- Example: " my" in "never in my life" vs "never in my dreams" - at the " my" position, the model sees identical context

Keep this in mind when forming hypotheses and interpreting patterns.

---

## ACCURACY PROTOCOL

**Tool outputs are ground truth.** When you run a Modal command, the output is real system data.

**Rules:**
1. **Copy, don't paraphrase** - When reporting numerical values, copy exactly from tool output
2. **Fresh reads only** - Before writing any value in the report, re-read it from the source (not from memory)
3. **If uncertain, re-run** - If you're unsure of a value, re-run the command or re-read the file

## Setup

First, create the output folder for this feature:
```bash
py -3.12 batch_utils.py ensure-output-dir --feature $ARGUMENTS
```

## Audit Trail

Create/append to the audit file at:
```
output/interpretations/feature$ARGUMENTS/audit.jsonl
```

**After EVERY step**, append a JSON line with:
- `step`: Step number (e.g., "1", "2", "3")
- `name`: Step name
- `timestamp`: Current time in ISO 8601 format
- `action`: Type of action (`read_file`, `hypothesis_generation`, `test_design`, `modal_command`, `evaluation`, `synthesis`)
- `command`: Exact bash command if applicable
- `decision`: What you decided/concluded
- `justification`: 1-2 sentences explaining WHY
- `output_summary`: Key metrics/results from this step

This audit trail is **append-only**. Never delete previous entries.

---

## Your Task

You are an SAE feature interpretation agent. Follow this loop:

### Step 1: Load Existing Data

Read the pre-computed feature data from:
```
feature data/feature_$ARGUMENTS.json
```

**IMPORTANT:** If this file does not exist, stop and tell the user:
> "File `feature data/feature_$ARGUMENTS.json` not found. Run `py -3.12 run_modal_utf8.py analyze_feature_json --feature-idx $ARGUMENTS` first, or use `/interpret` instead."

The file contains:
- **stats**: Activation rate, mean/max values
- **top_tokens**: Most common tokens where feature fires
- **top_activations**: Example contexts with strongest activation
- **ngram_analysis**: Common bigrams, trigrams, 4-grams (helps identify patterns!)

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "1", "name": "Load Existing Data", "timestamp": "<ISO 8601>", "action": "read_file", "input": "feature data/feature_$ARGUMENTS.json", "decision": "<loaded successfully or error>", "justification": "<summary of what the data shows>", "output_summary": {"tokens_scanned": N, "activation_rate": X, "top_trigram": "..."}}
```

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

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "2", "name": "Generate Hypotheses", "timestamp": "<ISO 8601>", "action": "hypothesis_generation", "decision": "Generated N hypotheses", "justification": "<why these hypotheses based on the data>", "output_summary": {"hypotheses": [{"id": 1, "text": "..."}, ...]}}
```

### Step 3: Design Test Examples
For each hypothesis, create:
- **10 POSITIVE examples** (text that SHOULD activate this feature)
- **10 NEGATIVE examples** (text that should NOT activate, testing boundaries)

Critical for negatives: Use similar words in DIFFERENT contexts to test if it's semantic vs lexical.

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "3", "name": "Design Test Examples", "timestamp": "<ISO 8601>", "action": "test_design", "decision": "Designed N test cases", "justification": "<rationale for positive/negative selection>", "output_summary": {"positive_count": 10, "negative_count": 10, "positive_examples": ["..."], "negative_examples": ["..."]}}
```

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

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "4", "name": "Run Batch Tests", "timestamp": "<ISO 8601>", "action": "modal_command", "command": "<exact command>", "decision": "<which hypotheses supported/refuted>", "justification": "<interpretation of results>", "output_summary": {"accuracy": X, "tp": N, "tn": N, "fp": N, "fn": N}}
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

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "4a", "name": "Context Ablation", "timestamp": "<ISO 8601>", "action": "modal_command", "command": "<exact command>", "decision": "Critical token: <token>", "justification": "<what ablation reveals about causal structure>", "output_summary": {"target_token": "...", "critical_token": "...", "cliff_drop": X, "minimum_context": "..."}}
```

### Step 5: Evaluate
- Which hypotheses are supported by the evidence?
- Which are falsified?
- Does the ablation confirm which tokens are causally necessary?
- Are you confident enough to stop, or should you refine and iterate?

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "5", "name": "Evaluate", "timestamp": "<ISO 8601>", "action": "evaluation", "decision": "<final interpretation>", "justification": "<summary of evidence>", "output_summary": {"confidence": X, "label": "...", "supported_hypotheses": [1], "refuted_hypotheses": [2, 3]}}
```

### Step 6: Output

#### Pre-Output Verification

Before writing final outputs, re-read source files and fill in this verification template:

**Re-read `feature data/feature_$ARGUMENTS.json` now.** Copy exact values:

| Field | JSON Path | Value (copy exactly) |
|-------|-----------|---------------------|
| Activation rate | stats.activation_rate | _______ |
| Mean activation | stats.mean_activation | _______ |
| Max activation | stats.max_activation | _______ |
| Tokens scanned | stats.tokens_scanned | _______ |

**Re-read `output/batch_test_$ARGUMENTS.json` now.** Verify test results match what you're reporting.

Only proceed to write outputs after completing this verification.

#### Write Outputs

Write final results to (all in `output/interpretations/feature$ARGUMENTS/`):
- `results.json` (structured data)
- `report.md` (human-readable report)

**results.json schema:**
```json
{
  "feature_idx": $ARGUMENTS,
  "status": "complete",
  "label": "...",
  "category": "...",
  "description": "...",
  "confidence": 0.XX,
  "corpus_stats": {
    "tokens_scanned": N,
    "activation_rate": X,
    "mean_activation": X,
    "max_activation": X
  },
  "top_tokens": [
    {"token": "...", "count": N, "mean_activation": X}
  ],
  "top_activations": [
    {"text": "...", "token": "...", "activation": X, "position": N}
  ],
  "ngram_analysis": {
    "bigrams": [{"ngram": "...", "count": N}],
    "trigrams": [{"ngram": "...", "count": N}],
    "fourgrams": [{"ngram": "...", "count": N}]
  },
  "hypotheses": [
    {"id": 1, "description": "...", "result": "SUPPORTED|REJECTED|PARTIAL"}
  ],
  "test_results": [
    {"text": "...", "activation": 0.XXX, "token": "...", "expected": true, "actual": true}
  ],
  "ablation_results": {
    "text": "...",
    "target_token": "...",
    "critical_token": "...",
    "minimum_context": "..."
  },
  "key_examples": [
    {"context": "...", "token": "...", "activation": 0.XXX}
  ],
  "linguistic_function": "..."
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "6", "name": "Write Output", "timestamp": "<ISO 8601>", "action": "synthesis", "decision": "Final outputs written", "justification": "All phases complete", "output_summary": {"files": ["results.json", "report.md"]}}
```

---

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

**Source:** `output/feature_$ARGUMENTS.json` (pre-existing)

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

## Collect Intermediate Files (REQUIRED)

**Run these commands** to move working files into the feature folder:

```bash
mv output/batch_test_$ARGUMENTS.json output/interpretations/feature$ARGUMENTS/ 2>/dev/null || true
mv output/ablation_$ARGUMENTS_*.json output/interpretations/feature$ARGUMENTS/ 2>/dev/null || true
```

This keeps the feature folder self-contained with all analysis artifacts.

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "7", "name": "Collect Intermediate Files", "timestamp": "<ISO 8601>", "action": "file_move", "command": "mv output/batch_test_$ARGUMENTS.json and ablation files", "decision": "Commands executed", "justification": "Moving intermediate files to feature folder", "output_summary": {"files_moved": ["batch_test_$ARGUMENTS.json", "ablation files if present"], "verified": true}}
```

## Final Outputs

All outputs are in `output/interpretations/feature$ARGUMENTS/`:
- `audit.jsonl` - Step-by-step audit trail (append-only)
- `results.json` - Structured data
- `report.md` - Human-readable report
- `batch_test_$ARGUMENTS.json` - Raw batch test results (intermediate)
- `ablation_$ARGUMENTS_*.json` - Raw ablation results (intermediate)

## Begin
Start with Step 1 now.
