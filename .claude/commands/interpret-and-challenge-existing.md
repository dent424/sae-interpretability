# Interpret and Challenge SAE Feature (from existing data)

Run full interpretation pipeline for feature **$ARGUMENTS**: first interpret, then challenge. Uses pre-existing analysis data.

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
- `step`: Step number (e.g., "1.1", "2.3")
- `name`: Step name
- `timestamp`: Current time in ISO 8601 format
- `action`: Type of action (`read_file`, `hypothesis_generation`, `test_design`, `modal_command`, `challenge_test`, `evaluation`, `synthesis`)
- `command`: Exact bash command if applicable
- `decision`: What you decided/concluded
- `justification`: 1-2 sentences explaining WHY
- `output_summary`: Key metrics/results from this step

This audit trail is **append-only**. Never delete previous entries.

---

## Phase 1: Interpret

Interpret SAE feature **$ARGUMENTS** using iterative hypothesis testing.

### Step 1.1: Load Existing Data

Read the pre-computed feature data from:
```
feature data/feature_$ARGUMENTS.json
```

**IMPORTANT:** If this file does not exist, stop and tell the user:
> "File `feature data/feature_$ARGUMENTS.json` not found. Run `py -3.12 run_modal_utf8.py analyze_feature_json --feature-idx $ARGUMENTS` first."

The file contains:
- **stats**: Activation rate, mean/max values
- **top_tokens**: Most common tokens where feature fires
- **top_activations**: Example contexts with strongest activation
- **ngram_analysis**: Common bigrams, trigrams, 4-grams (helps identify patterns!)

**After this step, initialize the JSON file:**
Write to `output/interpretations/feature$ARGUMENTS/results.json`:
```json
{
  "feature_idx": $ARGUMENTS,
  "status": "in_progress",
  "current_phase": "interpretation",
  "corpus_stats": { ... from analyze output ... },
  "top_tokens": [ ... from analyze output ... ],
  "top_activations": [ ... from analyze output ... ],
  "ngram_analysis": { ... from analyze output ... }
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "1.1", "name": "Load Existing Data", "timestamp": "<ISO 8601>", "action": "read_file", "input": "feature data/feature_$ARGUMENTS.json", "decision": "<loaded successfully>", "justification": "<summary of key patterns in data>", "output_summary": {"tokens_scanned": N, "activation_rate": X, "top_trigram": "..."}}
```

### Step 1.2: Generate Hypotheses

Based on the n-grams and top activations, generate 2-3 hypotheses. Consider:
- Semantic patterns (meaning, concepts)
- Syntactic patterns (grammar, structure)
- Lexical patterns (specific words, n-grams)
- Positional patterns (sentence position)

**CRITICAL - Causal Masking:** GPT-2 uses causal attention. Activations at position N can ONLY depend on tokens 0 to N-1. When a feature fires on token X:
- It can see everything to the LEFT of X
- It CANNOT see anything to the RIGHT of X
- Example: If feature fires on " my" in "never in my life", it cannot see "life"
- Only distinguish patterns by their LEFT context, never by what follows

**After this step, update the JSON:**
```json
{
  ...existing fields...,
  "interpretation_phase": {
    "hypotheses": [
      {"id": 1, "description": "...", "result": "pending"},
      {"id": 2, "description": "...", "result": "pending"},
      {"id": 3, "description": "...", "result": "pending"}
    ]
  }
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "1.2", "name": "Generate Hypotheses", "timestamp": "<ISO 8601>", "action": "hypothesis_generation", "decision": "Generated N hypotheses", "justification": "<why these hypotheses based on data>", "output_summary": {"hypotheses": [{"id": 1, "text": "..."}, ...]}}
```

### Step 1.3: Design and Run Batch Tests

For each hypothesis, create:
- **10 POSITIVE examples** (text that SHOULD activate this feature)
- **10 NEGATIVE examples** (text that should NOT activate, testing boundaries)

Test ALL examples in a single call:
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --texts "positive 1|positive 2|...|negative 1|negative 2|..."
```

**Important:**
- Texts are separated by `|` (pipe character)
- All texts are processed in ONE Modal call (~10x faster)
- Results: `output/batch_test_$ARGUMENTS.json`

**After this step, update the JSON:**
```json
{
  ...existing fields...,
  "interpretation_phase": {
    "hypotheses": [
      {"id": 1, "description": "...", "result": "SUPPORTED|REJECTED|PARTIAL"},
      ...
    ],
    "test_results": [
      {"text": "...", "activation": 0.XXX, "token": "...", "expected": true, "actual": true},
      ...
    ]
  }
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "1.3", "name": "Design and Run Batch Tests", "timestamp": "<ISO 8601>", "action": "modal_command", "command": "<exact batch_test command>", "decision": "<which hypotheses supported/refuted>", "justification": "<interpretation of test results>", "output_summary": {"accuracy": X, "tp": N, "tn": N, "fp": N, "fn": N}}
```

**SAFEGUARD:** If ALL activations are 0.0:
1. Try a corpus context from `top_activations` (e.g., reconstruct "...I'm happy** to** say..." as full text)
2. If corpus context also fails → **STOP.** Flag: "Feature [N] won't activate. Check feature index or pipeline."
3. If corpus works but synthetic fails → redesign test examples based on corpus patterns

### Step 1.4: Context Ablation

For a text where the feature fires, run ablation to find the **causally necessary** context:
```bash
py -3.12 run_modal_utf8.py ablate_context --feature-idx $ARGUMENTS --text "Example text where feature fires strongly."
```

**After this step, update the JSON:**
```json
{
  ...existing fields...,
  "interpretation_phase": {
    ...existing fields...,
    "ablation_results": {
      "text": "...",
      "target_token": "...",
      "original_activation": 0.XXX,
      "critical_token": "...",
      "minimum_context_tokens": N,
      "ablation_sequence": [...]
    }
  }
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "1.4", "name": "Context Ablation", "timestamp": "<ISO 8601>", "action": "modal_command", "command": "<exact ablate_context command>", "decision": "Critical token: <token>", "justification": "<what ablation reveals>", "output_summary": {"target_token": "...", "critical_token": "...", "cliff_drop": X, "minimum_context": "..."}}
```

### Step 1.5: Initial Conclusion

Evaluate:
- Which hypotheses are supported by the evidence?
- Which are falsified?
- Does the ablation confirm which tokens are causally necessary?

**After this step, update the JSON:**
```json
{
  ...existing fields...,
  "interpretation_phase": {
    ...existing fields...,
    "initial_conclusion": "...",
    "initial_confidence": 0.XX,
    "initial_label": "..."
  }
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "1.5", "name": "Initial Conclusion", "timestamp": "<ISO 8601>", "action": "evaluation", "decision": "<initial interpretation>", "justification": "<summary of evidence>", "output_summary": {"confidence": X, "label": "...", "supported_hypotheses": [1], "refuted_hypotheses": [2, 3]}}
```

---

## Phase 2: Challenge

Now switch to **devil's advocate** mode. Your job is to **break** the hypothesis you just created.

### Step 2.1: Counterexample Hunt (5-8 tests)

Create texts that **fit the stated pattern but you suspect WON'T fire**:
- Edge cases the interpretation missed
- Unusual contexts for the same syntactic structure
- Domain transfers
- Archaic/formal/slang variants

Run tests:
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --texts "counterexample1|counterexample2|..."
```

**After this step, update the JSON:**
```json
{
  ...existing fields...,
  "challenge_phase": {
    "counterexamples": [
      {"text": "...", "expected": "no fire", "activation": 0.XXX, "outcome": "..."},
      ...
    ]
  }
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "2.1", "name": "Counterexample Hunt", "timestamp": "<ISO 8601>", "action": "challenge_test", "command": "<exact command>", "decision": "<counterexamples found or not>", "justification": "<what counterexamples reveal>", "output_summary": {"tests_run": N, "surprises": N}}
```

### Step 2.2: Alternative Explanation Tests (5-8 tests)

What ELSE could explain the activations? Test for:
- **Tokenization artifacts**: Does activation depend on how GPT-2 tokenizes, not meaning?
- **Position effects**: Same phrase at sentence start vs middle vs end
- **Frequency confounds**: Is it just detecting common words in common positions?
- **Spurious correlation**: What if it's actually detecting something NEARBY the claimed pattern?

Run tests:
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --texts "position_test1|tokenization_test1|..."
```

**After this step, update the JSON:**
```json
{
  ...existing fields...,
  "challenge_phase": {
    ...existing fields...,
    "alternative_tests": [
      {"test_type": "position", "text": "...", "activation": 0.XXX, "implication": "..."},
      ...
    ]
  }
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "2.2", "name": "Alternative Explanation Tests", "timestamp": "<ISO 8601>", "action": "challenge_test", "command": "<exact batch_test command>", "decision": "<alternative explanations ruled out or not>", "justification": "<what tests reveal>", "output_summary": {"tests_run": N, "position_effect": bool, "tokenization_artifact": bool}}
```

### Step 2.3: Minimal Pair Grid (9-16 tests)

Systematically vary ONE element at a time to isolate whether BOTH elements are necessary or just one.

Run tests:
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --texts "minimal_pair1|minimal_pair2|..."
```

**After this step, update the JSON:**
```json
{
  ...existing fields...,
  "challenge_phase": {
    ...existing fields...,
    "minimal_pairs": {
      "description": "...",
      "grid": [
        {"condition": "...", "text": "...", "activation": 0.XXX},
        ...
      ],
      "conclusion": "..."
    }
  }
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "2.3", "name": "Minimal Pair Grid", "timestamp": "<ISO 8601>", "action": "challenge_test", "command": "<exact batch_test command>", "decision": "<what elements are necessary>", "justification": "<grid analysis>", "output_summary": {"grid_size": "NxM", "key_finding": "...", "necessary_elements": [...]}}
```

### Step 2.4: Surprising Predictions (2-3 tests)

If the hypothesis is TRUE, what **unexpected** text SHOULD activate?

**After this step, update the JSON:**
```json
{
  ...existing fields...,
  "challenge_phase": {
    ...existing fields...,
    "surprising_predictions": [
      {"text": "...", "rationale": "...", "activation": 0.XXX, "result": "confirmed|refuted"},
      ...
    ]
  }
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "2.4", "name": "Surprising Predictions", "timestamp": "<ISO 8601>", "action": "challenge_test", "decision": "<predictions confirmed or refuted>", "justification": "<what this means for hypothesis>", "output_summary": {"tests_run": N, "confirmed": N, "refuted": N}}
```

### Step 2.5: Challenge Verdict

Assign one of: **CONFIRMED**, **REFINED**, **REFUTED**, or **UNCERTAIN**

**After this step, update the JSON:**
```json
{
  ...existing fields...,
  "challenge_phase": {
    ...existing fields...,
    "verdict": "CONFIRMED|REFINED|REFUTED|UNCERTAIN",
    "verdict_justification": "..."
  }
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "2.5", "name": "Challenge Verdict", "timestamp": "<ISO 8601>", "action": "evaluation", "decision": "<VERDICT>", "justification": "<summary of challenge findings>", "output_summary": {"verdict": "...", "pre_challenge_confidence": X, "post_challenge_confidence": Y}}
```

---

## Phase 3: Synthesize Final Report

Now synthesize everything into final outputs.

### Step 3.1: Finalize JSON

#### Pre-Synthesis Verification

Before finalizing outputs, re-read source files and verify all values:

**Re-read `feature data/feature_$ARGUMENTS.json` now.** Copy exact values:

| Field | Value (copy exactly) |
|-------|---------------------|
| stats.activation_rate | _______ |
| stats.mean_activation | _______ |
| stats.max_activation | _______ |
| stats.tokens_scanned | _______ |

**Re-read latest batch_test output.** Verify your test accuracy calculation:
- Total tests: _______
- True positives: _______
- True negatives: _______
- Accuracy: _______ (calculate: (TP+TN)/Total)

**Cross-check:** Do the values in your working notes match these source values? If not, correct them now.

#### Write Final JSON

Update `output/interpretations/feature$ARGUMENTS/results.json` with final fields:
```json
{
  "feature_idx": $ARGUMENTS,
  "status": "complete",
  "label": "...",
  "category": "...",
  "description": "...",
  "confidence": 0.XX,
  "verdict": "CONFIRMED|REFINED|REFUTED|UNCERTAIN",
  "necessary_conditions": ["...", "..."],
  "boundary_conditions": ["...", "..."],
  "does_not_detect": ["...", "..."],
  "corpus_stats": {...},
  "top_tokens": [...],
  "top_activations": [...],
  "ngram_analysis": {...},
  "interpretation_phase": {...},
  "challenge_phase": {...},
  "key_examples": [
    {"context": "...", "token": "...", "activation": 0.XXX, "meaning": "..."},
    ...
  ],
  "executive_summary": "...",
  "linguistic_function": "...",
  "potential_applications": "..."
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "3.1", "name": "Finalize JSON", "timestamp": "<ISO 8601>", "action": "synthesis", "decision": "Final JSON written", "justification": "All phases complete", "output_summary": {"file": "results.json"}}
```

### Step 3.2: Write Markdown Report

Write the final report to `output/interpretations/feature$ARGUMENTS/report.md` with this structure:

```markdown
# Feature [N] Interpretation Report

**Date:** YYYY-MM-DD
**Final Verdict:** [CONFIRMED/REFINED/REFUTED/UNCERTAIN]
**Confidence:** X%

---

## Executive Summary

[2-3 sentence summary of what this feature detects, incorporating learnings from both interpretation AND challenge phases]

**Label:** [Short label, refined based on challenge results]
**Category:** [Category > Subcategory]

---

## The Pattern

### What It Detects
[Clear description of the pattern, refined by challenge testing]

### Necessary Conditions
[Bullet list of what MUST be present for activation, validated by minimal pairs]

### Boundary Conditions
[What edge cases don't fire, discovered during challenge]

### Does NOT Detect
[Common misconceptions or similar-looking patterns that don't activate]

---

## Evidence Summary

### Corpus Statistics
| Metric | Value |
|--------|-------|
| Tokens scanned | ... |
| Activation rate | ... |
| Mean activation | ... |
| Max activation | ... |

### Key Examples
[Top 5 most illustrative examples with activation values - curated from both phases]

| Activation | Token | Context | Why It Fires |
|------------|-------|---------|--------------|
| 0.XXX | '...' | "..." | [explanation] |

### Minimal Pair Evidence
[The most informative minimal pair comparisons from challenge phase]

| Test A | Activation | Test B | Activation | Conclusion |
|--------|------------|--------|------------|------------|
| "..." | 0.XXX | "..." | 0.000 | [what this proves] |

---

## Interpretation Process

### Phase 1: Initial Interpretation

#### Data Gathering
**Source:** `output/feature_$ARGUMENTS.json` (pre-existing)

##### Top Tokens
[Table of top 20 tokens]

##### N-gram Analysis
[Key patterns from bigrams, trigrams, 4-grams]

##### Top Corpus Activations
[Top 10 with full context]

#### Initial Hypotheses
1. [Hypothesis 1]
2. [Hypothesis 2]
3. [Hypothesis 3]

#### Hypothesis Testing
**Command:** [batch_test command with all texts]

| # | Type | Text | Activation | Token | Result |
|---|------|------|------------|-------|--------|
| 1 | POS | ... | ... | ... | Y/N |

#### Ablation Analysis
**Command:** [ablate_context command]

[Ablation results table and interpretation]

#### Initial Conclusion
[What Phase 1 concluded, with confidence level]

---

### Phase 2: Adversarial Challenge

#### Counterexample Hunt
| # | Text | Expected | Actual | Result |
|---|------|----------|--------|--------|
| 1 | ... | No fire | 0.XXX | [outcome] |

**Analysis:** [What counterexamples revealed]

#### Alternative Explanation Tests
| # | Test Type | Text | Activation | Implication |
|---|-----------|------|------------|-------------|
| 1 | Position | ... | ... | ... |

**Analysis:** [Were alternative explanations ruled out?]

#### Minimal Pair Grid
| | Var1 | Var2 | Var3 | Var4 |
|---|---|---|---|---|
| CondA | 0.XX | 0.XX | 0.XX | 0.XX |
| CondB | 0.XX | 0.XX | 0.XX | 0.XX |

**Analysis:** [What the grid reveals]

#### Surprising Predictions
| # | Text | Rationale | Activation | Result |
|---|------|-----------|------------|--------|
| 1 | ... | Should fire because... | 0.XX | Y/N |

#### Challenge Verdict
[CONFIRMED/REFINED/REFUTED/UNCERTAIN with justification]

---

## Synthesis

### How Challenge Changed the Interpretation
[Bullet list of refinements, boundary conditions discovered, confidence adjustments]

### Remaining Uncertainties
[What's still unclear or would benefit from more testing]

### Related Features to Investigate
[If any patterns suggest related features worth exploring]

---

## Conclusion

[Final 2-3 paragraph synthesis: what this feature does, how confident we are, what makes it interesting or useful for interpretability]

**Linguistic Function:** [One sentence describing the linguistic/semantic role]

**Potential Applications:** [How this feature could be used - e.g., detecting certain writing styles, sentiment patterns, etc.]
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "3.2", "name": "Write Report", "timestamp": "<ISO 8601>", "action": "synthesis", "decision": "Final report written", "justification": "Synthesized all evidence", "output_summary": {"file": "report.md"}}
```

### Step 3.3: Collect Intermediate Files (REQUIRED)

**Run these commands** to move working files into the feature folder:

```bash
mv output/batch_test_$ARGUMENTS.json output/interpretations/feature$ARGUMENTS/ 2>/dev/null || true
mv output/ablation_$ARGUMENTS_*.json output/interpretations/feature$ARGUMENTS/ 2>/dev/null || true
```

This keeps the feature folder self-contained with all analysis artifacts.

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "3.3", "name": "Collect Intermediate Files", "timestamp": "<ISO 8601>", "action": "file_move", "command": "mv output/batch_test_$ARGUMENTS.json and ablation files", "decision": "Commands executed", "justification": "Moving intermediate files to feature folder", "output_summary": {"files_moved": ["batch_test_$ARGUMENTS.json", "ablation files if present"], "verified": true}}
```

---

## Final Outputs

All outputs are in `output/interpretations/feature$ARGUMENTS/`:
- `audit.jsonl` - Step-by-step audit trail (append-only)
- `results.json` - Structured data
- `report.md` - Human-readable report
- `batch_test_$ARGUMENTS.json` - Raw batch test results (intermediate)
- `ablation_$ARGUMENTS_*.json` - Raw ablation results (intermediate)

## Begin

Start with Phase 1, Step 1.1 now. **Remember to update the JSON file AND audit log after each step** to ensure all data is captured incrementally.
