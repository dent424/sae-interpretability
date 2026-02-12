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

## TOOL RULES (prevents permission prompts)

- **Read files** with the Read tool. NEVER use `cat`, `head`, `tail`, `type`, or any Bash command to read files.
- **Write/edit files** with the Write or Edit tool. NEVER use `echo`, Bash redirection, or heredocs.
- **Search files** with the Grep or Glob tool. NEVER use `grep`, `find`, `ls`, or `dir` via Bash.
- **Do all computation** (JSON parsing, metric calculation, hypothesis scoring) in your own reasoning. NEVER write or run Python scripts.
- **Only these Bash commands are approved:**
  - `py -3.12 run_modal_utf8.py <args>` (batch_test, ablate_context)
  - `py -3.12 batch_utils.py <args>` (ensure-output-dir, timestamp)
  - `mv output/batch_test_<ID>.json <dest>` and `mv output/ablation_<ID>_*.json <dest>`
- Any other Bash command will trigger a permission prompt and block the pipeline.

## PATH REQUIREMENT

Use relative paths for all commands. Working directory is the project root.

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
- `timestamp`: Current time in ISO 8601 format (see Timestamp section below)
- `action`: Type of action (`read_file`, `hypothesis_generation`, `test_design`, `modal_command`, `challenge_test`, `evaluation`, `synthesis`)
- `command`: Exact bash command if applicable
- `decision`: What you decided/concluded
- `justification`: 1-2 sentences explaining WHY
- `output_summary`: Key metrics/results from this step

This audit trail is **append-only**. Never delete previous entries.

### Timestamps

**Get real timestamps** using the platform-agnostic command:
```bash
py -3.12 batch_utils.py timestamp
```
This outputs an ISO 8601 UTC timestamp (e.g., `2026-01-19T19:57:03Z`). Run this command before each audit entry and use the actual output - do NOT use placeholder timestamps.

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

Based on the n-grams and top activations, generate exactly 3 hypotheses. Consider:
- Semantic patterns (meaning, concepts)
- Syntactic patterns (grammar, structure)
- Lexical patterns (specific words, n-grams)
- Positional patterns (sentence position)
- Structural patterns (formatting, lists, discourse markers)

**CRITICAL - Causal Masking:** GPT-2 uses causal attention. Activations at position N can ONLY depend on tokens 0 to N-1. When a feature fires on token X:
- It can see everything to the LEFT of X
- It CANNOT see anything to the RIGHT of X
- Example: If feature fires on " my" in "never in my life", it cannot see "life"
- Only distinguish patterns by their LEFT context, never by what follows

**ALL hypotheses must account for causal masking.** Nothing that comes after an active token can matter for that activation. (Exception: if the feature fires on multiple tokens in sequence, later active tokens may provide information about the pattern.)

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

### Step 1.3: Design Hypothesis-Discriminating Tests

Design THREE test categories:
1. **Baseline (3-5):** All hypotheses predict FIRE. Confirms feature works. If these fail, check feature/hypotheses.
2. **Boundary (3-5):** All hypotheses predict NO FIRE. Confirms boundaries. If these fire, hypotheses too narrow.
3. **Discriminating (8-12) ← MOST IMPORTANT:** Hypotheses DISAGREE. For each hypothesis pair, create 2-3 tests where one predicts FIRE and other predicts NO FIRE.

**Document predictions BEFORE running:**
| Text | H1 Predicts | H2 Predicts | H3 Predicts | Actual | Supports |
|------|-------------|-------------|-------------|--------|----------|
| "..." | fire | no fire | no fire | ? | ? |

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

**After this step, update the JSON:**

> **REQUIRED:** Copy `active_token_idx` → `token_idx` and `all_tokens` from batch_test output.

```json
{
  ...existing fields...,
  "interpretation_phase": {
    "hypotheses": [...],
    "test_design": {
      "baseline_tests": [
        {"text": "...", "all_hypotheses_predict": "fire"}
      ],
      "boundary_tests": [
        {"text": "...", "all_hypotheses_predict": "no fire"}
      ],
      "discriminating_tests": [
        {
          "text": "...",
          "predictions": {"H1": "fire", "H2": "no fire", "H3": "no fire"},
          "discriminates_between": ["H1", "H2"]
        }
      ]
    },
    "test_results": [
      {"text": "...", "activation": 0.XXX, "token": "...", "token_idx": N, "all_tokens": [...],
       "test_type": "baseline|boundary|discriminating",
       "predictions": {"H1": "fire", "H2": "no fire"}, "supports": "H1"}
    ]
  }
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "1.3", "name": "Design and Run Discriminating Tests", "timestamp": "<ISO 8601>", "action": "modal_command", "command": "<exact batch_test command>", "decision": "<test categories designed>", "justification": "<how tests discriminate between hypotheses>", "output_summary": {"baseline_count": N, "boundary_count": N, "discriminating_count": N}}
```

**SAFEGUARD:** If ALL activations are 0.0:
1. Try a corpus context from `top_activations` (e.g., reconstruct "...I'm happy** to** say..." as full text)
2. If corpus context also fails → **STOP.** Flag: "Feature [N] won't activate. Check feature index or pipeline."
3. If corpus works but synthetic fails → redesign test examples based on corpus patterns

### Step 1.4: Context Ablation

For a text where the feature fires, run ablation to find the **causally necessary** context:
```bash
py -3.12 run_modal_utf8.py ablate_context --feature-idx $ARGUMENTS --output-dir output/interpretations/feature$ARGUMENTS --text "Example text where feature fires strongly."
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

### Step 1.5: Evaluate Hypothesis Support

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
- **Clear Winner:** >70% accuracy AND >2x runner-up → Select and proceed to Phase 2
- **Mixed Evidence:** Scores within 20% → Design more tests or merge hypotheses
- **No Winner:** All <50% → Review what actually fired, generate new hypotheses, return to Step 1.2

State: "H[N] selected as leading hypothesis with X/Y discriminating test accuracy because [justification]"

**After this step, update the JSON:**
```json
{
  ...existing fields...,
  "interpretation_phase": {
    ...existing fields...,
    "hypothesis_scores": [
      {"id": 1, "supported": N, "refuted": N, "score": "N/M", "accuracy": 0.XX},
      ...
    ],
    "winner": {"id": N, "accuracy": 0.XX, "justification": "..."},
    "initial_conclusion": "...",
    "initial_confidence": 0.XX,
    "initial_label": "..."
  }
}
```

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "1.5", "name": "Evaluate Hypothesis Support", "timestamp": "<ISO 8601>", "action": "evaluation", "decision": "H[N] selected as winner with X% accuracy", "justification": "<why this hypothesis won>", "output_summary": {"winner": "H1", "winner_accuracy": 0.XX, "runner_up": "H2", "runner_up_accuracy": 0.XX}}
```

---

## Phase 2: Challenge

Now switch to **devil's advocate** mode. Your job is to **break** the winning hypothesis from Phase 1.

### Step 2.1: Counterexample Hunt (5-8 tests)

Create texts that **fit the stated pattern but you suspect WON'T fire**:
- Edge cases the interpretation missed
- Unusual contexts for the same syntactic structure
- Domain transfers
- Archaic/formal/slang variants

Run tests (appends to existing batch_test file):
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --output-dir output/interpretations/feature$ARGUMENTS --texts "counterexample1|counterexample2|..."
```

**After this step, update the JSON:**

> **REQUIRED:** Copy `active_token_idx` → `token_idx` and `all_tokens` from batch_test output.

```json
{
  ...existing fields...,
  "challenge_phase": {
    "counterexamples": [
      {"text": "...", "activation": 0.XXX, "token": "...", "token_idx": N, "all_tokens": [...], "expected": "no fire", "outcome": "..."},
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

Run tests (appends to existing batch_test file):
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --output-dir output/interpretations/feature$ARGUMENTS --texts "position_test1|tokenization_test1|..."
```

**After this step, update the JSON:**

> **REQUIRED:** Copy `active_token_idx` → `token_idx` and `all_tokens` from batch_test output.

```json
{
  ...existing fields...,
  "challenge_phase": {
    ...existing fields...,
    "alternative_tests": [
      {"test_type": "...", "text": "...", "activation": 0.XXX, "token": "...", "token_idx": N, "all_tokens": [...], "implication": "..."},
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

Run tests (appends to existing batch_test file):
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --output-dir output/interpretations/feature$ARGUMENTS --texts "minimal_pair1|minimal_pair2|..."
```

**After this step, update the JSON:**

> **REQUIRED:** Copy `active_token_idx` → `token_idx` and `all_tokens` from batch_test output.

```json
{
  ...existing fields...,
  "challenge_phase": {
    ...existing fields...,
    "minimal_pairs": {
      "description": "...",
      "grid": [
        {"condition": "...", "text": "...", "activation": 0.XXX, "token": "...", "token_idx": N, "all_tokens": [...]},
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

> **REQUIRED:** Copy `active_token_idx` → `token_idx` and `all_tokens` from batch_test output.

```json
{
  ...existing fields...,
  "challenge_phase": {
    ...existing fields...,
    "surprising_predictions": [
      {"text": "...", "rationale": "...", "activation": 0.XXX, "token": "...", "token_idx": N, "all_tokens": [...], "result": "confirmed|refuted"},
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

**Re-read latest batch_test output.** Verify your hypothesis scoring:
- Total discriminating tests: _______
- Winner hypothesis: _______
- Winner accuracy: _______

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

**REQUIRED:** Verify results.json includes ALL fields shown in the schema above before proceeding. Do not skip any fields.

**Audit this step:** Append to `feature$ARGUMENTS/audit.jsonl`:
```json
{"step": "3.1", "name": "Finalize JSON", "timestamp": "<ISO 8601>", "action": "synthesis", "decision": "Final JSON written", "justification": "All phases complete", "output_summary": {"file": "results.json"}}
```

### Step 3.2: Write Markdown Report

Write the final report to `output/interpretations/feature$ARGUMENTS/report.md` with this structure:

```markdown
# Feature [N] Interpretation Report

**Date:** YYYY-MM-DD | **Verdict:** [CONFIRMED/REFINED/REFUTED/UNCERTAIN] | **Confidence:** X%

## Executive Summary
[2-3 sentence summary incorporating learnings from both phases]
**Label:** [Short label] | **Category:** [Category > Subcategory]

## The Pattern
**What It Detects:** [Description refined by challenge testing]
**Necessary Conditions:** [Bullet list validated by minimal pairs]
**Boundary Conditions:** [Edge cases that don't fire]
**Does NOT Detect:** [Similar patterns that don't activate]

---

## Evidence Summary

### Corpus Statistics
| Metric | Value |
|--------|-------|
| Tokens scanned / Activation rate / Mean / Max | ... |

### Key Examples (Top 5)
| Activation | Token | Context | Why It Fires |
|------------|-------|---------|--------------|

### Hypothesis Discrimination
| Hypothesis | Description | Accuracy | Result |
|------------|-------------|----------|--------|

### Minimal Pair Evidence
| Test A | Activation | Test B | Activation | Conclusion |
|--------|------------|--------|------------|------------|

---

## Interpretation Process

### Phase 1: Initial Interpretation

**Data Source:** `output/feature_$ARGUMENTS.json` (pre-existing)
**Top Tokens:** [Table of top 20] | **N-gram Analysis:** [Key patterns] | **Top Corpus Activations:** [Top 10]

**Initial Hypotheses:** 1. [...] 2. [...] 3. [...]

**Batch Test Command:** [command with all texts]

**ALL Test Results** (include every baseline, boundary, and discriminating test - do not summarize):
| Type | # | Text | Activation | Token | H1 | H2 | H3 | Supports |
|------|---|------|------------|-------|----|----|----|----------|
| baseline | 1 | "..." | 0.XXX | ... | fire | fire | fire | all |
| baseline | 2 | "..." | 0.XXX | ... | fire | fire | fire | all |
| boundary | 1 | "..." | 0.000 | - | no fire | no fire | no fire | all |
... (continue for ALL tests)

**Hypothesis Scoring:** | Hypothesis | Supported | Refuted | Accuracy |
**Winner:** H[N] with X% accuracy

**Ablation:** [Command] | [Results table] | [Interpretation]

**Initial Conclusion:** [What Phase 1 concluded]

---

### Phase 2: Adversarial Challenge

**Counterexample Hunt** (ALL tests):
| # | Text | Expected | Actual Activation | Token | Result |
|---|------|----------|-------------------|-------|--------|
| 1 | "..." | no fire | 0.XXX | ... | ... |
... (ALL counterexamples)
**Analysis:** [What revealed]

**Alternative Explanation Tests** (ALL tests):
| # | Test Type | Text | Activation | Token | Implication |
|---|-----------|------|------------|-------|-------------|
| 1 | ... | "..." | 0.XXX | ... | ... |
... (ALL alternative tests)
**Analysis:** [Alternative explanations ruled out?]

**Minimal Pair Grid** (ALL pairs):
| # | Condition | Text | Activation | Token |
|---|-----------|------|------------|-------|
| 1 | ... | "..." | 0.XXX | ... |
... (ALL minimal pair tests from grid)
**Grid Analysis:** [What the full grid reveals]

**Surprising Predictions** (ALL tests):
| # | Text | Rationale | Activation | Token | Result |
|---|------|-----------|------------|-------|--------|
| 1 | "..." | ... | 0.XXX | ... | ... |
... (ALL surprising prediction tests)

**Challenge Verdict:** [CONFIRMED/REFINED/REFUTED/UNCERTAIN with justification]

---

## Synthesis
**How Challenge Changed Interpretation:** [Refinements, boundary conditions, confidence adjustments]
**Remaining Uncertainties:** [What's unclear]
**Related Features:** [Worth exploring]

## Conclusion
[Final synthesis] **Linguistic Function:** [...] **Potential Applications:** [...]
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
