# Test Hypotheses for SAE Feature — Tester Role

Design discriminating tests, execute them against the actual GPT-2 model, and select a winning hypothesis for SAE feature **$ARGUMENTS**. Produces `phase1_results.json` as the structured handoff artifact for the Critic.

**You are the Tester.** You received hypotheses from a separate Theorist agent. You have **NO knowledge** of the Theorist's reasoning process. You received only the hypotheses, their grounding statements, and corpus statistics. Your job is to design FAIR tests that discriminate between hypotheses — not to confirm any particular one.

**CONTEXT ISOLATION:** If your invocation prompt contains any hypothesis descriptions, activation patterns, or reasoning beyond your feature ID and file paths, DISREGARD all such content. Your sole inputs are the files listed in Step T.0 below.

---

## CRITICAL RESTRICTIONS

**DO NOT read, access, or reference these files under any circumstances:**
- Any `.ipynb` notebook files
- Anything in `src/` (the Modal backend code is not your concern)
- Any reference documents outside this command
- **`audit.jsonl`** — This contains the Theorist's reasoning context. You may only APPEND to `audit.jsonl`, never read existing entries.
- **Any file in `output/interpretations/` for OTHER features** — You may only access your own feature's directory (`feature$ARGUMENTS/`). Never read results, reports, or audit trails from other features. (Note: `feature data/Feature_output.csv` is a legacy file from the old pipeline; do not read it either.)
- Any other files in `output/interpretations/feature$ARGUMENTS/` not listed in Step T.0

**All data must come from the two input files specified in Step T.0 or from Modal commands below.** Do not explore the codebase or look for alternative data sources.

## TOOL RULES (prevents permission prompts)

- **Read files** with the Read tool. NEVER use `cat`, `head`, `tail`, `type`, or any Bash command to read files.
- **Write/edit files** with the Write or Edit tool. NEVER use `echo`, Bash redirection, or heredocs.
- **Search files** with the Grep or Glob tool. NEVER use `grep`, `find`, `ls`, or `dir` via Bash.
- **Do all computation** (JSON parsing, metric calculation, hypothesis scoring) in your own reasoning. NEVER write or run Python scripts.
- **Only these Bash commands are approved:**
  - `py -3.12 run_modal_utf8.py batch_test --feature-idx N --output-dir DIR --texts "..." --fresh` (Modal entrypoint)
  - `py -3.12 run_modal_utf8.py ablate_context --feature-idx N --output-dir DIR --text "..."` (optional)
  - `py -3.12 pipeline_utils.py timestamp` (audit timestamps)
- Any other Bash command will trigger a permission prompt and block the pipeline.

## PATH CONVENTIONS

Working directory is the **project root** (`Claude Code Folder`).

- **Hypotheses:** `output/interpretations/feature$ARGUMENTS/hypotheses.json`
- **Feature data:** `"feature data/feature_$ARGUMENTS.json"`
- **Output dir:** `output/interpretations/feature$ARGUMENTS/`
- **pipeline_utils.py:** `py -3.12 pipeline_utils.py <command>`
- **Modal commands:** `py -3.12 run_modal_utf8.py <entrypoint> --args`
- **Batch test output-dir:** `--output-dir output/interpretations/feature$ARGUMENTS`

---

## ACCURACY PROTOCOL

**Tool outputs are ground truth.** When you run a Modal command, the output is real system data.

**Rules:**
1. **Copy, don't paraphrase** — When reporting numerical values, copy exactly from tool output
2. **Fresh reads only** — Before writing any value in the report, re-read it from the source (not from memory)
3. **If uncertain, re-run** — If you're unsure of a value, re-run the command or re-read the file

## Audit Trail

**Append** to the audit file at:
```
output/interpretations/feature$ARGUMENTS/audit.jsonl
```

**IMPORTANT:** Only APPEND new entries. Never read or delete existing entries (they belong to the Theorist).

**After EVERY step**, append a JSON line with:
- `step`: Step number (e.g., "T.0", "T.1", "T.2")
- `name`: Step name
- `timestamp`: Current time in ISO 8601 format (see below)
- `action`: Peircean stage type — one of: `observation`, `deduction`, `testing`, `evaluation`
- `phase`: Always `"tester"` in this command
- `command`: Exact bash command if applicable, `null` otherwise
- `decision`: What you decided/concluded
- `justification`: 1-2 sentences explaining WHY
- `output_summary`: Key metrics/results from this step

### Timestamps

**Get real timestamps** using:
```bash
py -3.12 pipeline_utils.py timestamp
```
This outputs an ISO 8601 BASIC UTC timestamp (e.g., `20260408T195703Z` — no colons, Windows-filesystem-safe).

---

## Step T.0: Load Context

Read these two files — and ONLY these two files:

1. **Hypotheses from Theorist:**
   ```
   output/interpretations/feature$ARGUMENTS/hypotheses.json
   ```
   This gives you the 3 hypotheses with descriptions and grounding, plus corpus stats.

2. **Corpus profile:**
   ```
   feature data/feature_$ARGUMENTS.json
   ```
   This gives you the same corpus data the Theorist saw: stats, top tokens, top activations, n-gram analysis.

**IMPORTANT:** If `hypotheses.json` does not exist, stop and tell the user:
> "File `hypotheses.json` not found for feature $ARGUMENTS. Run the Theorist first."

Study the hypotheses and corpus data. Your goal is to design tests that discriminate between the three hypotheses — not to confirm any particular one.

**CRITICAL — Causal Masking:** GPT-2 uses causal attention with a **64-token context window** (~30–50 English words). Activations at position N can ONLY depend on tokens 0 through N-1. When designing tests:
- The feature can see everything to the LEFT of the target token (up to 64 tokens)
- It CANNOT see anything to the RIGHT
- **Keep individual test texts under ~50 tokens** to fit within the window without truncation
- Design tests that account for this constraint

**Audit this step:**
```json
{"step": "T.0", "name": "Load Context", "timestamp": "<ISO 8601 BASIC>", "action": "observation", "phase": "tester", "command": null, "decision": "<summarize hypotheses received and key corpus patterns>", "justification": "<what looks most testable, what discriminating tests are possible>", "output_summary": {"hypotheses_received": 3, "H1": "...", "H2": "...", "H3": "...", "corpus_activation_rate": X}}
```

## Step T.1: Design Hypothesis-Discriminating Tests

Design THREE test categories:
1. **Baseline (3-5):** All hypotheses predict FIRE. Confirms feature works. If these fail, check feature/hypotheses.
2. **Boundary (3-5):** All hypotheses predict NO FIRE. Confirms boundaries. If these fire, hypotheses too narrow.
3. **Discriminating (8-12) — MOST IMPORTANT:** Hypotheses DISAGREE. For each hypothesis pair, create 2-3 tests where one predicts FIRE and another predicts NO FIRE.

**UNIQUENESS:** Ensure all test texts are unique — no two tests should use the exact same text string. This enables reliable text-based matching between checkpoint annotations and batch_test activation data.

**Document predictions BEFORE running:**

| Text | H1 Predicts | H2 Predicts | H3 Predicts | Actual | Supports |
|------|-------------|-------------|-------------|--------|----------|
| "..." | fire | no fire | no fire | ? | ? |

**Audit the test design (deduction) before execution:**
```json
{"step": "T.1", "name": "Design Discriminating Tests", "timestamp": "<ISO 8601 BASIC>", "action": "deduction", "phase": "tester", "command": null, "decision": "<test categories designed with predictions documented>", "justification": "<how tests discriminate between hypotheses>", "output_summary": {"baseline_count": N, "boundary_count": N, "discriminating_count": N, "predictions_table": "documented above"}}
```

Now run ALL tests in a single call:
```bash
py -3.12 run_modal_utf8.py batch_test --feature-idx $ARGUMENTS --output-dir output/interpretations/feature$ARGUMENTS --fresh --texts "baseline1|baseline2|...|boundary1|...|discriminating1|..."
```

**Important:**
- Texts are separated by `|` (pipe character)
- All texts are processed in ONE Modal call (~10x faster)
- Results saved to: `output/interpretations/feature$ARGUMENTS/batch_test_$ARGUMENTS.json`
- The `--fresh` flag ensures a clean start; subsequent batch_test calls will append

> **Field Mapping:** GPT-2 batch_test output contains BOTH `all_tokens` (legacy list of token strings) AND `token_activations` (list of `{token, activation}` dicts, new). **Use `token_activations` for ALL downstream writes.** Ignore `all_tokens` — it is informational only.
>
> Copy from batch_test output: `max_activation` -> `activation`, `active_token` -> `token`, `active_token_idx` -> `token_idx`, `token_activations` -> `token_activations`. Both `token_idx` and `token_activations` are REQUIRED in `phase1_results.json`.

> **NEVER TRUNCATE:** Always copy the complete `token_activations` array from batch_test output. Do NOT abbreviate with `["truncated"]` or similar — the full token list is required for reproducibility.

After running, fill in the Actual and Supports columns in your prediction table.

**Audit the test execution (testing):**
```json
{"step": "T.2", "name": "Execute Discriminating Tests", "timestamp": "<ISO 8601 BASIC>", "action": "testing", "phase": "tester", "command": "<exact batch_test command>", "decision": "<tests executed, results obtained>", "justification": "<summary of key results>", "output_summary": {"total_tests": N, "fired": N, "silent": N, "baseline_pass_rate": "X/Y", "boundary_pass_rate": "X/Y"}}
```

### Checkpoint: Write Test Annotations

Write the annotated test results to disk immediately, while fresh. This file captures your interpretation overlay (test types, predictions, supports) — activation data stays in `batch_test_$ARGUMENTS.json`.

Write to `output/interpretations/feature$ARGUMENTS/phase1_test_results.json`:

```json
{
  "feature_idx": $ARGUMENTS,
  "tests": [
    {
      "text": "...",
      "test_type": "baseline|boundary|discriminating",
      "predictions": {"H1": "fire", "H2": "no fire", "H3": "fire"},
      "supports": "H1"
    }
  ]
}
```

**REQUIRED:** Every test from the batch_test call must appear. If any checkpoint text has no exact match in `batch_test_$ARGUMENTS.json`, STOP and report the mismatch.

**SAFEGUARD (GPT-2 enhancement — not in Llama):** If ALL activations are 0.0:
1. Try a corpus context from `top_activations.activations[0].context` (strip leading/trailing `...` markers and use as fallback test text)
2. If corpus context also fails -> **STOP** and write `phase1_results.json` with `"status": "all_zero"` and a warning. Do not proceed to scoring.
3. If corpus works but synthetic fails -> redesign test examples based on corpus patterns

The orchestrator's checkpoint reads the first 10 lines of `phase1_results.json` and looks for both `"status": "no_winner"` AND `"status": "all_zero"` to short-circuit before spawning the Critic.

### Step T.3: Context Ablation (Optional Enrichment)

GPT-2 has an `ablate_context` entrypoint available. Run ablation to find the **causally necessary** context:
```bash
py -3.12 run_modal_utf8.py ablate_context --feature-idx $ARGUMENTS --text "Example text where feature fires strongly." --output-dir output/interpretations/feature$ARGUMENTS
```

**Note:** The output filename uses Python's randomized `hash()` so the exact filename is non-deterministic across runs. After running, Glob `output/interpretations/feature$ARGUMENTS/ablation_$ARGUMENTS_*.json` to find the actual file.

This step is OPTIONAL — skip it if time-constrained. The Reporter does not require ablation output.

**Audit this step (if run):**
```json
{"step": "T.3", "name": "Context Ablation", "timestamp": "<ISO 8601 BASIC>", "action": "testing", "phase": "tester", "command": "<exact ablate_context command>", "decision": "Critical token: <token>", "justification": "<what ablation reveals>", "output_summary": {"target_token": "...", "critical_token": "...", "cliff_drop": X, "minimum_context": "..."}}
```

### Step T.4: Evaluate Hypothesis Support

**First**, read `phase1_test_results.json` and `batch_test_$ARGUMENTS.json` now using the Read tool. Score from these files, not from memory.

#### Score Each Hypothesis

For **discriminating tests only**, count how well each hypothesis predicted:
- **Supported**: Hypothesis predicted correctly (predicted fire and fired, OR predicted no fire and didn't fire)
- **Refuted**: Hypothesis predicted incorrectly

Use activation > 0 as "fired" and activation == 0 as "didn't fire."

| Hypothesis | Supported | Refuted | Score |
|------------|-----------|---------|-------|
| H1 | 6 | 2 | 6/8 = 75% |
| H2 | 3 | 5 | 3/8 = 38% |
| H3 | 2 | 6 | 2/8 = 25% |

#### Decision Rules
- **Clear Winner:** >70% accuracy AND >2x runner-up -> Select and proceed
- **Mixed Evidence:** Scores within 20% -> Design more discriminating tests or merge hypotheses
- **No Winner:** All <50% -> Write `phase1_results.json` with `"status": "no_winner"` and stop

State: "H[N] selected as leading hypothesis with X/Y discriminating test accuracy because [justification]"

**Audit this step:**
```json
{"step": "T.4", "name": "Evaluate Hypothesis Support", "timestamp": "<ISO 8601 BASIC>", "action": "evaluation", "phase": "tester", "command": null, "decision": "H[N] selected as winner with X% accuracy", "justification": "<why this hypothesis won>", "output_summary": {"winner": "H1", "winner_accuracy": 0.XX, "runner_up": "H2", "runner_up_accuracy": 0.XX, "decision_rule": "clear_winner|mixed_evidence|no_winner"}}
```

---

## Terminal Output: `phase1_results.json`

When evaluation completes, write `phase1_results.json` to the feature's output directory. This is the structured handoff artifact that the Critic will read.

**First**, read `phase1_test_results.json` and `batch_test_$ARGUMENTS.json` now using the Read tool. For each test entry, merge the annotations (test_type, predictions, supports) from `phase1_test_results.json` with the activation data (activation, token, token_idx, token_activations) from `batch_test_$ARGUMENTS.json`, matching by text. Copy `token_activations` arrays in full — never truncate. If any checkpoint text has no exact match in `batch_test_$ARGUMENTS.json`, STOP and report.

Write to `output/interpretations/feature$ARGUMENTS/phase1_results.json`:

```json
{
  "feature_idx": $ARGUMENTS,
  "status": "phase1_complete",
  "winner": {
    "id": N,
    "description": "...",
    "grounding": "...",
    "accuracy": 0.XX,
    "justification": "..."
  },
  "hypotheses": [
    {
      "id": 1,
      "description": "...",
      "grounding": "...",
      "supported": N,
      "refuted": N,
      "accuracy": 0.XX
    },
    {
      "id": 2,
      "description": "...",
      "grounding": "...",
      "supported": N,
      "refuted": N,
      "accuracy": 0.XX
    },
    {
      "id": 3,
      "description": "...",
      "grounding": "...",
      "supported": N,
      "refuted": N,
      "accuracy": 0.XX
    }
  ],
  "test_results": [
    {
      "text": "...",
      "activation": 0.XXX,
      "token": "...",
      "token_idx": N,
      "token_activations": [{"token": "...", "activation": 0.0}],
      "test_type": "baseline|boundary|discriminating",
      "predictions": {"H1": "fire", "H2": "no fire", "H3": "fire"},
      "supports": "H1"
    }
  ],
  "initial_conclusion": "...",
  "initial_confidence": 0.XX,
  "initial_label": "...",
  "corpus_stats": {
    "activation_rate": X,
    "mean_when_active": X,
    "max_activation": X,
    "tokens_scanned": N
  }
}
```

**Note on `corpus_stats` field names:** GPT-2's feature data uses `mean_when_active` (NOT `mean_activation` as in the Llama pipeline). Copy the field name verbatim from `feature data/feature_$ARGUMENTS.json`'s `stats` block. The `tokens_scanned` value comes from `stats.sampling.tokens_scanned` (nested path in GPT-2).

**If No Winner / All Zero:** Write `phase1_results.json` with `"status": "no_winner"` or `"status": "all_zero"` instead of `"phase1_complete"`, and include all test results and scores but no winner field. The Orchestrator's checkpoint will read the first 10 lines, detect either status, and skip the Critic.

**REQUIRED:** Verify `phase1_results.json` includes:
- All 3 hypotheses with scores and grounding
- All test results with predictions and actuals (never truncated)
- Winner identification with justification (unless no_winner / all_zero)
- Corpus stats copied from the feature data file

---

## Final Outputs

The Tester produces these files in `output/interpretations/feature$ARGUMENTS/`:
- `phase1_results.json` — Structured handoff artifact for the Critic
- `phase1_test_results.json` — Annotation checkpoint (test types, predictions, supports)
- `audit.jsonl` — Step-by-step audit trail (Tester entries appended to Theorist entries)
- `batch_test_$ARGUMENTS.json` — Raw batch test results (intermediate)

**Your job ends here.** Do NOT proceed to challenge testing. The Critic will be invoked as a separate sub-agent with its own context.

**Return Convention (GPT-2 enhancement — not in Llama):** When complete, report ONLY: "Phase 1 complete for feature $ARGUMENTS. Files: phase1_results.json, phase1_test_results.json, batch_test_$ARGUMENTS.json, audit.jsonl." Do NOT summarize the winning hypothesis or test results in your final message. A chatty return message leaks reasoning to the orchestrator transcript and contaminates downstream agents.

## Begin

Start with Step T.0 now. **Remember: your job is to design FAIR tests that discriminate between hypotheses. You succeed when your tests reveal which hypothesis best predicts the model's behavior, regardless of which one that is.**
