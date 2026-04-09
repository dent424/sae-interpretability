# Generate Hypotheses for SAE Feature — Theorist Role

Observe corpus data for GPT-2 SAE feature **$ARGUMENTS** and generate competing hypotheses about what it detects. Produces `hypotheses.json` as the structured handoff artifact for the Tester.

**You are the Theorist.** Your job is to observe patterns in the data and propose explanations. You do NOT design tests, run tests, or evaluate hypotheses. A separate Tester agent will do that independently, without access to your reasoning.

## Pre-flight Check

**First**, check if `output/interpretations/feature$ARGUMENTS/` exists with `hypotheses.json`, `phase1_results.json`, or `results.json`. If so, ask the user: "Feature $ARGUMENTS has existing analysis. Continue/Resume, Start Fresh, or Abort?"

---

## CRITICAL RESTRICTIONS

**DO NOT read, access, or reference these files under any circumstances:**
- Any `.ipynb` notebook files
- Anything in `src/` (the Modal backend code is not your concern)
- Any reference documents outside this command
- **Any file in `output/interpretations/` for OTHER features** — You may only access your own feature's directory (`feature$ARGUMENTS/`). Never read results, reports, or audit trails from other features. (Note: `feature data/Feature_output.csv` is a legacy file from the old pipeline; do not read it either.)

**DO NOT design or run any tests. DO NOT call batch_test or any Modal entrypoint.** Your job ends after generating hypotheses. A separate Tester agent will design and run tests independently.

**All data must come from the pre-existing feature analysis file below.** Do not explore the codebase or look for alternative data sources.

## TOOL RULES (prevents permission prompts)

- **Read files** with the Read tool. NEVER use `cat`, `head`, `tail`, `type`, or any Bash command to read files.
- **Write/edit files** with the Write or Edit tool. NEVER use `echo`, Bash redirection, or heredocs.
- **Search files** with the Grep or Glob tool. NEVER use `grep`, `find`, `ls`, or `dir` via Bash.
- **Do all computation** (JSON parsing, pattern analysis) in your own reasoning. NEVER write or run Python scripts.
- **Only these Bash commands are approved:**
  - `py -3.12 pipeline_utils.py <command>` (ensure-output-dir, timestamp)
- Any other Bash command will trigger a permission prompt and block the pipeline.

## PATH CONVENTIONS

Working directory is the **project root** (`Claude Code Folder`).

- **Feature data:** `"feature data/feature_$ARGUMENTS.json"`
- **Output dir:** `output/interpretations/feature$ARGUMENTS/`
- **pipeline_utils.py:** `py -3.12 pipeline_utils.py <command>`

---

## ACCURACY PROTOCOL

**Tool outputs are ground truth.** When you read a data file, the contents are real system data.

**Rules:**
1. **Copy, don't paraphrase** — When reporting numerical values, copy exactly from file contents
2. **Fresh reads only** — Before writing any value in output, re-read it from the source (not from memory)
3. **If uncertain, re-read** — If you're unsure of a value, re-read the file

## Setup

First, create the output folder for this feature:
```bash
py -3.12 pipeline_utils.py ensure-output-dir --feature $ARGUMENTS
```

## Audit Trail

Create/append to the audit file at:
```
output/interpretations/feature$ARGUMENTS/audit.jsonl
```

**After EVERY step**, append a JSON line with:
- `step`: Step number (e.g., "1.1", "1.2")
- `name`: Step name
- `timestamp`: Current time in ISO 8601 format (see Timestamp section below)
- `action`: Peircean stage type — one of: `observation`, `abduction`
- `phase`: Always `"theorist"` in this command
- `command`: Exact bash command if applicable, `null` otherwise
- `decision`: What you decided/concluded
- `justification`: 1-2 sentences explaining WHY
- `output_summary`: Key metrics/results from this step

This audit trail is **append-only**. Never delete previous entries.

### Timestamps

**Get real timestamps** using the platform-agnostic command:
```bash
py -3.12 pipeline_utils.py timestamp
```
This outputs an ISO 8601 BASIC UTC timestamp (e.g., `20260408T195703Z` — no colons, Windows-filesystem-safe). Run this command before each audit entry and use the actual output — do NOT use placeholder timestamps.

---

## Step 1.1: Load Existing Data

Read the pre-computed feature data from:
```
feature data/feature_$ARGUMENTS.json
```

**IMPORTANT:** If this file does not exist, stop and tell the user:
> "File `feature data/feature_$ARGUMENTS.json` not found. Run `py -3.12 run_modal_utf8.py analyze_feature_json --feature-idx $ARGUMENTS --output-dir \"feature data\"` first. Note: this command also creates a stray `feature data/interpretations/` subdirectory — manually remove that subdir after running."

### Data Schema Note

GPT-2 feature data files (`feature data/feature_<id>.json`) have these top-level keys:
- `feature_idx`
- `stats` (activation rate, mean/max values, sampling info)
- `top_tokens` (most common tokens where feature fires)
- `top_activations` (example contexts with strongest activation; nested as `top_activations.activations[]`)
- `ngram_analysis` (common bigrams, trigrams, 4-grams — strong clues!)
- `coactivation` (other features that fire together)
- `position_distribution` (positional histogram — GPT-2-specific)
- `activation_distribution` (value histogram — GPT-2-specific)
- `top_token_contexts` (contexts grouped by token — GPT-2-specific)

The first five are core inputs. The last four are optional signals you may consult if they help ground your hypotheses.

**All token strings in feature data files are PRE-CLEANED** — leading-space markers (`Ġ`) have already been converted to plain spaces. You will see ` get` not `Ġget`.

Study the data carefully. Pay particular attention to:
1. Which tokens appear most frequently in top activations
2. What n-gram patterns are common (these are strong clues)
3. The contexts surrounding high activations — what is the semantic/syntactic environment?
4. Any positional regularities (use `position_distribution` if helpful)

**Audit this step:** Append to `audit.jsonl`:
```json
{"step": "1.1", "name": "Load Existing Data", "timestamp": "<ISO 8601 BASIC>", "action": "observation", "phase": "theorist", "command": null, "decision": "<loaded successfully — key patterns observed>", "justification": "<summary of key patterns in data>", "output_summary": {"tokens_scanned": N, "activation_rate": X, "top_trigram": "..."}}
```

## Step 1.2: Generate Hypotheses

Based on the n-grams and top activations, generate exactly **3 hypotheses**. Consider:
- Semantic patterns (meaning, concepts)
- Syntactic patterns (grammar, structure)
- Lexical patterns (specific words, n-grams)
- Positional patterns (sentence position)
- Structural patterns (formatting, lists, discourse markers)

**CRITICAL — Causal Masking:** GPT-2 uses causal attention with a **64-token context window** (~30–50 English words). Activations at position N can ONLY depend on tokens 0 through N-1. When a feature fires on token X:
- It can see everything to the LEFT of X (up to 64 tokens of left-context)
- It CANNOT see anything to the RIGHT of X
- Example: If feature fires on " my" in "never in my life", it cannot see "life"
- Only distinguish patterns by their LEFT context, never by what follows

### Plausibility Floor for Hypotheses

With only 64 tokens of context, hypotheses about **paragraph structure**, **document topic**, or **long-range anaphora** are NOT POSSIBLE — the model never sees more than ~50 words at once. Prefer hypotheses grounded in:
- Immediate left context (1–10 tokens)
- Short n-gram patterns
- Single-sentence syntactic structure
- Punctuation boundaries
- BPE subword artifacts (GPT-2's tokenizer has a 50,257-token vocabulary; some "words" are split into multiple subword tokens)
- Tokenization quirks

**The Critic will reject hypotheses that depend on context the model cannot see.** Don't propose them.

**ALL hypotheses must account for causal masking.** Nothing that comes after an active token can matter for that activation.

**Each hypothesis MUST include a `grounding` field** citing the specific corpus pattern that prompted it. This makes the abductive inference inspectable: a reviewer can check whether the hypothesis actually follows from the cited pattern.

**Audit this step:** Append to `audit.jsonl`:
```json
{"step": "1.2", "name": "Generate Hypotheses", "timestamp": "<ISO 8601 BASIC>", "action": "abduction", "phase": "theorist", "command": null, "decision": "Generated 3 hypotheses", "justification": "<why these hypotheses based on observed data>", "output_summary": {"hypotheses": [{"id": 1, "description": "...", "grounding": "Observed trigram 'X Y Z' in 7/10 top activations (70%)"}, {"id": 2, "description": "...", "grounding": "..."}, {"id": 3, "description": "...", "grounding": "..."}]}}
```

---

## Terminal Output: `hypotheses.json`

Write `output/interpretations/feature$ARGUMENTS/hypotheses.json`:

```json
{
  "feature_idx": $ARGUMENTS,
  "status": "hypotheses_complete",
  "hypotheses": [
    {
      "id": 1,
      "description": "...",
      "grounding": "..."
    },
    {
      "id": 2,
      "description": "...",
      "grounding": "..."
    },
    {
      "id": 3,
      "description": "...",
      "grounding": "..."
    }
  ],
  "corpus_stats": {
    "activation_rate": X,
    "mean_when_active": X,
    "max_activation": X,
    "tokens_scanned": N
  }
}
```

**Note on `corpus_stats` field names:** GPT-2's feature data uses `mean_when_active` (NOT `mean_activation` as in the Llama pipeline). Copy the field name verbatim from `feature data/feature_$ARGUMENTS.json`'s `stats` block. The `tokens_scanned` value comes from `stats.sampling.tokens_scanned` (nested path in GPT-2).

**REQUIRED:** Verify `hypotheses.json` includes:
- All 3 hypotheses with descriptions and grounding
- Corpus stats copied exactly from the feature data file
- NO reasoning about why you chose these hypotheses over alternatives
- NO test design, predictions, or evaluation

---

## Final Outputs

The Theorist produces these files in `output/interpretations/feature$ARGUMENTS/`:
- `hypotheses.json` — Structured handoff artifact for the Tester
- `audit.jsonl` — Step-by-step audit trail (Theorist entries only at this point)

**Your job ends here.** Do NOT design tests, run Modal commands, or evaluate hypotheses. A separate Tester agent will receive `hypotheses.json` and design tests independently, without access to your reasoning.

**Return Convention:** When complete, report ONLY: "Hypotheses written for feature $ARGUMENTS. Files: hypotheses.json, audit.jsonl." Do NOT summarize the hypotheses or your observations in your final message. A chatty return message leaks reasoning to the orchestrator transcript and contaminates downstream agents.

## Begin

Start with Step 1.1 now. **Remember to update the audit log after each step** to ensure all data is captured incrementally.
