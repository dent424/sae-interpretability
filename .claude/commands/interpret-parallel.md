# Interpret Multiple Features in Parallel

Run the full interpret-and-challenge pipeline for features **$ARGUMENTS** using parallel sub-agents.

## Instructions

1. Parse the feature indices from `$ARGUMENTS` (comma or space separated, e.g., "16751, 11328, 20379")

2. For EACH feature index, spawn a sub-agent using the Task tool:
   - `subagent_type: "general-purpose"`
   - Use this prompt template (replace {ID} with the feature index):
     ```
     Interpret SAE feature {ID}.

     1. Read the full instructions from: .claude/commands/interpret-and-challenge.md
     2. Follow ALL steps in that file for feature {ID} (replace $ARGUMENTS with {ID})
     3. Produce outputs in: output/interpretations/feature{ID}/
     ```
   - Each agent works completely independently

3. **CRITICAL:** Spawn ALL agents in a SINGLE message with multiple Task tool calls. This is what makes them run in parallel.

## Example

If user runs: `/interpret-parallel 16751, 11328, 20379`

You should send ONE message containing THREE Task tool calls:
- Task 1: interpret-and-challenge for feature 16751
- Task 2: interpret-and-challenge for feature 11328
- Task 3: interpret-and-challenge for feature 20379

Each agent writes to its own folder:
- `output/interpretations/feature16751/` (audit.jsonl, results.json, report.md)
- `output/interpretations/feature11328/` (audit.jsonl, results.json, report.md)
- `output/interpretations/feature20379/` (audit.jsonl, results.json, report.md)

## Output

After all agents complete, summarize which features were interpreted and their verdicts.
