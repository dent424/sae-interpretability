# Interpret Multiple Features in Parallel

Run the full interpret-and-challenge pipeline for features **$ARGUMENTS** using parallel sub-agents.

## Instructions

1. Parse the feature indices from `$ARGUMENTS` (comma or space separated, e.g., "16751, 11328, 20379")

2. Read the full prompt from `.claude/commands/interpret-and-challenge.md`

3. For EACH feature index, spawn a sub-agent using the Task tool:
   - `subagent_type: "general-purpose"`
   - Replace `$ARGUMENTS` in the interpret-and-challenge prompt with the specific feature index
   - Each agent works completely independently

4. **CRITICAL:** Spawn ALL agents in a SINGLE message with multiple Task tool calls. This is what makes them run in parallel.

## Example

If user runs: `/interpret-parallel 16751, 11328, 20379`

You should send ONE message containing THREE Task tool calls:
- Task 1: interpret-and-challenge for feature 16751
- Task 2: interpret-and-challenge for feature 11328
- Task 3: interpret-and-challenge for feature 20379

Each agent writes to its own files:
- `output/interpretations/feature_16751.md` and `.json`
- `output/interpretations/feature_11328.md` and `.json`
- `output/interpretations/feature_20379.md` and `.json`

## Output

After all agents complete, summarize which features were interpreted and their verdicts.
