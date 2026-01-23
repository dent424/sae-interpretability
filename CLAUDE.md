# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAE (Sparse Autoencoder) interpretability project for analyzing GPT-2 features extracted from 432K Mexican restaurant Yelp reviews. Uses Modal for serverless GPU compute.

## Build/Test Commands

```bash
# Run local CPU tests (from src/)
cd src && pytest

# Run specific test file
pytest src/tests/test_sae.py -v

# Run Modal GPU tests (requires Modal account)
modal run src/tests/modal_tests.py

# Run validation tests with real data
modal run src/tests/modal_tests.py::validate
```

## Architecture

### Module Structure
```
src/
├── config.py                # Constants: D_MODEL=768, K_ACTIVE=32, N_LATENTS=24576
├── modal_interpreter.py     # Modal backend service for SAE analysis
├── models/sae.py            # TopKSAE class - sparse autoencoder
├── processing/
│   ├── text.py              # Tokenization (64-token windows)
│   └── activations.py       # GPT-2 layer 8 activation extraction
├── data/
│   ├── h5_utils.py          # H5 sparse format queries
│   └── metadata.py          # Review metadata loading
├── scripts/
│   └── precompute_token_positions.py  # Generate review→token position pickle
└── tests/                   # pytest (local) + modal_tests.py (GPU)
```

### Data Pipeline
```
Text → tokenize_to_windows() → GPT-2 layer 8 → extract_activations() → TopKSAE → sparse features
```

### Key Constants (src/config.py)
- `D_MODEL = 768` - GPT-2 hidden dimension
- `EXPANSION = 32` - SAE expansion factor
- `K_ACTIVE = 32` - Top-K sparsity (32 active features per token)
- `N_LATENTS = 24576` - Total features (768 × 32)
- `HOOK = "blocks.8.hook_resid_pre"` - GPT-2 layer 8 residual stream

### H5 Sparse Format
The H5 files store activations in sparse format:
- `z_idx[token]` - Array of 32 active feature indices
- `z_val[token]` - Array of 32 corresponding activation values
- `rev_idx[token]` - Review ID for each token position

### TopKSAE Model
```python
# Forward pass: input → encode → top-k selection → decode
x_hat, z = sae(activations)  # z has exactly 32 non-zero entries per token
```

## Modal Deployment

```bash
# Upload data to Modal Volume (from Data/ directory)
modal volume create sae-data
cd Data
py -3.12 -m modal volume put sae-data sae_e32_k32_lr0.0003-final.pt sae_e32_k32_lr0.0003-final.pt
py -3.12 -m modal volume put sae-data mexican_national_metadata.npz mexican_national_metadata.npz
py -3.12 -m modal volume put sae-data mexican_national_sae_features_e32_k32_lr0_0003-final.h5 mexican_national_sae_features_e32_k32_lr0_0003-final.h5
py -3.12 -m modal volume put sae-data review_token_positions.pkl review_token_positions.pkl
cd ..

# Run interpreter test
py -3.12 -m modal run src/modal_interpreter.py::test_interpreter
```

### Precomputed Data
The `review_token_positions.pkl` file maps review IDs to their token positions in the H5 file. Regenerate if the H5 changes:
```bash
py -3.12 src/scripts/precompute_token_positions.py
```

## Notable Features

| Feature | Description |
|---------|-------------|
| 16751 | Emphatic expressions ("why in the world") |
| 20379 | Lexical patterns (gum/gumbo) |
| 11328 | Formatted lists (bullets, numbered) |

## Pre-computed Feature Data

The `feature data/` folder contains pre-generated analysis data for ~940 features:

### Contents
- **`feature_<ID>.json`** - Pre-computed feature analysis files (stats, top tokens, top activations, n-gram analysis)
- **`Feature_output.csv`** - Directory/index of features with columns:
  - `feature_index` - The feature ID
  - `rank_control` - Ranking with control metric
  - `rank_nocontrol` - Ranking without control metric
  - `interpretation` - Human-provided interpretation (if available)

### Usage
The `-existing` slash commands read from this folder:
```
/interpret-existing 16751        → reads feature data/feature_16751.json
/interpret-and-challenge-existing 16751  → reads feature data/feature_16751.json
```

## Feature Interpretation Workflow

Use slash commands to interpret SAE features with structured hypothesis testing.

**Important:** These features come from GPT-2's causally-masked residual stream. At each token position, the model only sees previous tokens (left context). Feature activations cannot depend on what comes *after* the token where they fire.

### Slash Commands

#### Interpretation Commands
| Command | Description |
|---------|-------------|
| `/interpret <feature_id>` | Full analysis: gather data from Modal, generate hypotheses, test, interpret |
| `/interpret-existing <feature_id>` | Same workflow but reads from `feature data/feature_<id>.json` |
| `/interpret-and-challenge <feature_id>` | Full analysis + adversarial challenge phase |
| `/interpret-and-challenge-existing <feature_id>` | Challenge workflow from `feature data/feature_<id>.json` |
| `/challenge <feature_id>` | Run challenge phase only on existing interpretation |

#### Batch/Parallel Commands
| Command | Description |
|---------|-------------|
| `/interpret-parallel <id1>, <id2>, ...` | Run multiple interpretations in parallel |
| `/interpret-parallel-existing <id1>, <id2>, ...` | Parallel from `feature data/` files |
| `/interpret-batch` | Batch workflow: find uninterpreted features, run in parallel, update CSV |

#### Verification Commands
| Command | Description |
|---------|-------------|
| `/verify <feature_id>` | Verify interpretation with logic, causal, and reproducibility checks |
| `/verify-batch` | Batch verify multiple features, update CSV with results |

#### Audit Commands
| Command | Description |
|---------|-------------|
| `/audit` | Audit all pending features in CSV for structural completeness |
| `/audit <feature_id>` | Audit single feature's results.json completeness |
| `/audit-schema` | Validate results.json schema fields, update CSV |
| `/audit-schema <ids> --fix` | Audit specific features, optionally fix missing fields |

### Output Structure

Each interpretation produces 3 files in `output/interpretations/feature<ID>/`:
- `audit.jsonl` - Step-by-step audit trail (append-only)
- `results.json` - Structured interpretation data
- `report.md` - Human-readable report

### Generating New Feature Data

To generate analysis data for a feature not in `feature data/`:
```bash
py -3.12 run_modal_utf8.py analyze_feature_json --feature-idx <FEATURE_ID>
```

Output: `output/feature_<ID>.json` (move to `feature data/` for use with `-existing` commands)

## Reference Documentation

When working on this project, consult these reference files based on your task:

### Architecture & Setup
| File | When to Use |
|------|-------------|
| `src/ARCHITECTURE.md` | Understanding module structure, data flow, class interfaces |
| `src/MODAL_SETUP.md` | Setting up Modal volumes, deployment configuration |

### Implementation References
| File | When to Use |
|------|-------------|
| `Reference Documents/Claude References/SAE_MODAL_IMPLEMENTATION.md` | Modal app patterns, GPU functions, volume mounting |
| `Reference Documents/Claude References/MODAL_REFERENCE.md` | Modal API syntax, decorators, Images, Volumes |
| `Reference Documents/Claude References/TRANSFORMERLENS_REFERENCE.md` | GPT-2 hooks, activation extraction, HookedTransformer API |
| `Reference Documents/Claude References/NOTEBOOK_CODE_REFERENCE.md` | Original analysis code from Jupyter notebooks |

### Data & Features
| File | When to Use |
|------|-------------|
| `Reference Documents/Claude References/DATA_FORMAT_REFERENCE.md` | H5 sparse format, metadata structure, file schemas |
| `Reference Documents/Claude References/FEATURE_CATALOG.md` | Known interesting SAE features, interpretations, examples |

### Original Code
| File | When to Use |
|------|-------------|
| `Reference Documents/Code/README.md` | Original notebook documentation, analysis methodology |

### Claude-Specific
| File | When to Use |
|------|-------------|
| `Reference Documents/Claude References/CLAUDE_SDK_REFERENCE.md` | Claude API patterns (if integrating LLM analysis) |
| `Reference Documents/Claude References/CLAUDE_SKILLS_REFERENCE.md` | Claude Code skills and capabilities |

## Permissions Policy

**IMPORTANT:** Before modifying `.claude/settings.local.json`, you MUST state the benefits and risks of the proposed change and get user approval.

### Current Allowed Permissions

The project uses minimal, scoped permissions:
- `Bash(py -3.12 run_modal_utf8.py:*)` - Modal wrapper only (safe: uses list-based subprocess)
- `Bash(mkdir output:*)` - Directory creation under output/ only
- `Write/Edit(output/**)` - File operations under output/ only

### Before Adding Any Permission

**State clearly:**

1. **What you want to add:** The exact permission string
2. **Why it's needed:** What operation requires this permission
3. **Benefit:** What becomes possible/easier
4. **Risk:** What could go wrong (be specific about attack vectors)
5. **Scope:** Is it minimal? Could it be narrower?

### Risk Levels by Permission Type

| Pattern | Risk | Why |
|---------|------|-----|
| `Bash(specific_script.py:*)` | LOW | If script uses list-based subprocess |
| `Bash(mkdir path:*)` | MEDIUM | Command chaining possible via `&` or `&&` |
| `Bash(python:*)` or `Bash(py:*)` | CRITICAL | Arbitrary code execution |
| `Bash(cat:*)` | HIGH | Can read any file (credentials, keys) |
| `Bash(echo:*)` | HIGH | Can write files via redirection |
| `Write(scoped/path/**)` | LOW | Properly scoped |
| `Write(**)` | CRITICAL | Can overwrite any file |

### Never Add Without Explicit User Request

- `Bash(python:*)` or `Bash(py -3.12:*)` - arbitrary Python execution
- `Bash(cat:*)` - read any file
- `Bash(rm:*)` - delete files
- `Bash(curl:*)` or `Bash(wget:*)` - network access
- Any permission with `**` paths outside `output/`

### Example: Proper Permission Request

```
I need to add a permission to run pytest.

**Permission:** `Bash(pytest:*)`
**Why:** To run test suite after code changes
**Benefit:** Automated test verification
**Risk:** MEDIUM - pytest could execute arbitrary Python in test files;
         command chaining possible via shell operators
**Scope:** Could narrow to `Bash(pytest src/tests:*)` to limit to test directory

Do you want me to add this permission?
```

## Tool Strategy (Permission-Aware Patterns)

When performing file operations, prefer Claude Code's native tools over Bash commands:

| Instead of... | Use... | Why |
|---------------|--------|-----|
| `copy file1 file2` | Read file1 → Write file2 | Bash copy not permitted |
| `cp file1 file2` | Read file1 → Write file2 | Bash cp not permitted |
| `cat file` | Read tool | Native tool, always works |
| `echo "x" > file` | Write tool | Native tool, always works |
| `grep pattern` | Grep tool | Native tool, more features |
| `find / ls` for patterns | Glob tool | Native tool, faster |
| `python script.py` | Only via scoped wrappers | Arbitrary Python not permitted |

**For Modal commands:** Always use the approved wrapper:
```bash
py -3.12 run_modal_utf8.py <command> --args
```

**For file copying:** Read the source file, then Write to destination:
```
1. Read("source/file.txt")
2. Write("destination/file.txt", <contents from step 1>)
```
