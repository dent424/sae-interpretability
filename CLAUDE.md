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
