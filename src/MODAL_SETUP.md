# Modal Setup

Configuration and setup details for running this project on Modal.

## Requirements

- **Python:** 3.9–3.13 (Modal does not support 3.14+)
- **Modal account:** [modal.com](https://modal.com)

## Installation

```bash
# Install Modal (use Python 3.12 if your default is 3.14)
py -3.12 -m pip install modal

# Authenticate (opens browser)
py -3.12 -m modal setup
```

## Volume Configuration

**Volume name:** `sae-data`

**Contents:**
| File | Size | Description |
|------|------|-------------|
| `sae_e32_k32_lr0.0003-final.pt` | ~604 MB | Trained SAE checkpoint |
| `mexican_national_metadata.npz` | ~220 MB | Review texts and metadata |
| `mexican_national_sae_features_e32_k32_lr0_0003-final.h5` | ~9 GB | Pre-computed sparse activations |
| `review_token_positions.pkl` | ~258 MB | Pre-computed review→token position map |

**Mount path in containers:** `/data`

## Volume Setup Commands

```bash
# Create volume (one-time)
py -3.12 -m modal volume create sae-data

# Upload data files (from Data/ directory to avoid path issues)
cd Data
py -3.12 -m modal volume put sae-data sae_e32_k32_lr0.0003-final.pt sae_e32_k32_lr0.0003-final.pt
py -3.12 -m modal volume put sae-data mexican_national_metadata.npz mexican_national_metadata.npz
py -3.12 -m modal volume put sae-data mexican_national_sae_features_e32_k32_lr0_0003-final.h5 mexican_national_sae_features_e32_k32_lr0_0003-final.h5
py -3.12 -m modal volume put sae-data review_token_positions.pkl review_token_positions.pkl
cd ..

# Verify contents
py -3.12 -m modal volume ls sae-data
```

## Running Modal Commands

Since Modal requires Python <3.14, prefix all commands with `py -3.12 -m`:

```bash
# Run tests
py -3.12 -m modal run src/tests/modal_tests.py

# Run specific test
py -3.12 -m modal run src/tests/modal_tests.py::test_pipeline

# Run validation (requires data on volume)
py -3.12 -m modal run src/tests/modal_tests.py::validate
```

## Local Token Storage

Modal stores authentication at `~/.modal/` (created by `modal setup`).

## Regenerating Token Positions Pickle

If the H5 file changes, regenerate the token positions map:

```bash
py -3.12 src/scripts/precompute_token_positions.py
cd Data
py -3.12 -m modal volume put sae-data review_token_positions.pkl review_token_positions.pkl
cd ..
```

This precomputes the review_id → token positions mapping, saving ~30-60s on container startup.
