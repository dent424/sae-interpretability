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

**Mount path in containers:** `/data`

## Volume Setup Commands

```bash
# Create volume (one-time)
py -3.12 -m modal volume create sae-data

# Upload data files
py -3.12 -m modal volume put sae-data "Data\sae_e32_k32_lr0.0003-final.pt" /
py -3.12 -m modal volume put sae-data "Data\mexican_national_metadata.npz" /
py -3.12 -m modal volume put sae-data "Data\mexican_national_sae_features_e32_k32_lr0_0003-final.h5" /

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
