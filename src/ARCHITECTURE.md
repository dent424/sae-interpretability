# SAE Interpretability - Modal Architecture

This document specifies how to port the Colab notebook to Modal serverless GPU infrastructure.

**Source notebook:** `Reference Documents/Code/fixed_path_new_interpretability_notebook_mexican_national_e32_k32_lr0_0003.py`
**Reference docs:** `Reference Documents/Claude References/NOTEBOOK_CODE_REFERENCE.md`

---

## 1. Project Overview

**Goal:** Run SAE feature analysis on Modal's serverless GPU platform instead of Colab.

**Benefits:**
- No Colab session timeouts
- Persistent data storage via Modal Volumes
- Scalable parallel processing
- Pay-per-second GPU compute
- **CPU/GPU split for cost efficiency**

**Architecture:**
| Class | Resource | Purpose |
|-------|----------|---------|
| `SAEDataReader` | CPU only | H5 queries, corpus statistics, n-gram analysis |
| `SAEInterpreter` | GPU (T4) | GPT-2 → SAE inference on new text |

**Key principle:** Only use GPU when running text through the model. All precomputed H5 data queries run on CPU.

**Data files (stored on Modal Volume `sae-data`):**
- `mexican_national_sae_features_e32_k32_lr0_0003-final.h5` (~9GB) - Sparse SAE activations
- `mexican_national_metadata.npz` (~220MB) - Review texts and metadata
- `sae_e32_k32_lr0.0003-final.pt` (~604MB) - Trained SAE checkpoint
- `review_token_positions.pkl` (~258MB) - Pre-computed review→token position map

---

## 2. Module Structure

```
src/
├── ARCHITECTURE.md          # This file
├── MODAL_SETUP.md           # Modal setup instructions
├── __init__.py
├── config.py                # Constants and configuration
├── modal_interpreter.py     # Modal backend service (main entry point)
├── models/
│   ├── __init__.py
│   └── sae.py              # TopKSAE class
├── processing/
│   ├── __init__.py
│   ├── text.py             # Text processing and tokenization
│   └── activations.py      # GPT-2 activation extraction
├── analysis/
│   ├── __init__.py
│   ├── features.py         # Feature analysis functions
│   └── visualization.py    # HTML visualization (optional)
├── data/
│   ├── __init__.py
│   ├── h5_utils.py         # H5 sparse format utilities
│   └── metadata.py         # Metadata loading
├── scripts/
│   └── precompute_token_positions.py  # Generate review→token position pickle
└── tests/                   # pytest (local) + modal_tests.py (GPU)
```

---

## 3. Module Specifications

### 3.1 config.py

```python
# src/config.py
"""SAE configuration constants."""

# Model dimensions
D_MODEL = 768              # GPT-2 hidden dimension
EXPANSION = 32             # SAE expansion factor
K_ACTIVE = 32              # Top-K sparsity (active features per token)
N_LATENTS = 24576          # Total features (D_MODEL * EXPANSION)

# Hook configuration
LAYER_IDX = 8
HOOK = f"blocks.{LAYER_IDX}.hook_resid_pre"

# Processing
WINDOW_SIZE = 64           # Token window size for processing

# Modal Volume paths (when mounted at /data)
VOLUME_MOUNT = "/data"
H5_PATH = f"{VOLUME_MOUNT}/mexican_national_sae_features_e32_k32_lr0_0003-final.h5"
METADATA_PATH = f"{VOLUME_MOUNT}/mexican_national_metadata.npz"
SAE_CHECKPOINT_PATH = f"{VOLUME_MOUNT}/sae_e32_k32_lr0.0003-final.pt"
TOKEN_POSITIONS_PATH = f"{VOLUME_MOUNT}/review_token_positions.pkl"
```

---

### 3.2 models/sae.py

```python
# src/models/sae.py
"""TopKSAE sparse autoencoder model."""

import math
import torch
from torch import nn

class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder.

    Args:
        d_model: Input dimension (768 for GPT-2)
        n_lat: Number of latent features (24576)
        k_act: Number of active features per token (32)
        baseline: Baseline tensor for centering (shape: d_model)
    """

    def __init__(self, d_model: int, n_lat: int, k_act: int, baseline: torch.Tensor):
        super().__init__()
        self.W_dec = nn.Parameter(torch.randn(d_model, n_lat) / math.sqrt(d_model))
        self.W_enc = nn.Parameter(self.W_dec.t().clone())  # tied init only
        self.b_pre = nn.Parameter(baseline.clone())         # learnable baseline
        self.k = k_act

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input activations, shape (batch, d_model)

        Returns:
            x_hat: Reconstructed activations, shape (batch, d_model)
            z: Sparse feature activations, shape (batch, n_lat)
        """
        h = (x - self.b_pre) @ self.W_enc.t()
        top_idx = torch.topk(h, self.k, dim=-1).indices
        z = torch.zeros_like(h, dtype=h.dtype)
        z.scatter_(-1, top_idx, h.gather(-1, top_idx))
        x_hat = z @ self.W_dec.t() + self.b_pre
        return x_hat, z


def load_sae(checkpoint_path: str, device: str = "cuda") -> TopKSAE:
    """Load SAE from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on

    Returns:
        Loaded TopKSAE model in eval mode
    """
    from .config import D_MODEL, N_LATENTS, K_ACTIVE

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sae = TopKSAE(D_MODEL, N_LATENTS, K_ACTIVE, torch.zeros(D_MODEL))
    sae.load_state_dict(ckpt["ema_sae"])
    sae.eval()
    return sae.to(device)
```

---

### 3.3 processing/text.py

```python
# src/processing/text.py
"""Text processing and tokenization."""

import torch
from transformers import AutoTokenizer

from ..config import WINDOW_SIZE


def get_tokenizer() -> AutoTokenizer:
    """Get GPT-2 tokenizer with pad token configured."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': tok.eos_token})
    return tok


def tokenize_to_windows(text: str, tok: AutoTokenizer) -> torch.Tensor:
    """Tokenize text into 64-token windows.

    Args:
        text: Input text string
        tok: GPT-2 tokenizer

    Returns:
        Tensor of shape (n_windows, 64) with token IDs
    """
    ids = tok(text, add_special_tokens=False).input_ids

    if not ids:
        return None

    windows = []
    for i in range(0, len(ids), WINDOW_SIZE):
        window = ids[i:i + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            window = window + [tok.pad_token_id] * (WINDOW_SIZE - len(window))
        windows.append(window)

    return torch.tensor(windows)


def get_token_strings(text: str, tok: AutoTokenizer) -> list[str]:
    """Get list of token strings for text."""
    ids = tok(text, add_special_tokens=False).input_ids
    return tok.convert_ids_to_tokens(ids)
```

---

### 3.4 processing/activations.py

```python
# src/processing/activations.py
"""GPT-2 activation extraction."""

import torch
from transformer_lens import HookedTransformer

from ..config import D_MODEL, HOOK


def get_gpt2_model(device: str = "cuda") -> HookedTransformer:
    """Load GPT-2 model with hooks."""
    return HookedTransformer.from_pretrained("gpt2", device=device, dtype="float32")


def extract_activations(
    model: HookedTransformer,
    windows: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """Extract and normalize GPT-2 layer 8 activations.

    Args:
        model: HookedTransformer GPT-2 model
        windows: Token windows, shape (n_windows, 64)
        device: Device for computation

    Returns:
        Normalized activations, shape (n_tokens, D_MODEL)
    """
    all_activations = []

    def save_acts(act, hook):
        flat = act.detach().reshape(-1, D_MODEL)
        # Normalize: mean-center and L2-normalize
        flat = flat - flat.mean(-1, keepdim=True)
        flat = flat / flat.norm(dim=-1, keepdim=True)
        all_activations.append(flat)

    windows = windows.to(device)
    model.run_with_hooks(
        windows,
        fwd_hooks=[(HOOK, save_acts)],
        return_type=None
    )

    return torch.cat(all_activations, dim=0)
```

---

### 3.5 data/h5_utils.py

```python
# src/data/h5_utils.py
"""H5 sparse format utilities."""

import numpy as np
import h5py


def get_feature_activation_for_token(
    h5_file: h5py.File,
    token_idx: int,
    feature_idx: int
) -> float:
    """Get activation value for specific token and feature.

    The H5 file stores activations in sparse format:
    - z_idx[token]: array of 32 active feature indices
    - z_val[token]: array of 32 corresponding activation values

    Args:
        h5_file: Open h5py.File object
        token_idx: Global token index
        feature_idx: Feature index (0-24575)

    Returns:
        Activation value (0.0 if feature not active)
    """
    active_features = h5_file['z_idx'][token_idx]
    active_values = h5_file['z_val'][token_idx]

    mask = active_features == feature_idx
    if np.any(mask):
        return float(active_values[mask][0])
    return 0.0


def get_feature_activations_chunk(
    h5_file: h5py.File,
    feature_idx: int,
    start_token: int,
    end_token: int
) -> np.ndarray:
    """Get activations for a feature across a token range.

    Args:
        h5_file: Open h5py.File object
        feature_idx: Feature to query
        start_token: Start of range (inclusive)
        end_token: End of range (exclusive)

    Returns:
        Array of activation values, shape (end_token - start_token,)
    """
    chunk_size = end_token - start_token
    activations = np.zeros(chunk_size, dtype=np.float32)

    indices_chunk = h5_file['z_idx'][start_token:end_token]
    values_chunk = h5_file['z_val'][start_token:end_token]

    for i in range(chunk_size):
        mask = indices_chunk[i] == feature_idx
        if np.any(mask):
            activations[i] = values_chunk[i][mask][0]

    return activations


def get_review_id_for_token(h5_file: h5py.File, token_idx: int) -> str:
    """Get review ID for a token position."""
    rev_id = h5_file['rev_idx'][token_idx]
    if isinstance(rev_id, bytes):
        return rev_id.decode('utf-8')
    return str(rev_id)
```

---

### 3.6 data/metadata.py

```python
# src/data/metadata.py
"""Metadata loading utilities."""

import numpy as np
import pandas as pd


def load_metadata(npz_path: str) -> pd.DataFrame:
    """Load review metadata from NPZ file.

    Args:
        npz_path: Path to metadata.npz file

    Returns:
        DataFrame with columns: review_id, full_text, stars, useful, user_id, business_id
    """
    data = np.load(npz_path, allow_pickle=True)

    return pd.DataFrame({
        'review_id': data['review_ids'],
        'full_text': data['texts'],
        'stars': data['stars'],
        'useful': data['useful'],
        'user_id': data['user_ids'],
        'business_id': data['business_ids']
    })


def create_review_lookup(metadata_df: pd.DataFrame) -> dict[str, str]:
    """Create review_id -> text lookup dictionary."""
    return dict(zip(
        metadata_df['review_id'].astype(str),
        metadata_df['full_text'].astype(str)
    ))
```

---

### 3.7 modal_app.py

```python
# src/modal_app.py
"""Modal deployment for SAE analysis."""

import modal

# Create Modal app
app = modal.App("sae-interpreter")

# Define container image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformer-lens",
        "transformers",
        "numpy",
        "pandas",
        "h5py",
        "tqdm"
    )
)

# Create or reference volume for data
volume = modal.Volume.from_name("sae-data", create_if_missing=True)

# Volume mount path
VOLUME_MOUNT = "/data"


@app.cls(
    image=image,
    gpu="T4",  # Use T4 for cost efficiency, A100 for speed
    volumes={VOLUME_MOUNT: volume},
    timeout=1200
)
class SAEAnalyzer:
    """Modal class for SAE feature analysis."""

    @modal.enter()
    def setup(self):
        """Load models once when container starts."""
        import torch
        from transformer_lens import HookedTransformer
        from transformers import AutoTokenizer

        # Import local modules
        from models.sae import load_sae
        from config import SAE_CHECKPOINT_PATH

        print("Loading GPT-2...")
        self.model = HookedTransformer.from_pretrained("gpt2", device="cuda", dtype="float32")

        print("Loading tokenizer...")
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        if self.tok.pad_token is None:
            self.tok.add_special_tokens({'pad_token': self.tok.eos_token})

        print("Loading SAE...")
        self.sae = load_sae(SAE_CHECKPOINT_PATH, device="cuda")

        print("Setup complete!")

    @modal.method()
    def analyze_text(self, text: str, feature_indices: list[int]) -> dict:
        """Analyze text and return feature activations.

        Args:
            text: Input text to analyze
            feature_indices: List of feature indices to track

        Returns:
            dict with 'tokens', 'activations' (per feature), 'n_tokens'
        """
        from processing.text import tokenize_to_windows, get_token_strings
        from processing.activations import extract_activations

        # Tokenize
        windows = tokenize_to_windows(text, self.tok)
        if windows is None:
            return None

        tokens = get_token_strings(text, self.tok)

        # Extract GPT-2 activations
        gpt2_acts = extract_activations(self.model, windows)

        # Run through SAE
        import torch
        with torch.no_grad():
            _, z = self.sae(gpt2_acts)

        # Extract requested features
        tracked = {}
        for idx in feature_indices:
            tracked[f"feature_{idx}"] = z[:len(tokens), idx].cpu().numpy().tolist()

        return {
            'tokens': tokens,
            'activations': tracked,
            'n_tokens': len(tokens)
        }

    @modal.method()
    def get_feature_examples(
        self,
        feature_idx: int,
        n_examples: int = 100,
        context_tokens: int = 20
    ) -> list[dict]:
        """Get example contexts where a feature activates from H5 data.

        Args:
            feature_idx: Feature to analyze
            n_examples: Number of examples to return
            context_tokens: Tokens of context around activation

        Returns:
            List of dicts with 'context', 'token', 'activation', 'review_id'
        """
        import h5py
        from config import H5_PATH, METADATA_PATH
        from data.metadata import load_metadata, create_review_lookup
        from data.h5_utils import get_feature_activations_chunk, get_review_id_for_token

        # Load metadata
        metadata_df = load_metadata(METADATA_PATH)
        review_lookup = create_review_lookup(metadata_df)

        examples = []

        with h5py.File(H5_PATH, 'r') as h5:
            total_tokens = h5['z_idx'].shape[0]
            chunk_size = 100_000

            for start in range(0, total_tokens, chunk_size):
                if len(examples) >= n_examples:
                    break

                end = min(start + chunk_size, total_tokens)
                acts = get_feature_activations_chunk(h5, feature_idx, start, end)

                # Find active positions
                active_positions = (acts > 0).nonzero()[0]

                for local_idx in active_positions:
                    if len(examples) >= n_examples:
                        break

                    global_idx = start + local_idx
                    activation = float(acts[local_idx])
                    review_id = get_review_id_for_token(h5, global_idx)

                    # Get review text
                    review_text = review_lookup.get(review_id, "")
                    if not review_text:
                        continue

                    # Tokenize and extract context
                    tokens = self.tok(review_text, add_special_tokens=False).input_ids
                    token_strings = self.tok.convert_ids_to_tokens(tokens)

                    # Find position in review (simplified)
                    # In production, use review_token_positions mapping

                    examples.append({
                        'activation': activation,
                        'review_id': review_id,
                        'review_preview': review_text[:200]
                    })

        return sorted(examples, key=lambda x: x['activation'], reverse=True)


@app.local_entrypoint()
def main():
    """Example usage."""
    analyzer = SAEAnalyzer()

    # Analyze arbitrary text
    result = analyzer.analyze_text.remote(
        "This restaurant has amazing tacos!",
        [16751, 208, 3223]
    )
    print("Text analysis:", result)

    # Get feature examples from H5 data
    examples = analyzer.get_feature_examples.remote(
        feature_idx=16751,
        n_examples=10
    )
    print("Feature examples:", examples)
```

---

## 4. API Output Format: Sampling Metadata

**All corpus query methods return dictionaries with `sampling` metadata** to document potential biases.

### Why This Matters

- Corpus has 51M+ tokens - we can't scan everything
- Different methods use different sampling strategies
- Results may be biased depending on method

### Example Output

```python
result = reader.get_feature_stats.remote(feature_idx, sample_size=100000)

# Output:
{
  "sampling": {
    "method": "sequential_from_start",
    "description": "First N tokens in corpus order (not random)",
    "tokens_scanned": 100000,
    "corpus_coverage": 0.0019
  },
  "total_activations": 92,
  "mean_when_active": 0.073,
  "activation_rate": 0.00092
}
```

### Sampling Methods

| Method | Strategy | Bias |
|--------|----------|------|
| `sequential_from_start` | First N tokens | Early corpus |
| `top_by_activation` | Strongest activations | Clearest patterns |
| `first_n_then_sort` | First N found, sorted | Early + strength |

See `Reference Documents/Claude References/SAE_MODAL_IMPLEMENTATION.md` for full output format documentation.

---

## 5. Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        TEXT ANALYSIS PATH                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Text                                                      │
│      │                                                           │
│      ▼                                                           │
│  tokenize_to_windows() ──► Token IDs (n_windows, 64)            │
│      │                                                           │
│      ▼                                                           │
│  extract_activations() ──► GPT-2 Layer 8 (n_tokens, 768)        │
│      │                     (mean-centered, L2-normalized)        │
│      ▼                                                           │
│  sae.forward() ──► Sparse features (n_tokens, 24576)            │
│      │              (only 32 non-zero per token)                 │
│      ▼                                                           │
│  z[:, feature_idx] ──► Per-token activation for feature         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        H5 QUERY PATH                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Feature Index                                                   │
│      │                                                           │
│      ▼                                                           │
│  get_feature_activations_chunk() ──► Scan H5 sparse data        │
│      │                               (z_idx, z_val arrays)       │
│      ▼                                                           │
│  Find active token positions                                     │
│      │                                                           │
│      ▼                                                           │
│  get_review_id_for_token() ──► rev_idx lookup                   │
│      │                                                           │
│      ▼                                                           │
│  review_lookup[review_id] ──► Full review text                  │
│      │                                                           │
│      ▼                                                           │
│  Extract context around activation                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Function Mapping

| Notebook Function | New Location | Notes |
|-------------------|--------------|-------|
| `TopKSAE` class | `models/sae.py` | Exact implementation |
| `process_new_text()` | `SAEAnalyzer.analyze_text()` | Modal method |
| `process_text_for_analysis()` | Combined into `analyze_text()` | |
| `get_active_features()` | `analysis/features.py` | Optional utility |
| `get_feature_statistics()` | `analysis/features.py` | Optional utility |
| `visualize_feature_activation()` | `analysis/visualization.py` | HTML output |
| `get_feature_activation_for_token()` | `data/h5_utils.py` | Unchanged |
| `get_feature_activations_chunk()` | `data/h5_utils.py` | Unchanged |
| `load_review_texts_from_df()` | `data/metadata.py` | As `create_review_lookup()` |
| `extract_feature_data_adapted()` | `SAEAnalyzer.get_feature_examples()` | Modal method |
| `analyze_feature_tokens_with_text()` | Future: separate Modal function | Complex, defer |

---

## 7. Modal Deployment

### Setup Volume
```bash
# Create volume and upload data files
modal volume create sae-data
cd Data
py -3.12 -m modal volume put sae-data mexican_national_sae_features_e32_k32_lr0_0003-final.h5 mexican_national_sae_features_e32_k32_lr0_0003-final.h5
py -3.12 -m modal volume put sae-data mexican_national_metadata.npz mexican_national_metadata.npz
py -3.12 -m modal volume put sae-data sae_e32_k32_lr0.0003-final.pt sae_e32_k32_lr0.0003-final.pt
py -3.12 -m modal volume put sae-data review_token_positions.pkl review_token_positions.pkl
cd ..
```

### Generate Token Positions Pickle (if H5 changes)
```bash
py -3.12 src/scripts/precompute_token_positions.py
# Then re-upload to volume
```

### Run
```bash
# Test the interpreter
py -3.12 -m modal run src/modal_interpreter.py::test_interpreter
```

### Deploy (persistent endpoint)
```bash
py -3.12 -m modal deploy src/modal_interpreter.py
```

---

## 8. Implementation Order

1. **config.py** - Constants only, no dependencies
2. **models/sae.py** - TopKSAE class, depends on config
3. **processing/text.py** - Tokenization utilities
4. **processing/activations.py** - GPT-2 extraction
5. **data/h5_utils.py** - H5 utilities, no dependencies
6. **data/metadata.py** - Metadata loading
7. **modal_app.py** - Brings it all together
8. **analysis/features.py** - Optional analysis utilities
9. **analysis/visualization.py** - Optional HTML output

---

## 9. Success Criteria

- [ ] `modal run modal_app.py` executes without errors
- [ ] `SAEAnalyzer.analyze_text()` returns correct activations for test text
- [ ] `SAEAnalyzer.get_feature_examples()` returns examples from H5 data
- [ ] Feature 16751 activates on emphatic expressions ("why in the world")
- [ ] Results match notebook output for same inputs
