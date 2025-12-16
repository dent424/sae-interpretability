# Data Format Reference

Reference for the H5 and NPZ data formats used in the SAE interpretability project.

## Overview

The project uses three data files:

| File | Format | Contents |
|------|--------|----------|
| `mexican_national_sae_features_*.h5` | HDF5 | Sparse SAE activations per token |
| `mexican_national_metadata.npz` | NumPy | Review texts and metadata |
| `review_token_positions.pkl` | Pickle | Pre-computed review→token position map |

**Key insight:** H5 contains activations, NPZ contains texts, pickle speeds up position lookups. They link via `review_id`.

---

## H5 File Structure (SAE Activations)

### Datasets

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `z_idx` | (n_tokens, k) | int | Feature indices for each token |
| `z_val` | (n_tokens, k) | float32 | Activation values for each token |
| `rev_idx` | (n_tokens,) | bytes | Review ID for each token |
| `review_ids_unique` | (n_reviews,) | bytes | List of unique review IDs |

Where:
- `n_tokens` = total tokens across all reviews (~50M+)
- `k` = 32 (Top-K sparsity - each token has exactly 32 active features)
- `n_reviews` = 432,248 reviews

### Sparse Format Explanation

The SAE has 24,576 features but only 32 are active per token (Top-K=32).

Instead of storing a dense (n_tokens, 24576) array, we store:
- `z_idx[t]` = array of 32 feature indices active at token t
- `z_val[t]` = array of 32 activation values for those features

**Example:**
```python
# Token 0 has these 32 features active:
z_idx[0] = [142, 891, 1023, 3456, ...]  # 32 feature indices
z_val[0] = [0.087, 0.052, 0.134, 0.023, ...]  # 32 activation values
```

### Reading H5 Data

```python
import h5py
import numpy as np

with h5py.File(H5_PATH, 'r') as f:
    # Check structure
    print(f.keys())  # ['z_idx', 'z_val', 'rev_idx', 'review_ids_unique']

    # Get dimensions
    n_tokens = f['z_idx'].shape[0]
    k = f['z_idx'].shape[1]  # 32

    # Read chunk of data
    chunk_idx = f['z_idx'][0:1000]  # First 1000 tokens
    chunk_val = f['z_val'][0:1000]
    chunk_rev = f['rev_idx'][0:1000]

    # Decode review IDs (stored as bytes)
    review_ids = [r.decode('utf-8') for r in chunk_rev]
```

### Query: Get Activation for Specific Token + Feature

```python
def get_feature_activation(h5_file, token_idx: int, feature_idx: int) -> float:
    """Get activation value for a specific token and feature"""
    active_features = h5_file['z_idx'][token_idx]  # Shape: (32,)
    active_values = h5_file['z_val'][token_idx]    # Shape: (32,)

    # Find if feature is in active set
    mask = active_features == feature_idx
    if np.any(mask):
        return float(active_values[mask][0])
    return 0.0
```

### Query: Get All Tokens Where Feature Activates

```python
def get_feature_activations_chunk(h5_file, feature_idx: int, start: int, end: int):
    """Get activations for a feature across a range of tokens (chunked for memory)"""
    chunk_size = end - start
    activations = np.zeros(chunk_size, dtype=np.float32)

    indices_chunk = h5_file['z_idx'][start:end]  # (chunk_size, 32)
    values_chunk = h5_file['z_val'][start:end]   # (chunk_size, 32)

    for i in range(chunk_size):
        mask = indices_chunk[i] == feature_idx
        if np.any(mask):
            activations[i] = values_chunk[i][mask][0]

    return activations

# Usage: scan for feature 16751
with h5py.File(H5_PATH, 'r') as f:
    total = f['z_idx'].shape[0]
    chunk_size = 100_000

    all_active = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        acts = get_feature_activations_chunk(f, 16751, start, end)
        active_positions = np.where(acts > 0)[0] + start
        all_active.extend(active_positions)
```

---

## NPZ File Structure (Metadata)

### Arrays

| Array | Dtype | Description |
|-------|-------|-------------|
| `review_ids` | object (str) | Review identifiers |
| `texts` | object (str) | Full review text |
| `stars` | int | Star rating (1-5) |
| `useful` | int | Usefulness votes |
| `user_ids` | object (str) | Reviewer identifier |
| `business_ids` | object (str) | Restaurant identifier |

### Loading NPZ Data

```python
import numpy as np
import pandas as pd

# Load as dict-like
data = np.load(METADATA_PATH, allow_pickle=True)
print(data.files)  # ['review_ids', 'texts', 'stars', 'useful', 'user_ids', 'business_ids']

# Create DataFrame
metadata_df = pd.DataFrame({
    'review_id': data['review_ids'],
    'full_text': data['texts'],
    'stars': data['stars'],
    'useful': data['useful'],
    'user_id': data['user_ids'],
    'business_id': data['business_ids']
})

print(f"Total reviews: {len(metadata_df)}")
```

### Create Lookup Dict

```python
review_lookup = {
    str(row['review_id']): str(row['full_text'])
    for _, row in metadata_df.iterrows()
}

# Or faster:
review_lookup = dict(zip(
    data['review_ids'].astype(str),
    data['texts'].astype(str)
))
```

---

## Linking H5 and NPZ

The `rev_idx` array in H5 links tokens to reviews in NPZ:

```python
with h5py.File(H5_PATH, 'r') as f:
    # Get review ID for token 12345
    rev_id_bytes = f['rev_idx'][12345]
    rev_id = rev_id_bytes.decode('utf-8')

    # Look up text in NPZ
    review_text = review_lookup[rev_id]
```

### Token Position Map (Pre-computed)

The `review_token_positions.pkl` file contains a pre-computed mapping from review_id to global token positions:

```python
import pickle

# Load pre-computed map (preferred - fast)
with open('review_token_positions.pkl', 'rb') as f:
    review_token_positions = pickle.load(f)

# review_token_positions['REVIEW_123'] = [token_pos1, token_pos2, ...]
```

To regenerate this file (if H5 changes):
```bash
py -3.12 src/scripts/precompute_token_positions.py
```

The script builds the map by iterating through the H5 file:

```python
from collections import defaultdict

review_token_positions = defaultdict(list)

with h5py.File(H5_PATH, 'r') as f:
    total_tokens = f['z_idx'].shape[0]
    chunk_size = 100_000

    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        chunk_rev_ids = f['rev_idx'][start:end]

        for local_idx, rev_id in enumerate(chunk_rev_ids):
            rev_id_str = rev_id.decode('utf-8') if isinstance(rev_id, bytes) else str(rev_id)
            global_idx = start + local_idx
            review_token_positions[rev_id_str].append(global_idx)
```

---

## Extract Context for Feature Activation

Given a token position where a feature activates, extract the surrounding context:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")

def extract_context(
    global_token_idx: int,
    rev_id: str,
    review_lookup: dict,
    review_token_positions: dict,
    context_before: int = 20,
    context_after: int = 10
):
    """Extract text context around a token position"""

    # Get review text
    review_text = review_lookup.get(rev_id)
    if not review_text:
        return None

    # Find token's position within this review
    positions = review_token_positions.get(rev_id, [])
    if global_token_idx not in positions:
        return None
    local_position = positions.index(global_token_idx)

    # Tokenize review
    tokens = tok(review_text, add_special_tokens=False).input_ids
    token_strings = tok.convert_ids_to_tokens(tokens)

    if local_position >= len(token_strings):
        return None

    # Extract window
    start = max(0, local_position - context_before)
    end = min(len(token_strings), local_position + context_after + 1)

    context_tokens = token_strings[start:end]
    context_str = ' '.join([t.replace('Ġ', ' ') for t in context_tokens])

    return {
        'context': context_str,
        'active_token': token_strings[local_position].replace('Ġ', ' '),
        'position': local_position,
        'review_id': rev_id
    }
```

---

## Memory-Efficient Patterns

### Chunked H5 Reading

```python
# GOOD - Process in chunks
with h5py.File(H5_PATH, 'r') as f:
    total = f['z_idx'].shape[0]
    for start in range(0, total, 100_000):
        end = min(start + 100_000, total)
        chunk = f['z_idx'][start:end]
        # Process chunk...

# BAD - Load entire file
with h5py.File(H5_PATH, 'r') as f:
    all_data = f['z_idx'][:]  # OOM on large files!
```

### Lazy H5 Access

```python
# H5 datasets support lazy slicing
with h5py.File(H5_PATH, 'r') as f:
    # This doesn't load into memory until accessed
    dataset = f['z_idx']
    print(dataset.shape)  # OK

    # Only this loads data
    single_row = dataset[0]  # Loads 1 row
    slice_data = dataset[100:200]  # Loads 100 rows
```

---

## Project-Specific Constants

```python
# Dataset size
N_REVIEWS = 432_248
N_TOKENS = 51_702_947

# SAE configuration
N_LATENTS = 24_576      # Total features (768 * 32)
K_ACTIVE = 32           # Active features per token
D_MODEL = 768           # GPT-2 hidden dimension

# File paths (Modal volume)
VOLUME_MOUNT = "/data"
H5_PATH = f"{VOLUME_MOUNT}/mexican_national_sae_features_e32_k32_lr0_0003-final.h5"
METADATA_PATH = f"{VOLUME_MOUNT}/mexican_national_metadata.npz"
TOKEN_POSITIONS_PATH = f"{VOLUME_MOUNT}/review_token_positions.pkl"
```

---

## Quick Reference

### Get feature activation for token
```python
act = h5['z_val'][token_idx][h5['z_idx'][token_idx] == feature_idx]
```

### Get review text for token
```python
rev_id = h5['rev_idx'][token_idx].decode('utf-8')
text = review_lookup[rev_id]
```

### Find all tokens where feature X activates
```python
for i in range(n_tokens):
    if feature_idx in h5['z_idx'][i]:
        # Token i has feature_idx active
```

### Get top-k strongest activations for a feature
```python
positions, values = [], []
for start in range(0, total, chunk_size):
    acts = get_feature_activations_chunk(h5, feature_idx, start, end)
    active = np.where(acts > 0)[0]
    positions.extend(active + start)
    values.extend(acts[active])

# Sort by value
sorted_idx = np.argsort(values)[::-1][:k]
top_positions = [positions[i] for i in sorted_idx]
```

---

## API Output Format: Sampling Metadata

**All corpus query methods return dictionaries with `sampling` metadata** to document what was actually sampled and potential biases.

### Why Sampling Metadata Matters

- Corpus has 51M+ tokens - we can't scan everything
- Different methods use different sampling strategies
- Results may be biased depending on method used
- You need to know what you're actually measuring

### Sampling Methods

| Method | Strategy | Potential Bias |
|--------|----------|----------------|
| `sequential_from_start` | First N tokens in corpus order | Patterns appearing early in corpus |
| `top_by_activation` | Strongest activations in scanned range | Only clearest/strongest patterns |
| `first_n_then_sort` | First N found, then sorted by strength | Early corpus + strength bias |

### Standard Sampling Fields

```python
{
  "sampling": {
    "method": "sequential_from_start",           # Sampling strategy used
    "description": "First N tokens in corpus order (not random)",
    "tokens_scanned": 100000,                    # How many tokens were examined
    "corpus_coverage": 0.0019                    # Fraction of total corpus
  },
  # ... method-specific data ...
}
```

### Per-Method Output Examples

**get_feature_stats:**
```json
{
  "sampling": {"method": "sequential_from_start", "tokens_scanned": 100000, "corpus_coverage": 0.0019},
  "total_activations": 92,
  "mean_when_active": 0.073,
  "max_activation": 0.282,
  "std_when_active": 0.045,
  "activation_rate": 0.00092
}
```

**get_top_activations:**
```json
{
  "sampling": {
    "method": "top_by_activation",
    "description": "Strongest activations within scanned tokens (not full corpus)",
    "top_k_requested": 10,
    "n_found": 10,
    "activations_scanned": 50000,
    "tokens_scanned": 5000000,
    "corpus_coverage": 0.0967
  },
  "activations": [{"context": "...", "active_token": " my", "activation": 0.229}]
}
```

**get_ngram_patterns:**
```json
{
  "sampling": {
    "method": "top_by_activation",
    "description": "Top N activations by strength (not random)",
    "n_requested": 500,
    "n_found": 487,
    "tokens_scanned": 5000000,
    "corpus_coverage": 0.0967,
    "activation_range": {"min": 0.0312, "max": 0.2297, "mean": 0.0891}
  },
  "n_contexts_analyzed": 487,
  "ngrams": {...}
}
```

### Using Sampling Metadata

```python
result = reader.get_feature_stats.remote(feature_idx, sample_size=100000)

# ALWAYS check what was actually sampled
print(f"Method: {result['sampling']['method']}")
print(f"Coverage: {result['sampling']['corpus_coverage']*100:.1f}% of corpus")
print(f"Tokens scanned: {result['sampling']['tokens_scanned']:,}")

# Interpret results with awareness of sampling bias
if result['sampling']['corpus_coverage'] < 0.01:
    print("WARNING: Less than 1% of corpus sampled - results may not be representative")

# Then use the actual data
print(f"Activation rate: {result['activation_rate']:.6f}")
```
