# Notebook Code Reference

Reference for the SAE interpretability Colab notebook (`fixed_path_new_interpretability_notebook_mexican_national_e32_k32_lr0_0003.py`).

**File size:** 415KB, 11,000+ lines

---

## Quick Reference

### Constants

```python
D_MODEL = 768           # GPT-2 hidden dimension
EXPANSION = 32          # SAE expansion factor
K_ACTIVE = 32           # Top-K sparsity (active features per token)
N_LATENTS = 24576       # Total features (D_MODEL * EXPANSION)
LAYER_IDX = 8           # GPT-2 layer for activation extraction
HOOK = f"blocks.{LAYER_IDX}.hook_resid_pre"
```

### Global Variables (after setup)

| Variable | Type | Description |
|----------|------|-------------|
| `tok` | AutoTokenizer | GPT-2 tokenizer |
| `model` | HookedTransformer | GPT-2 model with hooks |
| `ema_sae` | TopKSAE | Loaded SAE model |
| `metadata_df` | DataFrame | Reviews with columns: review_id, full_text, stars, useful, user_id, business_id |

### Data Flow

```
Text Input
    |
    v
tok.encode() --> token IDs
    |
    v
64-token windows (padded with pad_token)
    |
    v
model.run_with_hooks() --> GPT-2 layer 8 activations
    |
    v
Normalize (mean-center, L2-norm)
    |
    v
ema_sae(activations) --> (x_hat, z)
    |
    v
z[:, feature_idx] --> per-token feature activations
```

---

## Chapter Structure

| Chapter | Lines | Purpose |
|---------|-------|---------|
| 0 | 1-160 | Setup: imports, paths, load data |
| 1 | 161-790 | Text analysis: TopKSAE class, processing functions, visualization |
| 2 | 790-900 | Playground: example usage with sample texts |
| 3 | 900-2000 | H5 data analysis: structure inspection, helper functions |
| 3.4+ | 2000-11000 | Feature exploration: individual feature analysis examples |

---

## Chapter 0: Setup (Lines 1-160)

### Imports

```python
import torch
from torch import nn
import math
import numpy as np
import pandas as pd
import h5py
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from google.colab import drive
```

### Data Loading

```python
# Mount drive and load metadata
metadata_df = pd.DataFrame({
    'review_id': data['review_ids'],
    'full_text': data['texts'],
    'stars': data['stars'],
    'useful': data['useful'],
    'user_id': data['user_ids'],
    'business_id': data['business_ids']
})
```

---

## Chapter 1: Text Analysis (Lines 161-790)

### TopKSAE Class (Lines 168-182)

```python
class TopKSAE(nn.Module):
    def __init__(self, d_model, n_lat, k_act, baseline):
        super().__init__()
        self.W_dec = nn.Parameter(torch.randn(d_model, n_lat) / math.sqrt(d_model))
        self.W_enc = nn.Parameter(self.W_dec.t().clone())      # tied init only
        self.b_pre = nn.Parameter(baseline.clone())            # learnable baseline
        self.k = k_act

    def forward(self, x):
        h = (x - self.b_pre) @ self.W_enc.t()                  # [B, n_lat]
        top_idx = torch.topk(h, self.k, dim=-1).indices
        z = torch.zeros_like(h, dtype=h.dtype)
        z.scatter_(-1, top_idx, h.gather(-1, top_idx))
        x_hat = z @ self.W_dec.t() + self.b_pre
        return x_hat, z
```

**Initialization:**
```python
ema_sae = TopKSAE(D_MODEL, N_LATENTS, K_ACTIVE, torch.zeros(D_MODEL))
ema_sae.load_state_dict(ckpt["ema_sae"])
ema_sae.eval().cuda()
```

### Core Functions

| Function | Line | Signature | Purpose |
|----------|------|-----------|---------|
| `process_new_text` | 217 | `(text, feature_indices_to_track)` | Run text through GPT-2 + SAE pipeline |
| `process_text_for_analysis` | 279 | `(text)` | Wrapper that tracks all 24576 features |
| `get_active_features` | 287 | `(activations, top_k=None)` | Get non-zero features sorted by strength |
| `clean_token_for_display` | 314 | `(token)` | Handle GPT-2 `Ġ` prefix for display |
| `get_feature_statistics` | 318 | `(activations)` | Compute active_count, active_rate, max, mean_when_active |
| `analyze_text_activations` | 338 | `(text, top_k=5)` | Print top activations per token |
| `analyze_multiple_texts` | 374 | `(texts_to_analyze, top_k_per_token=5, show_top_features=20, exclude_features=None)` | Find common features across texts |
| `visualize_feature_activation` | 464 | `(text, feature_indices, show_overlap=True, activation_threshold=0.01, show_values_for_feature=None)` | HTML color-coded visualization |
| `compare_text_activations` | 696 | `(text1, text2, top_k=20, aggregation='max')` | Find features that differ between texts |

### process_new_text (Line 217)

Primary function for analyzing arbitrary text. **Note: Requires feature list, no default.**

```python
def process_new_text(text, feature_indices_to_track):
    """
    Args:
        text: Input string
        feature_indices_to_track: List of feature indices to extract (REQUIRED)

    Returns:
        dict with keys:
            'text': original input text
            'tokens': list of token strings
            'all_activations': full z tensor (n_tokens, N_LATENTS)
            'tracked_features': dict {'feature_{idx}': array of activations}
            'n_tokens': number of tokens
    """
```

**Usage:**
```python
result = process_new_text("The food was amazing!", [16751, 208])
tokens = result['tokens']  # ['The', 'Ġfood', 'Ġwas', 'Ġamazing', '!']
acts = result['tracked_features']['feature_16751']  # array of activations
all_z = result['all_activations']  # (n_tokens, 24576) tensor
```

### visualize_feature_activation (Line 464)

Creates HTML visualization with color-coded tokens. Supports multiple features and multiple texts.

```python
def visualize_feature_activation(text, feature_indices, show_overlap=True,
                                  activation_threshold=0.01, show_values_for_feature=None):
    """
    Args:
        text: Single string OR list of strings
        feature_indices: Single int OR list of feature indices
        show_overlap: If True, show gradient when multiple features active
        activation_threshold: Minimum activation to show color (default 0.01)
        show_values_for_feature: Feature index to show numeric values for (None = don't show)

    Returns:
        Displays HTML in Colab (no return value)
    """
```

**Usage:**
```python
# Single feature, single text
visualize_feature_activation("Great tacos!", 16751)

# Multiple features, single text
visualize_feature_activation("Great tacos!", [16751, 208, 3223])

# Single feature, multiple texts
visualize_feature_activation(["Text 1", "Text 2"], 16751)

# Show activation values
visualize_feature_activation("Great tacos!", [16751], show_values_for_feature=16751)
```

### analyze_multiple_texts (Line 374)

Find features that appear across multiple texts.

```python
def analyze_multiple_texts(texts_to_analyze, top_k_per_token=5,
                           show_top_features=20, exclude_features=None):
    """
    Args:
        texts_to_analyze: List of strings
        top_k_per_token: Number of top features to track per token
        show_top_features: Number of most common features to display
        exclude_features: List of feature indices to exclude

    Returns:
        tuple: (sorted_features, all_results)
            sorted_features: list of (feature_idx, count) sorted by frequency
            all_results: list of result dicts from analyze_text_activations
    """
```

### compare_text_activations (Line 696)

Compare which features differ between two texts.

```python
def compare_text_activations(text1, text2, top_k=20, aggregation='max'):
    """
    Args:
        text1: First text
        text2: Second text
        top_k: Number of top differences to show
        aggregation: 'max' or 'mean' for aggregating across tokens

    Returns:
        dict with keys:
            'differences': array of (agg_acts1 - agg_acts2) per feature
            'sorted_indices': feature indices sorted by absolute difference
            'agg_acts1': aggregated activations for text1
            'agg_acts2': aggregated activations for text2
            'unique_to_text1': features active only in text1
            'unique_to_text2': features active only in text2
    """
```

---

## Chapter 3: H5 Data Analysis (Lines 900-2000)

### H5 Structure Functions

| Function | Line | Purpose |
|----------|------|---------|
| `inspect_h5_structure` | 907 | Print H5 file structure and shapes |
| `get_review_text` | 1002 | Look up review text by ID from DataFrame |
| `get_feature_activation_for_token` | 1044 | Get activation from sparse format |
| `get_feature_activations_chunk` | 1060 | Chunked feature extraction |
| `load_review_texts_from_df` | 1084 | Create review_id -> text lookup dict |
| `load_combined_data` | 1172 | Combine H5 + metadata DataFrame |
| `extract_feature_data_adapted` | 1282 | Extract feature contexts from H5 |
| `analyze_feature_tokens_with_text` | 1475 | Comprehensive feature analysis with plots |
| `analyze_feature_to_tokens_csv` | 1743 | Export feature data to CSV |

### get_feature_activation_for_token (Line 1044)

```python
def get_feature_activation_for_token(h5_sae, token_idx, feature_idx):
    """
    Get activation value for specific token and feature from sparse H5 format.

    Args:
        h5_sae: Open h5py.File object
        token_idx: Global token index
        feature_idx: Feature index (0-24575)

    Returns:
        float: Activation value (0.0 if feature not active)
    """
```

### extract_feature_data_adapted (Line 1282)

Extract example contexts where a feature activates.

```python
def extract_feature_data_adapted(feature_idx, h5_sae_path, metadata_df,
                                  max_active_samples=50_000,
                                  n_inactive_samples=2_000,
                                  context_before=20,
                                  context_after=10):
    """
    Args:
        feature_idx: Feature to analyze
        h5_sae_path: Path to H5 file
        metadata_df: DataFrame with review texts
        max_active_samples: Maximum active contexts to collect
        n_inactive_samples: Number of inactive contexts for comparison
        context_before/after: Tokens of context around activation

    Returns:
        dict with keys:
            'feature_idx': the feature analyzed
            'active_contexts': list of context dicts
            'active_reviews': dict of review data
            'inactive_contexts': list of inactive context strings
            'token_counts': dict of token -> count
            'stats': dict with collection statistics
    """
```

### analyze_feature_tokens_with_text (Line 1475)

Full analysis of a feature with visualizations.

```python
def analyze_feature_tokens_with_text(feature_idx, h5_sae_path, metadata_df,
                                      top_k=20, max_samples=50000):
    """
    Comprehensive feature analysis:
    - Finds all tokens where feature activates
    - Creates 4 matplotlib plots (frequency, max activation, bigrams, trigrams)
    - Prints detailed statistics

    Args:
        feature_idx: Feature to analyze
        h5_sae_path: Path to H5 file
        metadata_df: DataFrame with review texts
        top_k: Number of top tokens/ngrams to show
        max_samples: Maximum activations to analyze

    Returns:
        dict with keys:
            'token_counts', 'max_activations', 'avg_activations',
            'top_by_count', 'top_by_max', 'top_bigrams', 'top_trigrams',
            'samples_analyzed', 'bigram_examples', 'trigram_examples'
    """
```

### analyze_feature_to_tokens_csv (Line 1743)

Export feature analysis to CSV for external analysis.

```python
def analyze_feature_to_tokens_csv(feature_idx, h5_path, metadata_df,
                                   max_samples=100000, output_filename=None):
    """
    Args:
        feature_idx: Feature ID to analyze
        h5_path: Path to H5 file
        metadata_df: DataFrame with review metadata
        max_samples: Maximum activations to analyze
        output_filename: Optional filename (default: feature_{idx}_tokens.csv)

    Creates CSV with columns:
        token_id, token_string, count, frequency,
        average_activation, total_activation, example_values
    """
```

---

## Common Workflows

### Workflow 1: Analyze Arbitrary Text

```python
# Process with all features
result = process_text_for_analysis("This restaurant has amazing tacos!")
for i, token in enumerate(result['tokens']):
    acts = result['all_activations'][i]
    top_5 = np.argsort(acts)[-5:][::-1]
    print(f"{token}: {[(idx, acts[idx]) for idx in top_5 if acts[idx] > 0]}")

# Or process with specific features only
result = process_new_text("Great tacos!", [16751, 208, 3223])
print(result['tracked_features'])
```

### Workflow 2: Visualize Feature Activations

```python
# Single feature visualization
visualize_feature_activation("The service was terrible", 16751)

# Multiple features with values shown
visualize_feature_activation(
    "The service was terrible",
    [16751, 3223],
    show_values_for_feature=16751
)
```

### Workflow 3: Compare Two Texts

```python
results = compare_text_activations(
    "The food was absolutely amazing!",
    "The food was okay.",
    top_k=20,
    aggregation='max'
)
print("Features unique to text1:", results['unique_to_text1'][:10])
print("Features unique to text2:", results['unique_to_text2'][:10])
```

### Workflow 4: Find Common Features

```python
texts = [
    "Best tacos I've ever had!",
    "These are the best tacos in town!",
    "Hands down the best tacos anywhere!"
]
sorted_features, all_results = analyze_multiple_texts(
    texts,
    top_k_per_token=5,
    show_top_features=20
)
# sorted_features = [(feature_idx, count), ...] sorted by frequency
```

### Workflow 5: Extract Feature Examples from H5

```python
feature_data = extract_feature_data_adapted(
    feature_idx=16751,
    h5_sae_path=LOCAL_H5_SAE_PATH,
    metadata_df=metadata_df,
    max_active_samples=10_000,
    context_before=20,
    context_after=10
)
for ctx in feature_data['active_contexts'][:10]:
    print(f"[{ctx['active_token']}] {ctx['context']}")
```

### Workflow 6: Generate Feature Report with Plots

```python
token_stats = analyze_feature_tokens_with_text(
    feature_idx=16751,
    h5_sae_path=LOCAL_H5_SAE_PATH,
    metadata_df=metadata_df,
    top_k=20,
    max_samples=50000
)
# Displays 4 matplotlib charts and prints statistics
```

### Workflow 7: Export to CSV

```python
analyze_feature_to_tokens_csv(
    feature_idx=16751,
    h5_path=LOCAL_H5_SAE_PATH,
    metadata_df=metadata_df,
    max_samples=100000,
    output_filename="feature_16751_analysis.csv"
)
```

---

## File Paths (Colab)

```python
# Google Drive paths (original locations)
METADATA_CSV_PATH = "/content/drive/My Drive/WORK/Bottom Up Psychometric Coding/Yelp Data/mexican_national_metadata.npz"
H5_SAE_PATH = "/content/drive/My Drive/WORK/Bottom Up Psychometric Coding/Yelp Data/mexican_national_sae_features_e32_k32_lr0_0003-final.h5"
MODEL_PATH = "/content/drive/My Drive/WORK/Bottom Up Psychometric Coding/sae_e32_k32_lr0.0003-final.pt"

# Local paths (copied for performance)
LOCAL_H5_SAE_PATH = "/content/sae_features.h5"
LOCAL_METADATA_PATH = "/content/metadata.npz"
LOCAL_MODEL_PATH = "/content/sae_e32_k32_lr0.0003-final.pt"
```

---

## Notes

- **Chapters 2000+** contain individual feature exploration examples, not reusable code
- **Token normalization**: GPT-2 uses `Ġ` prefix for tokens following whitespace
- **Window size**: Text is processed in 64-token windows, padded with pad_token if needed
- **Activation normalization**: Mean-centered and L2-normalized before SAE
- **Sparse format**: H5 uses z_idx (feature indices) and z_val (values) with K=32 active per token
