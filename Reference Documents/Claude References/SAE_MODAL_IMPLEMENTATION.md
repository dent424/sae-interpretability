# SAE Interpretability Project - Modal Implementation Guide

Complete reference for converting the SAE interpretability notebook to Modal.

## Project Configuration

```python
import modal

app = modal.App("sae-interpretability")

# Model constants (from original notebook)
D_MODEL = 768          # GPT-2 hidden dimension
EXPANSION = 32         # SAE expansion factor
K_ACTIVE = 32          # Top-K sparsity
N_LATENTS = 24576      # D_MODEL * EXPANSION
LAYER_IDX = 8          # GPT-2 layer to hook
HOOK = f"blocks.{LAYER_IDX}.hook_resid_pre"

# Volume paths
VOLUME_MOUNT = "/data"
H5_PATH = f"{VOLUME_MOUNT}/mexican_national_sae_features_e32_k32_lr0_0003-final.h5"
METADATA_PATH = f"{VOLUME_MOUNT}/mexican_national_metadata.npz"
MODEL_PATH = f"{VOLUME_MOUNT}/sae_e32_k32_lr0.0003-final.pt"
```

## Container Image

```python
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformer-lens",
        "transformers",
        "numpy",
        "h5py",
        "pandas",
        "matplotlib",
        "tqdm"
    )
)

data_volume = modal.Volume.from_name("sae-data", create_if_missing=True)
```

## Upload Data to Volume

```bash
# Upload your data files
modal volume put sae-data mexican_national_sae_features_e32_k32_lr0_0003-final.h5 /
modal volume put sae-data mexican_national_metadata.npz /
modal volume put sae-data sae_e32_k32_lr0.0003-final.pt /

# Verify uploads
modal volume ls sae-data
```

---

## Complete TopKSAE Implementation

```python
import torch
from torch import nn
import math

class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder for GPT-2 residual stream"""

    def __init__(self, d_model: int, n_lat: int, k_act: int, baseline: torch.Tensor):
        super().__init__()
        self.W_dec = nn.Parameter(torch.randn(d_model, n_lat) / math.sqrt(d_model))
        self.W_enc = nn.Parameter(self.W_dec.t().clone())  # tied init only
        self.b_pre = nn.Parameter(baseline.clone())        # learnable baseline
        self.k = k_act
        self.d_model = d_model
        self.n_lat = n_lat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input activations [batch, d_model]
        Returns:
            x_hat: Reconstructed activations [batch, d_model]
            z: Sparse latent activations [batch, n_lat]
        """
        h = (x - self.b_pre) @ self.W_enc.t()              # [B, n_lat]
        top_idx = torch.topk(h, self.k, dim=-1).indices    # [B, k]
        z = torch.zeros_like(h, dtype=h.dtype)
        z.scatter_(-1, top_idx, h.gather(-1, top_idx))     # sparse activations
        x_hat = z @ self.W_dec.t() + self.b_pre            # reconstruction
        return x_hat, z

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda"):
        """Load SAE from checkpoint file"""
        ckpt = torch.load(path, map_location="cpu")
        sae = cls(D_MODEL, N_LATENTS, K_ACTIVE, torch.zeros(D_MODEL))
        sae.load_state_dict(ckpt["ema_sae"])
        sae.eval()
        return sae.to(device)
```

---

## Core SAE Interpreter Class

```python
@app.cls(image=image, gpu="T4", volumes={VOLUME_MOUNT: data_volume})
class SAEInterpreter:
    """
    Persistent service that loads GPT-2 + SAE once, handles many requests.
    Use @app.cls for warm containers - avoids reloading models per request.
    """

    @modal.enter()
    def setup(self):
        """Runs once when container starts"""
        import torch
        from transformer_lens import HookedTransformer
        from transformers import AutoTokenizer
        import numpy as np
        import pandas as pd

        print("Loading tokenizer...")
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        if self.tok.pad_token is None:
            self.tok.add_special_tokens({'pad_token': self.tok.eos_token})

        print("Loading GPT-2...")
        self.model = HookedTransformer.from_pretrained(
            "gpt2", device="cuda", dtype="float32"
        )

        print("Loading SAE...")
        self.sae = TopKSAE.from_checkpoint(MODEL_PATH, device="cuda")

        print("Loading metadata...")
        data = np.load(METADATA_PATH, allow_pickle=True)
        self.metadata_df = pd.DataFrame({
            'review_id': data['review_ids'],
            'full_text': data['texts'],
            'stars': data['stars'],
            'useful': data['useful'],
            'user_id': data['user_ids'],
            'business_id': data['business_ids']
        })
        self.review_lookup = {
            str(row['review_id']): str(row['full_text'])
            for _, row in self.metadata_df.iterrows()
        }

        print(f"Loaded {len(self.metadata_df)} reviews")

    @modal.method()
    def process_text(self, text: str, feature_indices: list[int] = None):
        """
        Process arbitrary text through GPT-2 → SAE pipeline.
        Returns token-level activations for specified features.
        """
        import torch
        import numpy as np

        # Tokenize
        ids = self.tok(text, add_special_tokens=False).input_ids
        if not ids:
            return None

        # Create 64-token windows (matching original training)
        windows = []
        for i in range(0, len(ids), 64):
            window = ids[i:i+64]
            if len(window) < 64:
                window = window + [self.tok.pad_token_id] * (64 - len(window))
            windows.append(window)

        windows_tensor = torch.tensor(windows).cuda()

        # Capture GPT-2 activations via hook
        all_activations = []
        def save_acts(act, hook):
            flat = act.detach().reshape(-1, D_MODEL)
            flat = flat - flat.mean(-1, keepdim=True)
            flat = flat / flat.norm(dim=-1, keepdim=True)
            all_activations.append(flat)

        self.model.run_with_hooks(
            windows_tensor,
            fwd_hooks=[(HOOK, save_acts)],
            return_type=None
        )

        gpt2_acts = torch.cat(all_activations, dim=0)

        # Run through SAE
        with torch.no_grad():
            x_hat, z = self.sae(gpt2_acts)

        # Get tokens
        tokens = self.tok.convert_ids_to_tokens(ids)

        # Extract specific features or return all
        z_np = z.cpu().numpy()
        if feature_indices:
            tracked = {
                idx: z_np[:len(tokens), idx].tolist()
                for idx in feature_indices
            }
        else:
            tracked = None

        return {
            "tokens": tokens,
            "n_tokens": len(tokens),
            "tracked_features": tracked,
            "all_activations": z_np[:len(tokens)].tolist() if not feature_indices else None
        }

    @modal.method()
    def get_active_features_for_token(self, text: str, token_idx: int, top_k: int = 10):
        """Get top-k active features for a specific token position"""
        import numpy as np

        result = self.process_text(text)
        if result is None or token_idx >= result['n_tokens']:
            return None

        activations = np.array(result['all_activations'][token_idx])
        non_zero_mask = activations > 0
        non_zero_features = np.where(non_zero_mask)[0]
        non_zero_values = activations[non_zero_mask]

        if len(non_zero_features) == 0:
            return {"token": result['tokens'][token_idx], "features": [], "values": []}

        sorted_idx = np.argsort(non_zero_values)[::-1][:top_k]
        return {
            "token": result['tokens'][token_idx],
            "features": non_zero_features[sorted_idx].tolist(),
            "values": non_zero_values[sorted_idx].tolist()
        }

    @modal.method()
    def compare_texts(self, text1: str, text2: str, top_k: int = 20, aggregation: str = 'max'):
        """Compare feature activations between two texts"""
        import numpy as np

        result1 = self.process_text(text1)
        result2 = self.process_text(text2)

        acts1 = np.array(result1['all_activations'])
        acts2 = np.array(result2['all_activations'])

        if aggregation == 'max':
            agg1 = np.max(acts1, axis=0)
            agg2 = np.max(acts2, axis=0)
        else:
            agg1 = np.mean(acts1, axis=0)
            agg2 = np.mean(acts2, axis=0)

        diff = agg1 - agg2
        sorted_indices = np.argsort(np.abs(diff))[::-1][:top_k]

        return {
            "top_differences": [
                {
                    "feature": int(idx),
                    "text1_activation": float(agg1[idx]),
                    "text2_activation": float(agg2[idx]),
                    "difference": float(diff[idx])
                }
                for idx in sorted_indices
            ],
            "unique_to_text1": np.where((agg1 > 0) & (agg2 == 0))[0].tolist()[:20],
            "unique_to_text2": np.where((agg1 == 0) & (agg2 > 0))[0].tolist()[:20]
        }
```

---

## H5 Feature Extraction Functions

### Efficient Chunked H5 Access

```python
def get_feature_activations_chunk(h5_file, feature_idx: int, start: int, end: int):
    """
    Get activations for a specific feature across a range of tokens.
    Handles SPARSE format efficiently.
    """
    import numpy as np

    chunk_size = end - start
    activations = np.zeros(chunk_size, dtype=np.float32)

    indices_chunk = h5_file['z_idx'][start:end]  # [chunk_size, k]
    values_chunk = h5_file['z_val'][start:end]   # [chunk_size, k]

    for i in range(chunk_size):
        mask = indices_chunk[i] == feature_idx
        if np.any(mask):
            activations[i] = values_chunk[i][mask][0]

    return activations


@app.function(image=image, volumes={VOLUME_MOUNT: data_volume}, timeout=3600)
def extract_feature_contexts(
    feature_idx: int,
    max_samples: int = 10000,
    context_before: int = 20,
    context_after: int = 10
):
    """
    Extract example contexts where a feature activates.
    Memory-efficient chunked processing for large H5 files.
    """
    import h5py
    import numpy as np
    from transformers import AutoTokenizer
    from collections import defaultdict

    tok = AutoTokenizer.from_pretrained("gpt2")

    # Load metadata
    data = np.load(METADATA_PATH, allow_pickle=True)
    review_lookup = {
        str(data['review_ids'][i]): str(data['texts'][i])
        for i in range(len(data['review_ids']))
    }

    active_contexts = []
    token_counts = defaultdict(int)

    with h5py.File(H5_PATH, 'r') as h5:
        total_tokens = h5['z_idx'].shape[0]
        chunk_size = 100_000

        # Build review token position map
        print("Building position map...")
        review_token_positions = defaultdict(list)
        for start in range(0, total_tokens, chunk_size):
            end = min(start + chunk_size, total_tokens)
            chunk_rev_ids = h5['rev_idx'][start:end]

            for local_idx, rev_id in enumerate(chunk_rev_ids):
                rev_id_str = rev_id.decode('utf-8') if isinstance(rev_id, bytes) else str(rev_id)
                review_token_positions[rev_id_str].append(start + local_idx)

        # Scan for active tokens
        print(f"Scanning for feature {feature_idx}...")
        for start in range(0, total_tokens, chunk_size):
            if len(active_contexts) >= max_samples:
                break

            end = min(start + chunk_size, total_tokens)
            chunk_acts = get_feature_activations_chunk(h5, feature_idx, start, end)
            chunk_rev_ids = h5['rev_idx'][start:end]

            active_indices = np.where(chunk_acts > 0)[0]

            for local_idx in active_indices:
                if len(active_contexts) >= max_samples:
                    break

                global_idx = start + local_idx
                activation = float(chunk_acts[local_idx])
                rev_id = chunk_rev_ids[local_idx]
                rev_id_str = rev_id.decode('utf-8') if isinstance(rev_id, bytes) else str(rev_id)

                # Get review text
                review_text = review_lookup.get(rev_id_str)
                if not review_text:
                    continue

                # Find position in review
                positions = review_token_positions.get(rev_id_str, [])
                if global_idx not in positions:
                    continue
                local_position = positions.index(global_idx)

                # Tokenize review
                tokens = tok(review_text, add_special_tokens=False).input_ids
                token_strings = tok.convert_ids_to_tokens(tokens)

                if local_position >= len(token_strings):
                    continue

                # Extract context window
                ctx_start = max(0, local_position - context_before)
                ctx_end = min(len(token_strings), local_position + context_after + 1)
                context_tokens = token_strings[ctx_start:ctx_end]
                context_str = ' '.join([t.replace('Ġ', ' ') for t in context_tokens])

                active_token = token_strings[local_position].replace('Ġ', ' ')
                token_counts[active_token] += 1

                active_contexts.append({
                    'context': context_str,
                    'active_token': active_token,
                    'activation': activation,
                    'review_id': rev_id_str,
                    'position': local_position
                })

    # Sort by activation strength
    active_contexts.sort(key=lambda x: x['activation'], reverse=True)

    return {
        'feature_idx': feature_idx,
        'contexts': active_contexts,
        'token_counts': dict(token_counts),
        'total_found': len(active_contexts)
    }
```

---

## Parallel Analysis Patterns

### Analyze Multiple Features in Parallel

```python
@app.function(image=image, volumes={VOLUME_MOUNT: data_volume}, timeout=1800)
def analyze_single_feature(feature_idx: int, max_samples: int = 5000):
    """Analyze a single feature - designed for .map() parallelization"""
    result = extract_feature_contexts(feature_idx, max_samples)
    return {
        'feature': feature_idx,
        'total_activations': result['total_found'],
        'top_tokens': sorted(result['token_counts'].items(), key=lambda x: x[1], reverse=True)[:10],
        'top_contexts': result['contexts'][:5]
    }


@app.local_entrypoint()
def analyze_features_parallel():
    """Analyze multiple features in parallel using .map()"""
    feature_ids = [11328, 16751, 14292, 20379, 23016]

    print(f"Analyzing {len(feature_ids)} features in parallel...")
    results = list(analyze_single_feature.map(feature_ids))

    for r in results:
        print(f"\nFeature {r['feature']}:")
        print(f"  Total activations: {r['total_activations']}")
        print(f"  Top tokens: {r['top_tokens'][:5]}")
```

### Batch Processing Reviews

```python
@app.function(image=image, gpu="T4", volumes={VOLUME_MOUNT: data_volume})
def process_review_batch(review_indices: list[int], feature_idx: int):
    """Process a batch of reviews from the dataset"""
    import numpy as np

    data = np.load(METADATA_PATH, allow_pickle=True)
    interpreter = SAEInterpreter()

    results = []
    for idx in review_indices:
        text = str(data['texts'][idx])
        result = interpreter.process_text.local(text, [feature_idx])
        if result:
            max_activation = max(result['tracked_features'][feature_idx])
            results.append({
                'review_idx': idx,
                'max_activation': max_activation,
                'n_tokens': result['n_tokens']
            })

    return results


@app.local_entrypoint()
def batch_analysis():
    """Process reviews in parallel batches"""
    n_reviews = 10000
    batch_size = 100
    feature_idx = 16751

    batches = [
        list(range(i, min(i + batch_size, n_reviews)))
        for i in range(0, n_reviews, batch_size)
    ]

    # Process all batches in parallel
    all_results = []
    for batch_results in process_review_batch.map(
        batches,
        kwargs={"feature_idx": feature_idx}
    ):
        all_results.extend(batch_results)

    print(f"Processed {len(all_results)} reviews")
```

---

## Web Endpoints

```python
@app.function(image=image, gpu="T4", volumes={VOLUME_MOUNT: data_volume})
@modal.web_endpoint(method="POST")
def analyze_text_api(text: str, features: list[int] = None):
    """HTTP API for text analysis"""
    interpreter = SAEInterpreter()
    result = interpreter.process_text.local(text, features)
    return {"success": True, "result": result}


@app.function(image=image, volumes={VOLUME_MOUNT: data_volume})
@modal.web_endpoint(method="GET")
def get_feature_info(feature_idx: int, n_examples: int = 10):
    """Get information about a specific feature"""
    result = extract_feature_contexts.local(feature_idx, max_samples=n_examples)
    return {
        "feature": feature_idx,
        "examples": result['contexts'],
        "top_tokens": result['token_counts']
    }
```

Deploy: `modal deploy script.py`

---

## Return Type Guidelines

Modal serializes return values - use JSON-compatible types:

```python
# GOOD - Returns JSON-serializable types
return {
    "tokens": tokens,                           # list[str]
    "activations": activations.tolist(),        # list[float] from numpy
    "count": int(count),                        # int from numpy.int64
    "mean": float(mean)                         # float from numpy.float32
}

# BAD - Returns non-serializable types
return {
    "activations": activations,     # numpy array - will fail
    "df": dataframe,                # pandas DataFrame - will fail
    "tensor": torch_tensor          # torch.Tensor - will fail
}
```

---

## Memory Management for Large H5 Files

```python
# GOOD - Chunked reading, processes in batches
with h5py.File(H5_PATH, 'r') as f:
    total = f['z_idx'].shape[0]
    for start in range(0, total, 100_000):
        end = min(start + 100_000, total)
        chunk = f['z_idx'][start:end]
        # process chunk...

# BAD - Loads entire file into memory
with h5py.File(H5_PATH, 'r') as f:
    all_data = f['z_idx'][:]  # May crash on large files!
```

---

## CLI Commands Reference

```bash
# Run local entrypoint
modal run sae_modal.py

# Run specific function
modal run sae_modal.py::analyze_features_parallel

# Deploy web endpoints
modal deploy sae_modal.py

# Check logs
modal app logs sae-interpretability

# Volume management
modal volume ls sae-data
modal volume put sae-data file.h5 /file.h5
modal volume get sae-data /file.h5 ./local_file.h5
```

---

## Debugging Tips

1. **Test locally first**: Use `.local()` instead of `.remote()` during development
   ```python
   result = interpreter.process_text.local(text, features)  # runs in current process
   ```

2. **Print statements appear in logs**: Use `modal app logs` to see them

3. **Memory issues**: Reduce chunk_size, use generators, avoid loading full arrays

4. **Timeout errors**: Increase `timeout=` parameter (max 86400 for functions)

5. **GPU not found**: Ensure function has `gpu="T4"` or similar in decorator

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Examples](https://modal.com/docs/examples)
- [Modal Discord](https://discord.gg/modal)
