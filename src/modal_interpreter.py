"""Modal backend for SAE interpretability analysis.

This module provides the compute backend that runs on Modal with GPU support.
It exposes methods for analyzing SAE features and processing text through
the GPT-2 → SAE pipeline.

Usage:
    # Deploy
    modal deploy src/modal_interpreter.py

    # Run tests
    modal run src/modal_interpreter.py::test_interpreter
"""

import modal
from collections import defaultdict

# Model constants (duplicated from config.py for Modal container access)
D_MODEL = 768              # GPT-2 hidden dimension
EXPANSION = 32             # SAE expansion factor
K_ACTIVE = 32              # Top-K sparsity
N_LATENTS = 24576          # D_MODEL * EXPANSION
LAYER_IDX = 8              # GPT-2 layer to hook
HOOK = f"blocks.{LAYER_IDX}.hook_resid_pre"
WINDOW_SIZE = 64           # Token window size

# Modal Volume paths
VOLUME_MOUNT = "/data"
H5_PATH = f"{VOLUME_MOUNT}/mexican_national_sae_features_e32_k32_lr0_0003-final.h5"
METADATA_PATH = f"{VOLUME_MOUNT}/mexican_national_metadata.npz"
SAE_CHECKPOINT_PATH = f"{VOLUME_MOUNT}/sae_e32_k32_lr0.0003-final.pt"

# Modal app configuration
app = modal.App("sae-interpretability")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformer-lens",
        "transformers",
        "numpy",
        "h5py",
        "pandas",
        "tqdm"
    )
)

# Data volume
data_volume = modal.Volume.from_name("sae-data", create_if_missing=False)


@app.cls(
    image=image,
    gpu="T4",
    volumes={VOLUME_MOUNT: data_volume},
    scaledown_window=300,  # Keep warm for 5 minutes
    timeout=600,  # 10 minute timeout per method call
)
class SAEInterpreter:
    """
    Persistent service for SAE interpretability analysis.

    Uses @modal.cls to keep containers warm - models loaded once on startup,
    reused across multiple method calls.

    Methods:
        process_text: Run text through GPT-2 → SAE pipeline
        get_top_tokens: Get most common tokens activating a feature
        get_feature_contexts: Get example contexts where feature fires
        get_feature_stats: Get activation statistics for a feature
    """

    @modal.enter()
    def setup(self):
        """Initialize models and data on container startup."""
        import warnings
        # Suppress transformer-lens deprecation warning about torch_dtype
        warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated")

        import torch
        import math
        from torch import nn
        from transformer_lens import HookedTransformer
        from transformers import AutoTokenizer
        import numpy as np
        import pandas as pd
        import h5py
        from tqdm import tqdm

        print("Loading tokenizer...")
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        if self.tok.pad_token is None:
            self.tok.add_special_tokens({'pad_token': self.tok.eos_token})

        print("Loading GPT-2...")
        self.model = HookedTransformer.from_pretrained(
            "gpt2", device="cuda", dtype="float32"
        )

        print("Loading SAE checkpoint...")
        self.sae = self._load_sae(SAE_CHECKPOINT_PATH)

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

        # Create review lookup with validation
        self.review_lookup = {}
        for _, row in self.metadata_df.iterrows():
            review_id = str(row['review_id'])
            text = str(row['full_text'])
            # Handle empty/NaN text
            if pd.isna(row['full_text']) or text == 'nan' or text.strip() == '':
                text = ""
            self.review_lookup[review_id] = text

        print(f"Loaded {len(self.metadata_df)} reviews")

        print("Opening H5 file...")
        self.h5 = h5py.File(H5_PATH, "r")
        self.total_tokens = self.h5['z_idx'].shape[0]
        print(f"H5 file has {self.total_tokens:,} tokens")

        print("Building review token position map (this takes ~30-60s)...")
        self.review_token_positions = self._build_review_token_map()
        print(f"Built position map for {len(self.review_token_positions):,} reviews")

        print("Setup complete!")

    def _load_sae(self, path: str):
        """Load SAE from checkpoint."""
        import torch
        import math
        from torch import nn

        class TopKSAE(nn.Module):
            def __init__(self, d_model, n_lat, k_act, baseline):
                super().__init__()
                self.W_dec = nn.Parameter(torch.randn(d_model, n_lat) / math.sqrt(d_model))
                self.W_enc = nn.Parameter(self.W_dec.t().clone())
                self.b_pre = nn.Parameter(baseline.clone())
                self.k = k_act

            def forward(self, x):
                h = (x - self.b_pre) @ self.W_enc.t()
                top_idx = torch.topk(h, self.k, dim=-1).indices
                z = torch.zeros_like(h, dtype=h.dtype)
                z.scatter_(-1, top_idx, h.gather(-1, top_idx))
                x_hat = z @ self.W_dec.t() + self.b_pre
                return x_hat, z

        ckpt = torch.load(path, map_location="cpu")
        sae = TopKSAE(D_MODEL, N_LATENTS, K_ACTIVE, torch.zeros(D_MODEL))
        sae.load_state_dict(ckpt["ema_sae"])
        sae.eval()
        return sae.cuda()

    def _build_review_token_map(self) -> dict:
        """Pre-compute mapping from review_id to token positions."""
        from tqdm import tqdm

        review_token_positions = defaultdict(list)
        chunk_size = 100_000

        for start in tqdm(range(0, self.total_tokens, chunk_size), desc="Building map"):
            end = min(start + chunk_size, self.total_tokens)
            chunk_rev_ids = self.h5['rev_idx'][start:end]

            for local_idx, rev_id in enumerate(chunk_rev_ids):
                if isinstance(rev_id, bytes):
                    rev_id_str = rev_id.decode('utf-8')
                else:
                    rev_id_str = str(rev_id)
                global_idx = start + local_idx
                review_token_positions[rev_id_str].append(global_idx)

        return dict(review_token_positions)

    def _get_feature_activations_chunk(self, feature_idx: int, start: int, end: int):
        """Get activations for a feature across a token range from H5 sparse format."""
        import numpy as np

        chunk_size = end - start
        activations = np.zeros(chunk_size, dtype=np.float32)

        indices_chunk = self.h5['z_idx'][start:end]
        values_chunk = self.h5['z_val'][start:end]

        for i in range(chunk_size):
            mask = indices_chunk[i] == feature_idx
            if mask.any():
                activations[i] = values_chunk[i][mask][0]

        return activations

    @modal.method()
    def process_text(self, text: str) -> dict:
        """
        Run text through GPT-2 → SAE pipeline.

        Args:
            text: Input text to analyze

        Returns:
            dict with:
                tokens: List of token strings
                n_tokens: Number of tokens
                top_features_per_token: For each token, list of (feature_idx, activation)
                feature_activations: Full activation matrix as nested list
        """
        import torch
        import numpy as np

        # Tokenize
        ids = self.tok(text, add_special_tokens=False).input_ids
        if not ids:
            return {"tokens": [], "n_tokens": 0, "top_features_per_token": [], "feature_activations": {}}

        # Create 64-token windows
        windows = []
        for i in range(0, len(ids), WINDOW_SIZE):
            window = ids[i:i + WINDOW_SIZE]
            if len(window) < WINDOW_SIZE:
                window = window + [self.tok.pad_token_id] * (WINDOW_SIZE - len(window))
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

        # Get tokens (only up to actual token count, excluding padding)
        tokens = self.tok.convert_ids_to_tokens(ids)
        z_np = z.cpu().numpy()[:len(tokens)]

        # Get top features per token
        top_features_per_token = []
        for token_idx in range(len(tokens)):
            activations = z_np[token_idx]
            non_zero_mask = activations > 0
            non_zero_features = np.where(non_zero_mask)[0]
            non_zero_values = activations[non_zero_mask]

            if len(non_zero_features) > 0:
                sorted_idx = np.argsort(non_zero_values)[::-1][:10]
                top_features = [
                    {"feature_idx": int(non_zero_features[i]), "activation": float(non_zero_values[i])}
                    for i in sorted_idx
                ]
            else:
                top_features = []

            top_features_per_token.append(top_features)

        # Build feature_activations dict for specific feature lookups
        feature_activations = {}
        # Include any feature that has non-zero activation somewhere
        active_features = set()
        for token_idx in range(len(tokens)):
            for feat in top_features_per_token[token_idx]:
                active_features.add(feat["feature_idx"])

        for feat_idx in active_features:
            feature_activations[str(feat_idx)] = z_np[:, feat_idx].tolist()

        return {
            "tokens": [t.replace('Ġ', ' ') for t in tokens],
            "n_tokens": len(tokens),
            "top_features_per_token": top_features_per_token,
            "feature_activations": feature_activations
        }

    @modal.method()
    def get_top_tokens(self, feature_idx: int, top_k: int = 20, max_samples: int = 50000) -> list:
        """
        Get the most common tokens that activate a feature.

        Args:
            feature_idx: Feature index (0-24575)
            top_k: Number of top tokens to return
            max_samples: Max active tokens to scan

        Returns:
            List of dicts: [{token, count, mean_activation}, ...]
        """
        import numpy as np
        from tqdm import tqdm

        token_stats = defaultdict(lambda: {"count": 0, "total_activation": 0.0})
        samples_collected = 0
        chunk_size = 100_000

        for start in tqdm(range(0, self.total_tokens, chunk_size), desc=f"Scanning feature {feature_idx}"):
            if samples_collected >= max_samples:
                break

            end = min(start + chunk_size, self.total_tokens)
            chunk_acts = self._get_feature_activations_chunk(feature_idx, start, end)
            chunk_rev_ids = self.h5['rev_idx'][start:end]

            active_indices = np.where(chunk_acts > 0)[0]

            for local_idx in active_indices:
                if samples_collected >= max_samples:
                    break

                global_idx = start + local_idx
                activation = float(chunk_acts[local_idx])
                rev_id = chunk_rev_ids[local_idx]
                rev_id_str = rev_id.decode('utf-8') if isinstance(rev_id, bytes) else str(rev_id)

                # Get review text
                review_text = self.review_lookup.get(rev_id_str, "")
                if not review_text:
                    continue

                # Find position in review
                positions = self.review_token_positions.get(rev_id_str, [])
                if global_idx not in positions:
                    continue
                local_position = positions.index(global_idx)

                # Tokenize review to get the actual token
                tokens = self.tok(review_text, add_special_tokens=False).input_ids
                token_strings = self.tok.convert_ids_to_tokens(tokens)

                if local_position >= len(token_strings):
                    continue

                token = token_strings[local_position].replace('Ġ', ' ')
                token_stats[token]["count"] += 1
                token_stats[token]["total_activation"] += activation
                samples_collected += 1

        # Sort by count and return top_k
        sorted_tokens = sorted(
            token_stats.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:top_k]

        return [
            {
                "token": token,
                "count": stats["count"],
                "mean_activation": stats["total_activation"] / stats["count"]
            }
            for token, stats in sorted_tokens
        ]

    @modal.method()
    def get_feature_contexts(
        self,
        feature_idx: int,
        n_samples: int = 20,
        context_before: int = 15,
        context_after: int = 10
    ) -> list:
        """
        Get example contexts where a feature activates.

        Args:
            feature_idx: Feature index (0-24575)
            n_samples: Number of example contexts to return
            context_before: Tokens to include before the active token
            context_after: Tokens to include after the active token

        Returns:
            List of dicts: [{context, active_token, activation, review_id, position}, ...]
        """
        import numpy as np
        from tqdm import tqdm

        contexts = []
        chunk_size = 100_000

        for start in tqdm(range(0, self.total_tokens, chunk_size), desc=f"Finding contexts for {feature_idx}"):
            if len(contexts) >= n_samples:
                break

            end = min(start + chunk_size, self.total_tokens)
            chunk_acts = self._get_feature_activations_chunk(feature_idx, start, end)
            chunk_rev_ids = self.h5['rev_idx'][start:end]

            active_indices = np.where(chunk_acts > 0)[0]

            for local_idx in active_indices:
                if len(contexts) >= n_samples:
                    break

                global_idx = start + local_idx
                activation = float(chunk_acts[local_idx])
                rev_id = chunk_rev_ids[local_idx]
                rev_id_str = rev_id.decode('utf-8') if isinstance(rev_id, bytes) else str(rev_id)

                # Get review text
                review_text = self.review_lookup.get(rev_id_str, "")
                if not review_text:
                    continue

                # Find position in review
                positions = self.review_token_positions.get(rev_id_str, [])
                if global_idx not in positions:
                    continue
                local_position = positions.index(global_idx)

                # Tokenize review
                tokens = self.tok(review_text, add_special_tokens=False).input_ids
                token_strings = self.tok.convert_ids_to_tokens(tokens)

                if local_position >= len(token_strings):
                    continue

                # Extract context window
                ctx_start = max(0, local_position - context_before)
                ctx_end = min(len(token_strings), local_position + context_after + 1)
                context_tokens = token_strings[ctx_start:ctx_end]

                # Mark the active token position in context
                active_pos_in_context = local_position - ctx_start

                # Format context string with marker
                context_parts = []
                for i, t in enumerate(context_tokens):
                    clean_token = t.replace('Ġ', ' ')
                    if i == active_pos_in_context:
                        context_parts.append(f"**{clean_token}**")
                    else:
                        context_parts.append(clean_token)

                context_str = ''.join(context_parts)
                active_token = token_strings[local_position].replace('Ġ', ' ')

                contexts.append({
                    "context": context_str,
                    "active_token": active_token,
                    "activation": activation,
                    "review_id": rev_id_str,
                    "position": local_position
                })

        # Sort by activation strength
        contexts.sort(key=lambda x: x["activation"], reverse=True)
        return contexts

    @modal.method()
    def get_feature_stats(self, feature_idx: int, sample_size: int = 500000) -> dict:
        """
        Get statistics about a feature's activations.

        Args:
            feature_idx: Feature index (0-24575)
            sample_size: Number of tokens to sample for statistics

        Returns:
            dict with: total_activations, mean_when_active, max_activation, activation_rate
        """
        import numpy as np
        from tqdm import tqdm

        all_activations = []
        active_count = 0
        tokens_scanned = 0
        chunk_size = 100_000

        for start in tqdm(range(0, min(self.total_tokens, sample_size), chunk_size), desc=f"Stats for {feature_idx}"):
            end = min(start + chunk_size, self.total_tokens, sample_size)
            chunk_acts = self._get_feature_activations_chunk(feature_idx, start, end)

            active_mask = chunk_acts > 0
            active_count += np.sum(active_mask)
            all_activations.extend(chunk_acts[active_mask].tolist())
            tokens_scanned += len(chunk_acts)

        if not all_activations:
            return {
                "total_activations": 0,
                "mean_when_active": 0.0,
                "max_activation": 0.0,
                "activation_rate": 0.0,
                "tokens_scanned": tokens_scanned
            }

        return {
            "total_activations": int(active_count),
            "mean_when_active": float(np.mean(all_activations)),
            "max_activation": float(np.max(all_activations)),
            "activation_rate": float(active_count / tokens_scanned),
            "tokens_scanned": tokens_scanned
        }


# Test entrypoint
@app.local_entrypoint()
def test_interpreter():
    """Test the SAEInterpreter methods."""
    interpreter = SAEInterpreter()

    print("\n=== Testing process_text ===")
    result = interpreter.process_text.remote("Why in the world would anyone eat here?")
    print(f"Tokens: {result['tokens']}")
    print(f"Top features for first token: {result['top_features_per_token'][0][:3]}")

    print("\n=== Testing get_feature_stats ===")
    stats = interpreter.get_feature_stats.remote(16751, sample_size=100000)
    print(f"Feature 16751 stats: {stats}")

    print("\n=== Testing get_top_tokens ===")
    top_tokens = interpreter.get_top_tokens.remote(16751, top_k=10, max_samples=10000)
    print(f"Top tokens for feature 16751:")
    for t in top_tokens[:5]:
        print(f"  '{t['token']}': {t['count']} times, mean act={t['mean_activation']:.3f}")

    print("\n=== Testing get_feature_contexts ===")
    contexts = interpreter.get_feature_contexts.remote(16751, n_samples=5)
    print(f"Example contexts for feature 16751:")
    for ctx in contexts[:3]:
        print(f"  [{ctx['activation']:.2f}] {ctx['context'][:80]}...")

    print("\n=== All tests passed! ===")
