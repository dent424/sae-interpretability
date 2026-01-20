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
TOKEN_POSITIONS_PATH = f"{VOLUME_MOUNT}/review_token_positions.pkl"

# Modal app configuration
app = modal.App("sae-interpretability")

# Container image with all dependencies (CPU - no torch GPU needed)
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers",
        "numpy",
        "h5py",
        "pandas",
        "tqdm"
    )
)

# Container image with GPU dependencies
gpu_image = (
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


# ============================================================================
# CPU-Only Class: Database/H5 Operations
# ============================================================================

@app.cls(
    image=cpu_image,
    volumes={VOLUME_MOUNT: data_volume},
    scaledown_window=300,  # Keep warm for 5 minutes
    timeout=600,
)
class SAEDataReader:
    """
    CPU-only service for reading precomputed SAE activations from H5.

    Use this for corpus analysis (scanning existing activations).
    Does NOT load GPT-2 or SAE models - just reads from H5 sparse format.
    """

    @modal.enter()
    def setup(self):
        """Initialize data access on container startup."""
        from transformers import AutoTokenizer
        import numpy as np
        import pandas as pd
        import h5py

        print("Loading tokenizer...")
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        if self.tok.pad_token is None:
            self.tok.add_special_tokens({'pad_token': self.tok.eos_token})

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
            if pd.isna(row['full_text']) or text == 'nan' or text.strip() == '':
                text = ""
            self.review_lookup[review_id] = text

        print(f"Loaded {len(self.metadata_df)} reviews")

        print("Opening H5 file...")
        self.h5 = h5py.File(H5_PATH, "r")
        self.total_tokens = self.h5['z_idx'].shape[0]
        print(f"H5 file has {self.total_tokens:,} tokens")

        print("Loading review token position map...")
        self.review_token_positions = self._load_token_positions()
        print(f"Loaded position map for {len(self.review_token_positions):,} reviews")

        print("Setup complete!")

    def _load_token_positions(self) -> dict:
        """Load token positions from pickle, or build if not available."""
        import pickle
        import os

        if os.path.exists(TOKEN_POSITIONS_PATH):
            print(f"Loading from {TOKEN_POSITIONS_PATH}...")
            with open(TOKEN_POSITIONS_PATH, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"Pickle not found, building map (this takes ~30-60s)...")
            return self._build_review_token_map()

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
    def get_top_tokens(self, feature_idx: int, top_k: int = 20, max_samples: int = 50000, quiet: bool = False) -> dict:
        """Get the most common tokens that activate a feature.

        Note: Samples activations sequentially from the start of the corpus,
        not randomly. Results may be biased toward patterns appearing early
        in the dataset.
        """
        import numpy as np
        from tqdm import tqdm

        token_stats = defaultdict(lambda: {"count": 0, "total_activation": 0.0})
        samples_collected = 0
        tokens_scanned = 0
        chunk_size = 100_000

        for start in tqdm(range(0, self.total_tokens, chunk_size), desc=f"Scanning feature {feature_idx}", disable=quiet):
            if samples_collected >= max_samples:
                break

            end = min(start + chunk_size, self.total_tokens)
            chunk_acts = self._get_feature_activations_chunk(feature_idx, start, end)
            chunk_rev_ids = self.h5['rev_idx'][start:end]
            tokens_scanned += (end - start)

            active_indices = np.where(chunk_acts > 0)[0]

            for local_idx in active_indices:
                if samples_collected >= max_samples:
                    break

                global_idx = start + local_idx
                activation = float(chunk_acts[local_idx])
                rev_id = chunk_rev_ids[local_idx]
                rev_id_str = rev_id.decode('utf-8') if isinstance(rev_id, bytes) else str(rev_id)

                review_text = self.review_lookup.get(rev_id_str, "")
                if not review_text:
                    continue

                positions = self.review_token_positions.get(rev_id_str, [])
                if global_idx not in positions:
                    continue
                local_position = positions.index(global_idx)

                tokens = self.tok(review_text, add_special_tokens=False).input_ids
                token_strings = self.tok.convert_ids_to_tokens(tokens)

                if local_position >= len(token_strings):
                    continue

                token = token_strings[local_position].replace('Ġ', ' ')
                token_stats[token]["count"] += 1
                token_stats[token]["total_activation"] += activation
                samples_collected += 1

        sorted_tokens = sorted(
            token_stats.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:top_k]

        tokens_list = [
            {
                "token": token,
                "count": stats["count"],
                "mean_activation": stats["total_activation"] / stats["count"]
            }
            for token, stats in sorted_tokens
        ]

        return {
            "sampling": {
                "method": "sequential_from_start",
                "description": "First N activations in corpus order (not random)",
                "activations_collected": samples_collected,
                "tokens_scanned": tokens_scanned,
                "corpus_coverage": round(tokens_scanned / self.total_tokens, 4)
            },
            "top_tokens": tokens_list
        }

    @modal.method()
    def get_feature_contexts(
        self,
        feature_idx: int,
        n_samples: int = 20,
        context_before: int = 15,
        context_after: int = 10,
        quiet: bool = False
    ) -> dict:
        """Get example contexts where a feature activates.

        Note: Takes first N activations found in corpus order, then sorts
        by activation strength. Not a random sample or globally top activations.
        """
        import numpy as np
        from tqdm import tqdm

        contexts = []
        tokens_scanned = 0
        chunk_size = 100_000

        for start in tqdm(range(0, self.total_tokens, chunk_size), desc=f"Finding contexts for {feature_idx}", disable=quiet):
            if len(contexts) >= n_samples:
                break

            end = min(start + chunk_size, self.total_tokens)
            chunk_acts = self._get_feature_activations_chunk(feature_idx, start, end)
            chunk_rev_ids = self.h5['rev_idx'][start:end]
            tokens_scanned += (end - start)

            active_indices = np.where(chunk_acts > 0)[0]

            for local_idx in active_indices:
                if len(contexts) >= n_samples:
                    break

                global_idx = start + local_idx
                activation = float(chunk_acts[local_idx])
                rev_id = chunk_rev_ids[local_idx]
                rev_id_str = rev_id.decode('utf-8') if isinstance(rev_id, bytes) else str(rev_id)

                review_text = self.review_lookup.get(rev_id_str, "")
                if not review_text:
                    continue

                positions = self.review_token_positions.get(rev_id_str, [])
                if global_idx not in positions:
                    continue
                local_position = positions.index(global_idx)

                tokens = self.tok(review_text, add_special_tokens=False).input_ids
                token_strings = self.tok.convert_ids_to_tokens(tokens)

                if local_position >= len(token_strings):
                    continue

                ctx_start = max(0, local_position - context_before)
                ctx_end = min(len(token_strings), local_position + context_after + 1)
                context_tokens = token_strings[ctx_start:ctx_end]

                active_pos_in_context = local_position - ctx_start

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

        contexts.sort(key=lambda x: x["activation"], reverse=True)

        return {
            "sampling": {
                "method": "first_n_then_sort",
                "description": "First N activations found in corpus order, sorted by strength",
                "n_requested": n_samples,
                "n_found": len(contexts),
                "tokens_scanned": tokens_scanned
            },
            "contexts": contexts
        }

    @modal.method()
    def get_top_activations(
        self,
        feature_idx: int,
        top_k: int = 50,
        max_scan: int = 50000,
        context_before: int = 15,
        context_after: int = 10,
        quiet: bool = False
    ) -> dict:
        """Find globally strongest activations for a feature.

        Note: Finds top activations within first max_scan tokens of corpus,
        not the entire corpus. For rare features, this may miss stronger
        activations later in the corpus.
        """
        import numpy as np
        from tqdm import tqdm
        import heapq

        candidates = []
        activations_scanned = 0
        tokens_scanned = 0
        chunk_size = 100_000

        for start in tqdm(range(0, self.total_tokens, chunk_size), desc=f"Scanning feature {feature_idx}", disable=quiet):
            if activations_scanned >= max_scan:
                break

            end = min(start + chunk_size, self.total_tokens)
            chunk_acts = self._get_feature_activations_chunk(feature_idx, start, end)
            chunk_rev_ids = self.h5['rev_idx'][start:end]
            tokens_scanned += (end - start)

            active_indices = np.where(chunk_acts > 0)[0]

            for local_idx in active_indices:
                if activations_scanned >= max_scan:
                    break

                global_idx = start + local_idx
                activation = float(chunk_acts[local_idx])
                rev_id = chunk_rev_ids[local_idx]
                rev_id_str = rev_id.decode('utf-8') if isinstance(rev_id, bytes) else str(rev_id)

                if len(candidates) < top_k:
                    heapq.heappush(candidates, (activation, global_idx, rev_id_str))
                elif activation > candidates[0][0]:
                    heapq.heapreplace(candidates, (activation, global_idx, rev_id_str))

                activations_scanned += 1

        candidates.sort(reverse=True)

        contexts = []
        for activation, global_idx, rev_id_str in candidates:
            review_text = self.review_lookup.get(rev_id_str, "")
            if not review_text:
                continue

            positions = self.review_token_positions.get(rev_id_str, [])
            if global_idx not in positions:
                continue
            local_position = positions.index(global_idx)

            tokens = self.tok(review_text, add_special_tokens=False).input_ids
            token_strings = self.tok.convert_ids_to_tokens(tokens)

            if local_position >= len(token_strings):
                continue

            ctx_start = max(0, local_position - context_before)
            ctx_end = min(len(token_strings), local_position + context_after + 1)
            context_tokens = token_strings[ctx_start:ctx_end]

            active_pos_in_context = local_position - ctx_start

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

        return {
            "sampling": {
                "method": "top_by_activation",
                "description": "Strongest activations within scanned tokens (not full corpus)",
                "top_k_requested": top_k,
                "n_found": len(contexts),
                "activations_scanned": activations_scanned,
                "tokens_scanned": tokens_scanned,
                "corpus_coverage": round(tokens_scanned / self.total_tokens, 4)
            },
            "activations": contexts
        }

    @modal.method()
    def get_activation_distribution(self, feature_idx: int, sample_size: int = 100000, quiet: bool = False) -> dict:
        """Get detailed distribution statistics for a feature's activations.

        Computes percentiles, skewness, and kurtosis for non-zero activations.
        Uses numpy approximations to avoid scipy dependency.

        Args:
            feature_idx: Feature to analyze
            sample_size: Number of tokens to scan
            quiet: Suppress progress bars

        Returns:
            dict with percentiles (p10, p25, p50, p75, p90, p95, p99),
            skewness, kurtosis, and sampling metadata
        """
        import numpy as np
        from tqdm import tqdm

        all_activations = []
        tokens_scanned = 0
        chunk_size = 100_000

        for start in tqdm(range(0, min(self.total_tokens, sample_size), chunk_size),
                          desc=f"Distribution for {feature_idx}", disable=quiet):
            end = min(start + chunk_size, self.total_tokens, sample_size)
            chunk_acts = self._get_feature_activations_chunk(feature_idx, start, end)

            active_mask = chunk_acts > 0
            all_activations.extend(chunk_acts[active_mask].tolist())
            tokens_scanned += len(chunk_acts)

        if not all_activations:
            return {
                "sampling": {
                    "method": "sequential_from_start",
                    "description": "First N tokens in corpus order",
                    "tokens_scanned": tokens_scanned,
                    "n_activations": 0,
                    "corpus_coverage": round(tokens_scanned / self.total_tokens, 4)
                },
                "percentiles": {},
                "skewness": None,
                "kurtosis": None
            }

        acts = np.array(all_activations)

        # Compute percentiles
        percentile_values = [10, 25, 50, 75, 90, 95, 99]
        percentiles = {f"p{p}": round(float(np.percentile(acts, p)), 4) for p in percentile_values}

        # Compute skewness: E[(X-μ)³] / σ³
        mean = np.mean(acts)
        std = np.std(acts)
        if std > 0:
            skewness = float(np.mean(((acts - mean) / std) ** 3))
            kurtosis = float(np.mean(((acts - mean) / std) ** 4) - 3)  # Excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0

        return {
            "sampling": {
                "method": "sequential_from_start",
                "description": "First N tokens in corpus order",
                "tokens_scanned": tokens_scanned,
                "n_activations": len(all_activations),
                "corpus_coverage": round(tokens_scanned / self.total_tokens, 4)
            },
            "percentiles": percentiles,
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "min": round(float(np.min(acts)), 4),
            "max": round(float(np.max(acts)), 4),
            "mean": round(float(mean), 4),
            "std": round(float(std), 4)
        }

    @modal.method()
    def get_coactivated_features(self, feature_idx: int, top_k: int = 20, max_samples: int = 10000, quiet: bool = False) -> dict:
        """Get features that frequently co-activate with the target feature.

        For tokens where target feature fires, collects the other 31 active features
        and returns frequency-sorted list of co-occurring features.

        Args:
            feature_idx: Target feature to analyze
            top_k: Number of top co-activated features to return
            max_samples: Max number of activations to sample
            quiet: Suppress progress bars

        Returns:
            dict with list of co-activated features sorted by frequency
        """
        import numpy as np
        from collections import Counter
        from tqdm import tqdm

        coactivation_counts = Counter()
        samples_collected = 0
        tokens_scanned = 0
        chunk_size = 100_000

        for start in tqdm(range(0, self.total_tokens, chunk_size),
                          desc=f"Co-activation for {feature_idx}", disable=quiet):
            if samples_collected >= max_samples:
                break

            end = min(start + chunk_size, self.total_tokens)
            chunk_acts = self._get_feature_activations_chunk(feature_idx, start, end)
            tokens_scanned += (end - start)

            # Find positions where target feature is active
            active_indices = np.where(chunk_acts > 0)[0]

            for local_idx in active_indices:
                if samples_collected >= max_samples:
                    break

                global_idx = start + local_idx

                # Read the 32 active feature indices for this token
                z_idx = self.h5['z_idx'][global_idx]

                # Count all features except the target
                for feat_idx in z_idx:
                    if feat_idx != feature_idx:
                        coactivation_counts[int(feat_idx)] += 1

                samples_collected += 1

        # Get top co-activated features
        top_features = coactivation_counts.most_common(top_k)

        return {
            "sampling": {
                "method": "sequential_from_start",
                "description": "First N activations in corpus order",
                "activations_sampled": samples_collected,
                "tokens_scanned": tokens_scanned,
                "corpus_coverage": round(tokens_scanned / self.total_tokens, 4)
            },
            "coactivated_features": [
                {
                    "feature_idx": feat_idx,
                    "count": count,
                    "percent": round(100.0 * count / samples_collected, 2) if samples_collected > 0 else 0
                }
                for feat_idx, count in top_features
            ]
        }

    @modal.method()
    def get_position_distribution(self, feature_idx: int, n_bins: int = 5, max_samples: int = 10000, quiet: bool = False) -> dict:
        """Get distribution of where in reviews the feature tends to fire.

        Computes relative position (0-1) of each activation within its review
        and bins into early/middle/late categories.

        Args:
            feature_idx: Feature to analyze
            n_bins: Number of position bins (default 5 = 20% each)
            max_samples: Max activations to sample
            quiet: Suppress progress bars

        Returns:
            dict with position bins, mean_position, and std_position
        """
        import numpy as np
        from tqdm import tqdm

        relative_positions = []
        samples_collected = 0
        tokens_scanned = 0
        chunk_size = 100_000

        for start in tqdm(range(0, self.total_tokens, chunk_size),
                          desc=f"Position distribution for {feature_idx}", disable=quiet):
            if samples_collected >= max_samples:
                break

            end = min(start + chunk_size, self.total_tokens)
            chunk_acts = self._get_feature_activations_chunk(feature_idx, start, end)
            chunk_rev_ids = self.h5['rev_idx'][start:end]
            tokens_scanned += (end - start)

            active_indices = np.where(chunk_acts > 0)[0]

            for local_idx in active_indices:
                if samples_collected >= max_samples:
                    break

                global_idx = start + local_idx
                rev_id = chunk_rev_ids[local_idx]
                rev_id_str = rev_id.decode('utf-8') if isinstance(rev_id, bytes) else str(rev_id)

                # Get all token positions for this review
                positions = self.review_token_positions.get(rev_id_str, [])
                if not positions or global_idx not in positions:
                    continue

                # Find local position within review
                local_position = positions.index(global_idx)
                review_length = len(positions)

                # Compute relative position (0 = start, 1 = end)
                relative_pos = local_position / max(review_length - 1, 1)
                relative_positions.append(relative_pos)
                samples_collected += 1

        if not relative_positions:
            return {
                "sampling": {
                    "method": "sequential_from_start",
                    "description": "First N activations in corpus order",
                    "activations_sampled": 0,
                    "tokens_scanned": tokens_scanned,
                    "corpus_coverage": round(tokens_scanned / self.total_tokens, 4)
                },
                "bins": [],
                "mean_position": None,
                "std_position": None
            }

        positions_arr = np.array(relative_positions)

        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_labels = ["early", "early-mid", "middle", "late-mid", "late"] if n_bins == 5 else [f"bin_{i}" for i in range(n_bins)]

        bins = []
        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:  # Include right edge for last bin
                mask = (positions_arr >= low) & (positions_arr <= high)
            else:
                mask = (positions_arr >= low) & (positions_arr < high)
            count = int(np.sum(mask))
            bins.append({
                "range": f"{int(low*100)}-{int(high*100)}%",
                "label": bin_labels[i] if i < len(bin_labels) else f"bin_{i}",
                "count": count,
                "percent": round(100.0 * count / len(relative_positions), 2)
            })

        return {
            "sampling": {
                "method": "sequential_from_start",
                "description": "First N activations in corpus order",
                "activations_sampled": len(relative_positions),
                "tokens_scanned": tokens_scanned,
                "corpus_coverage": round(tokens_scanned / self.total_tokens, 4)
            },
            "bins": bins,
            "mean_position": round(float(np.mean(positions_arr)), 4),
            "std_position": round(float(np.std(positions_arr)), 4)
        }

    @modal.method()
    def get_directional_ngrams(
        self,
        feature_idx: int,
        ngram_sizes: list = None,
        top_k: int = 15,
        max_samples: int = 500,
        max_scan: int = 100000,
        quiet: bool = False
    ) -> dict:
        """Extract n-grams that appear BEFORE vs AFTER the active token.

        Separates left context from right context to identify directional patterns:
        - left_ngrams: tokens immediately preceding the active token
        - right_ngrams: tokens immediately following the active token

        This helps identify if a feature fires at specific syntactic boundaries,
        e.g., "after punctuation" or "before certain words".
        """
        from collections import Counter
        import numpy as np

        if ngram_sizes is None:
            ngram_sizes = [2, 3]

        # Get top activations with context
        context_window = max(ngram_sizes) + 1  # Need enough context for largest ngram
        top_acts_result = self.get_top_activations.local(
            feature_idx,
            top_k=max_samples,
            max_scan=max_scan,
            context_before=context_window,
            context_after=context_window,
            quiet=quiet
        )

        top_acts = top_acts_result.get("activations", [])
        source_sampling = top_acts_result.get("sampling", {})

        if not top_acts:
            empty_ngrams = {}
            for n in ngram_sizes:
                empty_ngrams[f"left_{n}grams"] = []
                empty_ngrams[f"right_{n}grams"] = []
            return {
                "sampling": {
                    "method": "top_by_activation",
                    "description": "Top N activations by strength",
                    "n_requested": max_samples,
                    "n_found": 0,
                    "tokens_scanned": source_sampling.get("tokens_scanned", max_scan),
                    "corpus_coverage": source_sampling.get("corpus_coverage", 0)
                },
                "n_contexts_analyzed": 0,
                **empty_ngrams
            }

        # Initialize counters for left and right ngrams
        left_counters = {n: Counter() for n in ngram_sizes}
        right_counters = {n: Counter() for n in ngram_sizes}

        for act_info in top_acts:
            context = act_info.get("context", "")

            # Find the active token position (marked with **)
            # The context has format: "tokens **active** more tokens"
            if "**" not in context:
                continue

            # Split by the active token marker
            parts = context.split("**")
            if len(parts) < 3:
                continue

            left_text = parts[0]  # Text before active token
            right_text = parts[2] if len(parts) > 2 else ""  # Text after active token

            # Tokenize left and right contexts
            if left_text.strip():
                left_tokens = self.tok(left_text, add_special_tokens=False).input_ids
                left_token_strings = [t.replace('Ġ', ' ') for t in self.tok.convert_ids_to_tokens(left_tokens)]
            else:
                left_token_strings = []

            if right_text.strip():
                right_tokens = self.tok(right_text, add_special_tokens=False).input_ids
                right_token_strings = [t.replace('Ġ', ' ') for t in self.tok.convert_ids_to_tokens(right_tokens)]
            else:
                right_token_strings = []

            # Extract ngrams from left context (take from the END, closest to active token)
            for n in ngram_sizes:
                if len(left_token_strings) >= n:
                    # Take the last n tokens (immediately before active token)
                    ngram = tuple(left_token_strings[-n:])
                    left_counters[n][ngram] += 1

            # Extract ngrams from right context (take from the START, closest to active token)
            for n in ngram_sizes:
                if len(right_token_strings) >= n:
                    # Take the first n tokens (immediately after active token)
                    ngram = tuple(right_token_strings[:n])
                    right_counters[n][ngram] += 1

        n_contexts = len(top_acts)
        result = {
            "sampling": {
                "method": "top_by_activation",
                "description": "Top N activations by strength",
                "n_requested": max_samples,
                "n_found": n_contexts,
                "tokens_scanned": source_sampling.get("tokens_scanned", max_scan),
                "corpus_coverage": source_sampling.get("corpus_coverage", 0)
            },
            "n_contexts_analyzed": n_contexts
        }

        # Add left ngrams
        for n in ngram_sizes:
            top_left = left_counters[n].most_common(top_k)
            result[f"left_{n}grams"] = [
                {
                    "ngram": list(ngram),
                    "ngram_str": "".join(ngram),
                    "count": count,
                    "percent": round(100.0 * count / n_contexts, 1) if n_contexts > 0 else 0
                }
                for ngram, count in top_left
            ]

        # Add right ngrams
        for n in ngram_sizes:
            top_right = right_counters[n].most_common(top_k)
            result[f"right_{n}grams"] = [
                {
                    "ngram": list(ngram),
                    "ngram_str": "".join(ngram),
                    "count": count,
                    "percent": round(100.0 * count / n_contexts, 1) if n_contexts > 0 else 0
                }
                for ngram, count in top_right
            ]

        return result

    @modal.method()
    def get_token_stability(
        self,
        feature_idx: int,
        top_k_tokens: int = 20,
        max_samples: int = 5000,
        quiet: bool = False
    ) -> dict:
        """Analyze activation variance by token to identify context-dependent features.

        For each unique token where the feature fires, computes:
        - mean activation across all occurrences
        - std deviation of activation
        - coefficient of variation (CV = std / mean)

        High CV indicates the feature is context-dependent (activates differently
        in different contexts). Low CV indicates stable activation regardless of context.
        """
        from collections import defaultdict
        import numpy as np
        from tqdm import tqdm

        # Collect activations grouped by token
        token_activations = defaultdict(list)
        tokens_scanned = 0
        activations_sampled = 0
        chunk_size = 100_000

        for start in tqdm(range(0, self.total_tokens, chunk_size), desc=f"Scanning feature {feature_idx}", disable=quiet):
            if activations_sampled >= max_samples:
                break

            end = min(start + chunk_size, self.total_tokens)
            chunk_acts = self._get_feature_activations_chunk(feature_idx, start, end)
            chunk_rev_ids = self.h5['rev_idx'][start:end]
            tokens_scanned += (end - start)

            active_indices = np.where(chunk_acts > 0)[0]

            for local_idx in active_indices:
                if activations_sampled >= max_samples:
                    break

                global_idx = start + local_idx
                activation = float(chunk_acts[local_idx])
                rev_id = chunk_rev_ids[local_idx]
                rev_id_str = rev_id.decode('utf-8') if isinstance(rev_id, bytes) else str(rev_id)

                # Get the token at this position
                review_text = self.review_lookup.get(rev_id_str, "")
                if not review_text:
                    continue

                positions = self.review_token_positions.get(rev_id_str, [])
                if global_idx not in positions:
                    continue
                local_position = positions.index(global_idx)

                tokens = self.tok(review_text, add_special_tokens=False).input_ids
                if local_position >= len(tokens):
                    continue

                token_id = tokens[local_position]
                token_str = self.tok.convert_ids_to_tokens([token_id])[0].replace('Ġ', ' ')

                token_activations[token_str].append(activation)
                activations_sampled += 1

        if not token_activations:
            return {
                "sampling": {
                    "method": "sequential_from_start",
                    "description": "First N activations in corpus order",
                    "activations_sampled": 0,
                    "tokens_scanned": tokens_scanned,
                    "corpus_coverage": round(tokens_scanned / self.total_tokens, 4)
                },
                "tokens": []
            }

        # Compute statistics for each token
        token_stats = []
        for token, acts in token_activations.items():
            if len(acts) < 2:  # Need at least 2 samples for meaningful stats
                continue

            acts_arr = np.array(acts)
            mean_act = float(np.mean(acts_arr))
            std_act = float(np.std(acts_arr))
            cv = std_act / mean_act if mean_act > 0 else 0.0

            token_stats.append({
                "token": token,
                "mean_act": round(mean_act, 4),
                "std_act": round(std_act, 4),
                "n_samples": len(acts),
                "cv": round(cv, 4)
            })

        # Sort by frequency (n_samples) descending, take top_k
        token_stats.sort(key=lambda x: x["n_samples"], reverse=True)
        token_stats = token_stats[:top_k_tokens]

        return {
            "sampling": {
                "method": "sequential_from_start",
                "description": "First N activations in corpus order",
                "activations_sampled": activations_sampled,
                "tokens_scanned": tokens_scanned,
                "corpus_coverage": round(tokens_scanned / self.total_tokens, 4),
                "unique_tokens_found": len(token_activations)
            },
            "tokens": token_stats
        }

    @modal.method()
    def get_top_token_contexts(
        self,
        feature_idx: int,
        top_k_tokens: int = 10,
        contexts_per_token: int = 2,
        max_scan: int = 50000,
        context_size: int = 15,
        quiet: bool = False
    ) -> dict:
        """Get highest-activation contexts for each of the top tokens.

        For each of the top K tokens (by frequency), collects M contexts showing
        the highest activations. Each context comes from a different review.

        Each context includes feature_activations array showing the target
        feature's activation at every token position in the context window.
        """
        from collections import defaultdict
        import numpy as np
        from tqdm import tqdm
        import heapq

        # First pass: collect all activations grouped by token
        # Store (activation, global_idx, rev_id) for each token
        token_candidates = defaultdict(list)
        tokens_scanned = 0
        activations_found = 0
        chunk_size = 100_000

        for start in tqdm(range(0, self.total_tokens, chunk_size), desc=f"Scanning feature {feature_idx}", disable=quiet):
            if tokens_scanned >= max_scan:
                break

            end = min(start + chunk_size, self.total_tokens)
            chunk_acts = self._get_feature_activations_chunk(feature_idx, start, end)
            chunk_rev_ids = self.h5['rev_idx'][start:end]
            tokens_scanned += (end - start)

            active_indices = np.where(chunk_acts > 0)[0]

            for local_idx in active_indices:
                global_idx = start + local_idx
                activation = float(chunk_acts[local_idx])
                rev_id = chunk_rev_ids[local_idx]
                rev_id_str = rev_id.decode('utf-8') if isinstance(rev_id, bytes) else str(rev_id)

                # Get the token at this position
                review_text = self.review_lookup.get(rev_id_str, "")
                if not review_text:
                    continue

                positions = self.review_token_positions.get(rev_id_str, [])
                if global_idx not in positions:
                    continue
                local_position = positions.index(global_idx)

                tokens = self.tok(review_text, add_special_tokens=False).input_ids
                if local_position >= len(tokens):
                    continue

                token_id = tokens[local_position]
                token_str = self.tok.convert_ids_to_tokens([token_id])[0].replace('Ġ', ' ')

                token_candidates[token_str].append((activation, global_idx, rev_id_str, local_position))
                activations_found += 1

        if not token_candidates:
            return {
                "sampling": {
                    "method": "sequential_from_start",
                    "description": "First N tokens in corpus order, top activations per token",
                    "tokens_scanned": tokens_scanned,
                    "corpus_coverage": round(tokens_scanned / self.total_tokens, 4),
                    "activations_found": 0
                },
                "tokens": []
            }

        # Get top K tokens by frequency
        token_counts = [(tok, len(cands)) for tok, cands in token_candidates.items()]
        token_counts.sort(key=lambda x: x[1], reverse=True)
        top_tokens = token_counts[:top_k_tokens]

        result_tokens = []
        for token_str, count in top_tokens:
            candidates = token_candidates[token_str]

            # Sort by activation strength, then select top from different reviews
            candidates.sort(key=lambda x: x[0], reverse=True)

            # Collect contexts from different reviews
            selected_contexts = []
            seen_reviews = set()

            for activation, global_idx, rev_id_str, local_position in candidates:
                if rev_id_str in seen_reviews:
                    continue
                if len(selected_contexts) >= contexts_per_token:
                    break

                seen_reviews.add(rev_id_str)

                # Get context
                review_text = self.review_lookup.get(rev_id_str, "")
                if not review_text:
                    continue

                tokens = self.tok(review_text, add_special_tokens=False).input_ids
                token_strings = self.tok.convert_ids_to_tokens(tokens)

                ctx_start = max(0, local_position - context_size)
                ctx_end = min(len(token_strings), local_position + context_size + 1)
                context_tokens = token_strings[ctx_start:ctx_end]
                active_pos_in_context = local_position - ctx_start

                # Build context string with active token marked
                context_parts = []
                for i, t in enumerate(context_tokens):
                    clean_token = t.replace('Ġ', ' ')
                    if i == active_pos_in_context:
                        context_parts.append(f"**{clean_token}**")
                    else:
                        context_parts.append(clean_token)
                context_str = ''.join(context_parts)

                # Get feature activations for each position in context
                positions = self.review_token_positions.get(rev_id_str, [])
                feature_activations = []

                for i in range(ctx_start, ctx_end):
                    if i < len(positions):
                        h5_idx = positions[i]
                        # Read z_idx and z_val for this position
                        z_idx = self.h5['z_idx'][h5_idx]
                        z_val = self.h5['z_val'][h5_idx]
                        # Check if target feature is active
                        if feature_idx in z_idx:
                            feat_pos = np.where(z_idx == feature_idx)[0][0]
                            feature_activations.append(round(float(z_val[feat_pos]), 4))
                        else:
                            feature_activations.append(0.0)
                    else:
                        feature_activations.append(0.0)

                selected_contexts.append({
                    "context": context_str,
                    "tokens": [t.replace('Ġ', ' ') for t in context_tokens],
                    "feature_activations": feature_activations,
                    "active_token_idx": active_pos_in_context,
                    "review_id": rev_id_str
                })

            # Compute mean activation across all occurrences
            all_acts = [c[0] for c in candidates]
            mean_activation = float(np.mean(all_acts)) if all_acts else 0.0

            result_tokens.append({
                "token": token_str,
                "count": count,
                "mean_activation": round(mean_activation, 4),
                "contexts": selected_contexts
            })

        return {
            "sampling": {
                "method": "sequential_from_start",
                "description": "First N tokens in corpus order, top activations per token",
                "tokens_scanned": tokens_scanned,
                "corpus_coverage": round(tokens_scanned / self.total_tokens, 4),
                "activations_found": activations_found,
                "unique_tokens_found": len(token_candidates)
            },
            "tokens": result_tokens
        }

    @modal.method()
    def get_feature_stats(self, feature_idx: int, sample_size: int = 500000, quiet: bool = False) -> dict:
        """Get statistics about a feature's activations.

        Note: Scans tokens sequentially from start of corpus, not random.
        Statistics may differ if the corpus has systematic ordering.
        """
        import numpy as np
        from tqdm import tqdm

        all_activations = []
        active_count = 0
        tokens_scanned = 0
        chunk_size = 100_000

        for start in tqdm(range(0, min(self.total_tokens, sample_size), chunk_size), desc=f"Stats for {feature_idx}", disable=quiet):
            end = min(start + chunk_size, self.total_tokens, sample_size)
            chunk_acts = self._get_feature_activations_chunk(feature_idx, start, end)

            active_mask = chunk_acts > 0
            active_count += np.sum(active_mask)
            all_activations.extend(chunk_acts[active_mask].tolist())
            tokens_scanned += len(chunk_acts)

        if not all_activations:
            return {
                "sampling": {
                    "method": "sequential_from_start",
                    "description": "First N tokens in corpus order (not random)",
                    "tokens_scanned": tokens_scanned,
                    "corpus_coverage": round(tokens_scanned / self.total_tokens, 4)
                },
                "total_activations": 0,
                "mean_when_active": 0.0,
                "max_activation": 0.0,
                "activation_rate": 0.0
            }

        return {
            "sampling": {
                "method": "sequential_from_start",
                "description": "First N tokens in corpus order (not random)",
                "tokens_scanned": tokens_scanned,
                "corpus_coverage": round(tokens_scanned / self.total_tokens, 4)
            },
            "total_activations": int(active_count),
            "mean_when_active": float(np.mean(all_activations)),
            "max_activation": float(np.max(all_activations)),
            "std_when_active": float(np.std(all_activations)) if len(all_activations) > 1 else 0.0,
            "activation_rate": float(active_count / tokens_scanned)
        }

    @modal.method()
    def get_ngram_patterns(
        self,
        feature_idx: int,
        top_k_activations: int = 500,
        ngram_sizes: list = None,
        context_window: int = 5,
        max_scan: int = 100000,
        quiet: bool = False
    ) -> dict:
        """Extract common n-grams from contexts where a feature activates.

        Note: Samples the TOP activations by strength, not random. Results
        reflect patterns in the strongest activations, which may differ from
        weaker activations.
        """
        from collections import Counter
        import numpy as np

        if ngram_sizes is None:
            ngram_sizes = [2, 3, 4]

        top_acts_result = self.get_top_activations.local(
            feature_idx,
            top_k=top_k_activations,
            max_scan=max_scan,
            context_before=context_window,
            context_after=context_window,
            quiet=quiet
        )

        # Extract activations list and sampling info from result
        top_acts = top_acts_result.get("activations", [])
        source_sampling = top_acts_result.get("sampling", {})

        if not top_acts:
            return {
                "feature_idx": feature_idx,
                "sampling": {
                    "method": "top_by_activation",
                    "description": "Top N activations by strength (not random)",
                    "n_requested": top_k_activations,
                    "n_found": 0,
                    "tokens_scanned": source_sampling.get("tokens_scanned", max_scan),
                    "corpus_coverage": source_sampling.get("corpus_coverage", 0)
                },
                "n_contexts_analyzed": 0,
                "ngrams": {}
            }

        # Compute activation stats for the sample
        activations = [act["activation"] for act in top_acts]

        ngram_counters = {n: Counter() for n in ngram_sizes}

        for act_info in top_acts:
            context = act_info.get("context", "")
            context_clean = context.replace("**", "")
            tokens = self.tok(context_clean, add_special_tokens=False).input_ids
            token_strings = [t.replace('Ġ', ' ') for t in self.tok.convert_ids_to_tokens(tokens)]

            for n in ngram_sizes:
                for i in range(len(token_strings) - n + 1):
                    ngram = tuple(token_strings[i:i + n])
                    ngram_counters[n][ngram] += 1

        n_contexts = len(top_acts)
        ngrams_result = {}

        for n in ngram_sizes:
            top_ngrams = ngram_counters[n].most_common(15)
            ngrams_result[f"{n}grams"] = [
                {
                    "ngram": list(ngram),
                    "ngram_str": "".join(ngram),
                    "count": count,
                    "percent": round(100.0 * count / n_contexts, 1)
                }
                for ngram, count in top_ngrams
            ]

        return {
            "feature_idx": feature_idx,
            "sampling": {
                "method": "top_by_activation",
                "description": "Top N activations by strength (not random)",
                "n_requested": top_k_activations,
                "n_found": n_contexts,
                "tokens_scanned": source_sampling.get("tokens_scanned", max_scan),
                "corpus_coverage": source_sampling.get("corpus_coverage", 0),
                "activation_range": {
                    "min": round(float(np.min(activations)), 4),
                    "max": round(float(np.max(activations)), 4),
                    "mean": round(float(np.mean(activations)), 4)
                }
            },
            "n_contexts_analyzed": n_contexts,
            "context_window": context_window,
            "ngrams": ngrams_result
        }


# ============================================================================
# GPU Class: Model Inference (GPT-2 → SAE)
# ============================================================================

@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={VOLUME_MOUNT: data_volume},
    scaledown_window=900,  # Keep warm for 15 minutes
    timeout=600,  # 10 minute timeout per method call
)
class SAEInterpreter:
    """
    GPU service for running text through GPT-2 → SAE pipeline.

    Use this for processing NEW text through the model.
    For querying precomputed activations from the corpus, use SAEDataReader instead.

    Methods:
        process_text: Run text through GPT-2 → SAE pipeline
        test_feature_examples: Batch test texts for feature activation
        context_ablation: Find minimum context needed for feature activation
        compare_text_activations: Compare features between two texts
    """

    @modal.enter()
    def setup(self):
        """Initialize models on container startup (GPU required)."""
        # Suppress warnings BEFORE importing transformer_lens
        import warnings
        warnings.filterwarnings("ignore", message=".*torch_dtype.*")
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        import torch

        # Import with warnings suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from transformer_lens import HookedTransformer
        from transformers import AutoTokenizer

        print("Loading tokenizer...")
        self.tok = AutoTokenizer.from_pretrained("gpt2")
        if self.tok.pad_token is None:
            self.tok.add_special_tokens({'pad_token': self.tok.eos_token})

        print("Loading GPT-2...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = HookedTransformer.from_pretrained(
                "gpt2", device="cuda", dtype="float32"
            )

        print("Loading SAE checkpoint...")
        self.sae = self._load_sae(SAE_CHECKPOINT_PATH)

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

    def _process_text_internal(self, text: str) -> dict:
        """
        Internal implementation of text processing through GPT-2 → SAE pipeline.

        This is the core implementation that can be called from other methods.
        The public process_text method wraps this for external Modal calls.
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
            "feature_activations": feature_activations,
            "z_np": z_np  # Full SAE activations (all 32 features per token)
        }

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
        result = self._process_text_internal(text)
        # Remove z_np before returning - it's a numpy array that can't be serialized
        result.pop("z_np", None)
        return result

    @modal.method()
    def test_feature_examples(
        self,
        feature_idx: int,
        texts: list
    ) -> list:
        """
        Test multiple texts and return activation info for a specific feature.

        Used by hypothesis testing loop for efficient batch processing.

        Args:
            feature_idx: Feature to check for activation
            texts: List of text strings to test

        Returns:
            List of dicts with activation info for each text
        """
        results = []

        for text in texts:
            # Process text through GPT-2 → SAE (using internal method)
            result = self._process_text_internal(text)

            # Get full SAE activations for the specific feature (not truncated top-10)
            z_np = result.get("z_np")
            tokens = result.get("tokens", [])

            # Find max activation and which token
            max_activation = 0.0
            active_token = None
            active_token_idx = None

            if z_np is not None and len(tokens) > 0:
                # Get activations for this specific feature across all tokens
                feature_activations = z_np[:len(tokens), feature_idx]

                for idx, act_val in enumerate(feature_activations):
                    if act_val > max_activation:
                        max_activation = float(act_val)
                        active_token_idx = idx

                if active_token_idx is not None and active_token_idx < len(tokens):
                    active_token = tokens[active_token_idx]

            results.append({
                "text": text,
                "activated": max_activation > 0,
                "max_activation": float(max_activation),
                "active_token": active_token,
                "active_token_idx": active_token_idx,
                "all_tokens": tokens
            })

        return results

    @modal.method()
    def context_ablation(
        self,
        text: str,
        feature_idx: int,
        target_token_idx: int = None,
        activation_threshold: float = 0.01
    ) -> dict:
        """
        Ablate left context to find minimum necessary context for feature activation.

        Since GPT-2 uses causal attention, activation at position N only depends
        on tokens 0 to N-1. By progressively truncating from the left, we can
        identify exactly which preceding tokens are causally necessary.

        Args:
            text: Input text where feature fires
            feature_idx: Feature to analyze
            target_token_idx: Token position to analyze (default: max activation position)
            activation_threshold: Minimum activation to consider "active"

        Returns:
            dict with ablation results at each truncation level
        """
        # First, process full text to get baseline and find target token
        full_result = self._process_text_internal(text)
        tokens = full_result.get("tokens", [])
        feature_key = str(feature_idx)

        if not tokens:
            return {"error": "No tokens in text"}

        # Get activations for this feature
        activations = full_result.get("feature_activations", {}).get(feature_key, [])

        # If no target specified, find position of max activation
        if target_token_idx is None:
            if not activations or max(activations) == 0:
                return {
                    "error": f"Feature {feature_idx} does not activate on this text",
                    "text": text,
                    "tokens": tokens
                }
            target_token_idx = activations.index(max(activations))

        if target_token_idx >= len(tokens):
            return {"error": f"target_token_idx {target_token_idx} out of range (max {len(tokens)-1})"}

        target_token = tokens[target_token_idx]
        original_activation = activations[target_token_idx] if target_token_idx < len(activations) else 0.0

        # Perform ablation: progressively remove tokens from the left
        ablation_steps = []

        for depth in range(target_token_idx + 1):
            # Truncate: remove first 'depth' tokens
            # We need to work with the original text and re-tokenize
            if depth == 0:
                truncated_text = text
            else:
                # Re-tokenize original and truncate
                original_ids = self.tok(text, add_special_tokens=False).input_ids
                truncated_ids = original_ids[depth:]
                truncated_text = self.tok.decode(truncated_ids)

            # Process truncated text
            result = self._process_text_internal(truncated_text)
            trunc_tokens = result.get("tokens", [])
            trunc_activations = result.get("feature_activations", {}).get(feature_key, [])

            # The target token is now at position (target_token_idx - depth)
            new_target_idx = target_token_idx - depth

            if new_target_idx < 0 or new_target_idx >= len(trunc_tokens):
                activation = 0.0
                token_at_pos = None
            else:
                activation = trunc_activations[new_target_idx] if new_target_idx < len(trunc_activations) else 0.0
                token_at_pos = trunc_tokens[new_target_idx]

            # Get left context tokens
            left_context = trunc_tokens[:new_target_idx] if new_target_idx > 0 else []

            ablation_steps.append({
                "depth": depth,
                "tokens_removed": depth,
                "truncated_text": truncated_text[:80] + "..." if len(truncated_text) > 80 else truncated_text,
                "left_context": left_context,
                "target_token": token_at_pos,
                "target_token_idx": new_target_idx,
                "activation": float(activation),
                "activated": activation > activation_threshold
            })

        # Analyze: find the cliff (largest drop)
        cliff_idx = None
        max_drop = 0.0
        for i in range(1, len(ablation_steps)):
            drop = ablation_steps[i - 1]["activation"] - ablation_steps[i]["activation"]
            if drop > max_drop:
                max_drop = drop
                cliff_idx = i

        # Find minimum context (first depth where it stops activating)
        min_context_depth = None
        for step in ablation_steps:
            if not step["activated"]:
                min_context_depth = step["depth"]
                break

        analysis = {
            "cliff_at_depth": cliff_idx,
            "cliff_drop": float(max_drop) if cliff_idx else 0.0,
            "cliff_drop_percent": round(100.0 * max_drop / original_activation, 1) if original_activation > 0 and cliff_idx else 0.0,
            "critical_token_removed": tokens[cliff_idx - 1] if cliff_idx and cliff_idx > 0 else None,
            "minimum_context_depth": min_context_depth,
            "minimum_context_tokens": ablation_steps[min_context_depth - 1]["left_context"] if min_context_depth and min_context_depth > 0 else None
        }

        return {
            "feature_idx": feature_idx,
            "original_text": text,
            "tokens": tokens,
            "target_token": target_token,
            "target_token_idx": target_token_idx,
            "original_activation": float(original_activation),
            "activation_threshold": activation_threshold,
            "ablation_steps": ablation_steps,
            "analysis": analysis
        }

    @modal.method()
    def compare_text_activations(
        self,
        text1: str,
        text2: str,
        top_k: int = 20,
        aggregation: str = 'max'
    ) -> dict:
        """
        Compare which features differ between two texts.

        Equivalent to notebook's compare_text_activations().

        Args:
            text1: First text
            text2: Second text
            top_k: Number of top differences to return
            aggregation: 'max' or 'mean' for aggregating across tokens

        Returns:
            dict with:
                top_text1: Features stronger in text1 [(feature_idx, diff), ...]
                top_text2: Features stronger in text2 [(feature_idx, diff), ...]
                unique_to_text1: Features only active in text1
                unique_to_text2: Features only active in text2
        """
        import numpy as np

        # Process both texts (using internal method)
        result1 = self._process_text_internal(text1)
        result2 = self._process_text_internal(text2)

        # Build full activation arrays
        acts1 = np.zeros(N_LATENTS)
        acts2 = np.zeros(N_LATENTS)

        for feat_str, values in result1['feature_activations'].items():
            feat_idx = int(feat_str)
            if aggregation == 'max':
                acts1[feat_idx] = max(values)
            else:
                acts1[feat_idx] = np.mean([v for v in values if v > 0]) if any(v > 0 for v in values) else 0

        for feat_str, values in result2['feature_activations'].items():
            feat_idx = int(feat_str)
            if aggregation == 'max':
                acts2[feat_idx] = max(values)
            else:
                acts2[feat_idx] = np.mean([v for v in values if v > 0]) if any(v > 0 for v in values) else 0

        # Find differences
        diff = acts1 - acts2
        sorted_idx = np.argsort(np.abs(diff))[::-1]

        # Top features stronger in text1 (positive diff)
        top_text1 = []
        for idx in sorted_idx:
            if diff[idx] > 0:
                top_text1.append({"feature_idx": int(idx), "diff": float(diff[idx]), "text1_act": float(acts1[idx]), "text2_act": float(acts2[idx])})
            if len(top_text1) >= top_k:
                break

        # Top features stronger in text2 (negative diff)
        top_text2 = []
        for idx in sorted_idx:
            if diff[idx] < 0:
                top_text2.append({"feature_idx": int(idx), "diff": float(-diff[idx]), "text1_act": float(acts1[idx]), "text2_act": float(acts2[idx])})
            if len(top_text2) >= top_k:
                break

        # Unique features
        active1 = set(int(k) for k in result1['feature_activations'].keys())
        active2 = set(int(k) for k in result2['feature_activations'].keys())

        return {
            "text1": text1,
            "text2": text2,
            "aggregation": aggregation,
            "top_text1": top_text1,
            "top_text2": top_text2,
            "unique_to_text1": list(active1 - active2),
            "unique_to_text2": list(active2 - active1)
        }


# ============================================================================
# CLI Entry Points - Human-Readable (Verbose)
# ============================================================================

@app.local_entrypoint()
def analyze_feature(feature_idx: int, max_samples: int = 50000, top_k: int = 20):
    """
    Human-friendly feature analysis with progress bars.

    Usage:
        py -3.12 -m modal run src/modal_interpreter.py::analyze_feature -- --feature-idx 16751
    """
    # Use CPU-only reader for corpus queries (no GPU needed)
    reader = SAEDataReader()

    print(f"\n{'='*60}")
    print(f"FEATURE {feature_idx} ANALYSIS")
    print(f"{'='*60}")

    # Stats
    print("\n--- Statistics ---")
    stats = reader.get_feature_stats.remote(feature_idx, sample_size=100000)
    print(f"Sampling: {stats['sampling']['description']}")
    print(f"Corpus coverage: {stats['sampling']['corpus_coverage']*100:.1f}%")
    print(f"Total activations: {stats['total_activations']:,}")
    print(f"Mean (when active): {stats['mean_when_active']:.4f}")
    print(f"Max activation: {stats['max_activation']:.4f}")
    print(f"Activation rate: {stats['activation_rate']:.6f}")

    # Top tokens
    print(f"\n--- Top {top_k} Tokens (by frequency) ---")
    top_tokens_result = reader.get_top_tokens.remote(feature_idx, top_k=top_k, max_samples=max_samples)
    print(f"Sampling: {top_tokens_result['sampling']['description']}")
    for i, t in enumerate(top_tokens_result['top_tokens'], 1):
        print(f"  {i:2d}. '{t['token']}': {t['count']} times, mean act={t['mean_activation']:.4f}")

    # Top activations
    print(f"\n--- Top 10 Strongest Activations ---")
    top_acts_result = reader.get_top_activations.remote(feature_idx, top_k=10, max_scan=max_samples)
    print(f"Sampling: {top_acts_result['sampling']['description']}")
    for i, ctx in enumerate(top_acts_result['activations'], 1):
        print(f"\n  {i}. Activation: {ctx['activation']:.4f}")
        print(f"     Token: '{ctx['active_token']}'")
        print(f"     Context: {ctx['context'][:100]}...")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


@app.local_entrypoint()
def process_user_text(text: str):
    """
    Human-friendly text analysis with verbose output.

    Usage:
        py -3.12 -m modal run src/modal_interpreter.py::process_user_text -- --text "Great tacos!"
    """
    interpreter = SAEInterpreter()

    print(f"\n{'='*60}")
    print("TEXT ANALYSIS")
    print(f"{'='*60}")
    print(f"\nInput: {text}")

    result = interpreter.process_text.remote(text)
    print(f"Tokens: {result['tokens']}")
    print(f"\n--- Top Features Per Token ---")

    for token_idx, (token, features) in enumerate(zip(result['tokens'], result['top_features_per_token'])):
        if features:
            top_3 = features[:3]
            feats_str = ", ".join([f"{f['feature_idx']}({f['activation']:.3f})" for f in top_3])
            print(f"  '{token}': {feats_str}")

    # Overall top activations
    all_acts = []
    for token_idx, (token, features) in enumerate(zip(result['tokens'], result['top_features_per_token'])):
        for feat in features:
            all_acts.append({'token': token, 'feature_idx': feat['feature_idx'], 'activation': feat['activation']})
    all_acts.sort(key=lambda x: x['activation'], reverse=True)

    print(f"\n--- Top 10 Overall Activations ---")
    for i, act in enumerate(all_acts[:10], 1):
        print(f"  {i:2d}. Token '{act['token']}' → Feature {act['feature_idx']} ({act['activation']:.3f})")

    print(f"\n{'='*60}\n")


@app.local_entrypoint()
def compare_texts(text1: str, text2: str, top_k: int = 10):
    """
    Human-friendly comparison of two texts.

    Usage:
        py -3.12 -m modal run src/modal_interpreter.py::compare_texts -- --text1 "Great food!" --text2 "Okay food."
    """
    interpreter = SAEInterpreter()

    print(f"\n{'='*60}")
    print("TEXT COMPARISON")
    print(f"{'='*60}")
    print(f"\nText 1: {text1}")
    print(f"Text 2: {text2}")

    result = interpreter.compare_text_activations.remote(text1, text2, top_k=top_k)

    print(f"\n--- Features Stronger in Text 1 ---")
    for i, feat in enumerate(result['top_text1'][:top_k], 1):
        print(f"  {i}. Feature {feat['feature_idx']}: {feat['text1_act']:.3f} vs {feat['text2_act']:.3f} (diff: +{feat['diff']:.3f})")

    print(f"\n--- Features Stronger in Text 2 ---")
    for i, feat in enumerate(result['top_text2'][:top_k], 1):
        print(f"  {i}. Feature {feat['feature_idx']}: {feat['text1_act']:.3f} vs {feat['text2_act']:.3f} (diff: -{feat['diff']:.3f})")

    print(f"\nFeatures unique to text 1: {len(result['unique_to_text1'])}")
    print(f"Features unique to text 2: {len(result['unique_to_text2'])}")

    print(f"\n{'='*60}\n")


# ============================================================================
# CLI Entry Points - Agent-Friendly (JSON output to file)
# ============================================================================

@app.local_entrypoint()
def analyze_feature_json(
    feature_idx: int,
    max_samples: int = 50000,
    top_k: int = 10,
    output_dir: str = "output",
    include_ngrams: bool = True,
    include_coactivation: bool = True,
    include_position: bool = True,
    include_directional_ngrams: bool = False,
    include_distribution: bool = True,
    include_stability: bool = False,
    include_top_token_contexts: bool = True,
    contexts_per_token: int = 2,
    context_chars: int = 50,
    run_ablation: bool = False,
    ablation_top_k: int = 3
):
    """
    Agent-friendly feature analysis - writes JSON to file.

    Usage:
        py -3.12 -m modal run src/modal_interpreter.py::analyze_feature_json -- --feature-idx 16751

        # With all analyses enabled:
        py -3.12 -m modal run src/modal_interpreter.py::analyze_feature_json -- \\
            --feature-idx 16751 --include-directional-ngrams --include-stability

        # With ablation on top 3 activations:
        py -3.12 -m modal run src/modal_interpreter.py::analyze_feature_json -- \\
            --feature-idx 16751 --run-ablation

    Args:
        feature_idx: Feature to analyze
        max_samples: Max tokens to scan for statistics
        top_k: Number of top tokens to return (default: 10)
        output_dir: Directory for output file
        include_ngrams: Include n-gram pattern analysis (default: True)
        include_coactivation: Include co-activation matrix (default: True)
        include_position: Include position distribution analysis (default: True)
        include_directional_ngrams: Include directional n-grams (default: False - niche)
        include_distribution: Include activation percentiles/skewness (default: True)
        include_stability: Include token stability analysis (default: False - niche)
        include_top_token_contexts: Include top token contexts (default: True)
        contexts_per_token: Contexts per token in top_token_contexts (default: 2)
        context_chars: Max chars for context strings (default: 50)
        run_ablation: Run context ablation on top activations (default: False)
        ablation_top_k: Number of top activations to run ablation on
    """
    import json
    import os

    def truncate_context(ctx: str, max_chars: int) -> str:
        """Truncate context string while preserving active token markers."""
        if len(ctx) <= max_chars:
            return ctx
        # Find the active token position (marked with **)
        if "**" in ctx:
            start = ctx.find("**")
            end = ctx.rfind("**") + 2
            # Keep the active token centered if possible
            active_len = end - start
            remaining = max_chars - active_len
            if remaining > 0:
                before = remaining // 2
                after = remaining - before
                new_start = max(0, start - before)
                new_end = min(len(ctx), end + after)
                return ("..." if new_start > 0 else "") + ctx[new_start:new_end] + ("..." if new_end < len(ctx) else "")
        # No marker found, simple truncation
        return ctx[:max_chars] + "..."

    def render_unicode(text: str) -> str:
        """Convert common unicode escapes to readable labels for human/LLM interpretation.

        GPT-2 tokenizer uses special characters:
        - Ċ (U+010A) for newline
        - Ġ (U+0120) for space-prefixed tokens (we leave these alone)
        - ĉ (U+0109) for tab
        """
        if not text:
            return text
        # Order matters - check double newline first
        text = text.replace("\u010a\u010a", "\\n\\n")
        text = text.replace("\u010a", "\\n")
        text = text.replace("\u0109", "\\t")
        return text

    # Use CPU-only reader for corpus queries (no GPU needed)
    reader = SAEDataReader()

    print(f"Analyzing feature {feature_idx}...")

    # Collect all data quietly (CPU operations)
    # All methods now return dicts with sampling metadata
    stats = reader.get_feature_stats.remote(feature_idx, sample_size=100000, quiet=True)
    top_tokens_result = reader.get_top_tokens.remote(feature_idx, top_k=top_k, max_samples=max_samples, quiet=True)
    top_acts_result = reader.get_top_activations.remote(feature_idx, top_k=10, max_scan=max_samples, quiet=True)

    result = {
        "feature_idx": feature_idx,
        "stats": stats,
        "top_tokens": top_tokens_result,
        "top_activations": top_acts_result
    }

    # Render unicode in top_tokens
    if "top_tokens" in top_tokens_result:
        for tok in top_tokens_result["top_tokens"]:
            if "token" in tok:
                tok["token"] = render_unicode(tok["token"])

    # Truncate contexts and render unicode in top_activations
    if "activations" in top_acts_result:
        for act in top_acts_result["activations"]:
            if "context" in act:
                act["context"] = render_unicode(truncate_context(act["context"], context_chars))
            if "active_token" in act:
                act["active_token"] = render_unicode(act["active_token"])

    # N-gram analysis (CPU)
    if include_ngrams:
        print("Running n-gram analysis (top 500 activations)...")
        ngram_result = reader.get_ngram_patterns.remote(
            feature_idx,
            top_k_activations=500,
            max_scan=max(max_samples, 100000),
            quiet=True
        )
        # Render unicode in ngrams
        if "ngrams" in ngram_result:
            for ngram_size, ngram_list in ngram_result["ngrams"].items():
                for ng in ngram_list:
                    if "ngram" in ng:
                        ng["ngram"] = [render_unicode(t) for t in ng["ngram"]]
                    if "ngram_str" in ng:
                        ng["ngram_str"] = render_unicode(ng["ngram_str"])
        result["ngram_analysis"] = ngram_result

    # Co-activation analysis (CPU)
    if include_coactivation:
        print("Running co-activation analysis...")
        coact_result = reader.get_coactivated_features.remote(
            feature_idx,
            top_k=20,
            max_samples=10000,
            quiet=True
        )
        result["coactivation"] = coact_result

    # Position distribution (CPU)
    if include_position:
        print("Running position distribution analysis...")
        pos_result = reader.get_position_distribution.remote(
            feature_idx,
            n_bins=5,
            max_samples=10000,
            quiet=True
        )
        result["position_distribution"] = pos_result

    # Directional n-grams (CPU) - OFF by default
    if include_directional_ngrams:
        print("Running directional n-gram analysis...")
        dir_ngram_result = reader.get_directional_ngrams.remote(
            feature_idx,
            ngram_sizes=[2, 3],
            top_k=15,
            max_samples=500,
            quiet=True
        )
        result["directional_ngrams"] = dir_ngram_result

    # Activation distribution (CPU)
    if include_distribution:
        print("Running activation distribution analysis...")
        dist_result = reader.get_activation_distribution.remote(
            feature_idx,
            sample_size=100000,
            quiet=True
        )
        result["activation_distribution"] = dist_result

    # Token stability (CPU) - OFF by default
    if include_stability:
        print("Running token stability analysis...")
        stab_result = reader.get_token_stability.remote(
            feature_idx,
            top_k_tokens=20,
            max_samples=5000,
            quiet=True
        )
        result["token_stability"] = stab_result

    # Top token contexts (CPU)
    if include_top_token_contexts:
        print("Running top token contexts analysis...")
        ctx_result = reader.get_top_token_contexts.remote(
            feature_idx,
            top_k_tokens=top_k,
            contexts_per_token=contexts_per_token,
            max_scan=max_samples,
            context_size=15,
            quiet=True
        )
        # Simplify output: remove detailed contexts arrays (too verbose for interpretation)
        # Keep only aggregated stats per token (count, mean_activation)
        if "tokens" in ctx_result:
            for token_info in ctx_result["tokens"]:
                # Render unicode in token name
                if "token" in token_info:
                    token_info["token"] = render_unicode(token_info["token"])
                # Remove the verbose contexts array (31-token windows + feature_activations)
                if "contexts" in token_info:
                    del token_info["contexts"]
        result["top_token_contexts"] = ctx_result

    # Context ablation on top activations (requires GPU for GPT-2 -> SAE)
    # Extract the activations list from the result dict
    top_acts_list = top_acts_result.get("activations", []) if isinstance(top_acts_result, dict) else top_acts_result
    if run_ablation and top_acts_list:
        print(f"Running context ablation on top {ablation_top_k} activations...")
        interpreter = SAEInterpreter()  # GPU instance only when needed
        ablation_results = []
        for i, act_info in enumerate(top_acts_list[:ablation_top_k]):
            # Get full review text for ablation
            review_id = act_info.get("review_id")
            context = act_info.get("context", "").replace("**", "")

            if context:
                ablation = interpreter.context_ablation.remote(
                    text=context,
                    feature_idx=feature_idx
                )
                ablation_results.append({
                    "rank": i + 1,
                    "original_context": act_info.get("context"),
                    "ablation": ablation
                })

        result["ablation_analysis"] = ablation_results

    # Write to file (also create interpretations subdir for downstream use)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "interpretations"), exist_ok=True)
    output_path = os.path.join(output_dir, f"feature_{feature_idx}.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results written to: {output_path}")


@app.local_entrypoint()
def process_text_json(text: str, output_dir: str = "output"):
    """
    Agent-friendly text analysis - writes JSON to file.

    Usage:
        py -3.12 -m modal run src/modal_interpreter.py::process_text_json -- --text "Great tacos!"
    """
    import json
    import os

    interpreter = SAEInterpreter()
    result = interpreter.process_text.remote(text)

    # Write to file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "text_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results written to: {output_path}")


@app.local_entrypoint()
def compare_texts_json(text1: str, text2: str, top_k: int = 20, output_dir: str = "output"):
    """
    Agent-friendly text comparison - writes JSON to file.

    Usage:
        py -3.12 -m modal run src/modal_interpreter.py::compare_texts_json -- --text1 "Great!" --text2 "Okay."
    """
    import json
    import os

    interpreter = SAEInterpreter()
    result = interpreter.compare_text_activations.remote(text1, text2, top_k=top_k)

    # Write to file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "compare_results.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results written to: {output_path}")


@app.local_entrypoint()
def batch_test(
    feature_idx: int,
    texts: str = "",
    texts_file: str = "",
    output_dir: str = "output",
    show_visual: bool = True,
    fresh: bool = False
):
    """
    Batch test multiple texts for a specific feature - single Modal call.

    This is the primary tool for hypothesis testing during feature interpretation.
    All texts are processed in ONE Modal invocation, making it ~10x faster than
    running individual process_text_json calls.

    Usage:
        # Inline texts (comma-separated, use | for texts containing commas)
        py -3.12 -m modal run src/modal_interpreter.py::batch_test -- \\
            --feature-idx 16751 \\
            --texts "I have never in my life|Why in the world|I live in my house"

        # From JSON file
        py -3.12 -m modal run src/modal_interpreter.py::batch_test -- \\
            --feature-idx 16751 \\
            --texts-file tests.json

    Args:
        feature_idx: Feature to test for activation
        texts: Pipe-separated (|) list of texts to test
        texts_file: Path to JSON file with list of texts
        output_dir: Directory for output file
        show_visual: Show visual activation bars in console
        fresh: If True, overwrite existing results. If False (default), append to existing.

    Output:
        Writes/appends to batch_test_<feature_idx>.json with results for each text.
        Results accumulate across multiple calls unless --fresh is specified.
    """
    import json
    import os

    interpreter = SAEInterpreter()

    # Parse input texts
    text_list = []
    if texts_file:
        with open(texts_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                text_list = data
            elif isinstance(data, dict) and 'texts' in data:
                text_list = data['texts']
    elif texts:
        # Use | as delimiter to allow commas in texts
        text_list = [t.strip() for t in texts.split('|') if t.strip()]

    if not text_list:
        print("Error: No texts provided. Use --texts or --texts-file")
        return

    print(f"\nTesting {len(text_list)} texts for feature {feature_idx}...")

    # Single remote call for all texts
    results = interpreter.test_feature_examples.remote(feature_idx, text_list)

    # Display results
    print(f"\n{'='*70}")
    print(f"BATCH TEST RESULTS - Feature {feature_idx}")
    print(f"{'='*70}\n")

    max_act_overall = max(r['max_activation'] for r in results) if results else 1.0
    if max_act_overall == 0:
        max_act_overall = 1.0  # Avoid division by zero

    for i, r in enumerate(results, 1):
        status = "+" if r['activated'] else "-"
        act_str = f"{r['max_activation']:.3f}" if r['activated'] else "0.000"

        # Visual bar
        if show_visual and r['activated']:
            bar_width = int(30 * r['max_activation'] / max_act_overall)
            bar = "█" * bar_width + "░" * (30 - bar_width)
            visual = f" |{bar}|"
        else:
            visual = " |" + "░" * 30 + "|" if show_visual else ""

        token_info = f" @ '{r['active_token']}'" if r['active_token'] else ""

        try:
            print(f"[{status}] {act_str}{visual}{token_info}")
            # Truncate long texts for display
            display_text = r['text'][:60] + "..." if len(r['text']) > 60 else r['text']
            print(f"    {display_text}\n")
        except UnicodeEncodeError:
            # Windows console encoding issue - skip visual output
            print(f"[{status}] {act_str}{token_info}")
            print(f"    {r['text'][:60]}...\n" if len(r['text']) > 60 else f"    {r['text']}\n")

    # Summary
    activated_count = sum(1 for r in results if r['activated'])
    print(f"{'='*70}")
    print(f"Summary: {activated_count}/{len(results)} texts activated feature {feature_idx}")
    print(f"{'='*70}\n")

    # Write JSON output (append to existing if present)
    import time
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"batch_test_{feature_idx}.json")

    # Add timestamp to each result
    batch_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for r in results:
        r["batch_timestamp"] = batch_timestamp

    # Read existing results if file exists and not fresh
    existing_results = []
    existing_batches = 0
    if not fresh and os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
                if isinstance(existing_data.get("results"), list):
                    existing_results = existing_data["results"]
                    existing_batches = existing_data.get("n_batches", 1)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read existing {output_path}: {e}")
            print("Starting fresh.")

    # Combine results
    all_results = existing_results + results

    output_data = {
        "feature_idx": feature_idx,
        "n_texts": len(all_results),
        "n_activated": sum(1 for r in all_results if r.get("activated")),
        "n_batches": existing_batches + 1,
        "last_updated": batch_timestamp,
        "results": all_results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    if existing_results:
        print(f"Results appended to: {output_path}")
        print(f"  (Total: {len(all_results)} tests across {output_data['n_batches']} batches)")
    else:
        print(f"Results written to: {output_path}")


@app.local_entrypoint()
def ablate_context(
    feature_idx: int,
    text: str,
    target_token_idx: int = -1,
    output_dir: str = "output"
):
    """
    Analyze minimum left-context needed for feature activation.

    Since GPT-2 uses causal attention, this progressively removes tokens
    from the left to find which preceding context is causally necessary.

    Usage:
        py -3.12 -m modal run src/modal_interpreter.py::ablate_context -- \\
            --feature-idx 16751 \\
            --text "I have never in my life tasted such amazing tacos."

    Args:
        feature_idx: Feature to analyze
        text: Text where the feature activates
        target_token_idx: Token position to analyze (-1 = auto-detect max activation)
        output_dir: Directory for output file
    """
    import json
    import os

    interpreter = SAEInterpreter()

    print(f"\nAnalyzing context ablation for feature {feature_idx}...")
    print(f"Text: {text}\n")

    # Run ablation
    target_idx = None if target_token_idx == -1 else target_token_idx
    result = interpreter.context_ablation.remote(feature_idx=feature_idx, text=text, target_token_idx=target_idx)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Display results
    print(f"{'='*75}")
    print(f"CONTEXT ABLATION - Feature {feature_idx}")
    print(f"{'='*75}")
    print(f"Target token: '{result['target_token']}' (position {result['target_token_idx']})")
    print(f"Original activation: {result['original_activation']:.3f}")
    print()

    # Table header
    print(f"{'Depth':<6} {'Left Context':<35} {'Activation':<12} {'Visual'}")
    print(f"{'-'*6} {'-'*35} {'-'*12} {'-'*30}")

    max_act = result['original_activation']
    if max_act == 0:
        max_act = 1.0

    for step in result['ablation_steps']:
        depth = step['depth']
        left_ctx = " ".join(step['left_context'])[-32:] if step['left_context'] else "(none)"
        if len(left_ctx) > 32:
            left_ctx = "..." + left_ctx[-29:]
        act = step['activation']
        activated = step['activated']

        # Visual bar
        bar_width = int(25 * act / max_act) if max_act > 0 else 0
        bar = "█" * bar_width + "░" * (25 - bar_width)

        # Mark cliff
        cliff_marker = ""
        if result['analysis']['cliff_at_depth'] == depth:
            cliff_marker = " <- CLIFF"

        status = "+" if activated else "-"
        try:
            print(f"{depth:<6} {left_ctx:<35} [{status}] {act:.3f}   |{bar}|{cliff_marker}")
        except UnicodeEncodeError:
            # Windows console encoding issue - no bar
            print(f"{depth:<6} {left_ctx:<35} [{status}] {act:.3f}{cliff_marker}")

    # Analysis summary
    print()
    print(f"{'='*75}")
    print("ANALYSIS")
    print(f"{'='*75}")
    analysis = result['analysis']

    if analysis['cliff_at_depth']:
        print(f"Critical token: '{analysis['critical_token_removed']}' (removing it drops activation by {analysis['cliff_drop_percent']:.1f}%)")

    if analysis['minimum_context_tokens']:
        min_ctx = "".join(analysis['minimum_context_tokens'])
        print(f"Minimum context: \"{min_ctx}\" ({len(analysis['minimum_context_tokens'])} tokens)")
    elif analysis['minimum_context_depth'] == 0:
        print(f"Minimum context: (none) - feature fires with just the target token")
    else:
        print(f"Minimum context depth: {analysis['minimum_context_depth']}")

    print(f"{'='*75}\n")

    # Write JSON output
    os.makedirs(output_dir, exist_ok=True)
    text_hash = abs(hash(text)) % 10000
    output_path = os.path.join(output_dir, f"ablation_{feature_idx}_{text_hash}.json")

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results written to: {output_path}")


# ============================================================================
# Test Entrypoints (for development)
# ============================================================================

@app.local_entrypoint()
def test_interpreter():
    """Test the SAEInterpreter and SAEDataReader methods."""
    # GPU service for inference
    interpreter = SAEInterpreter()
    # CPU service for corpus queries
    reader = SAEDataReader()

    print("\n=== Testing process_text (GPU) ===")
    test_text = "Why in the world would anyone eat here?"
    result = interpreter.process_text.remote(test_text)
    print(f"Text: {test_text}")
    print(f"Tokens: {result['tokens']}")

    # Collect top activations across all tokens
    all_activations = []
    for token_idx, (token, features) in enumerate(zip(result['tokens'], result['top_features_per_token'])):
        for feat in features[:5]:  # Top 5 per token
            all_activations.append({
                'token': token,
                'token_idx': token_idx,
                'feature_idx': feat['feature_idx'],
                'activation': feat['activation']
            })

    # Sort by activation strength and show top 10
    all_activations.sort(key=lambda x: x['activation'], reverse=True)
    print(f"\nTop 10 activations:")
    for i, act in enumerate(all_activations[:10], 1):
        print(f"  {i:2d}. Token '{act['token']}' [pos {act['token_idx']}] → Feature {act['feature_idx']} ({act['activation']:.3f})")

    print("\n=== Testing get_feature_stats (CPU) ===")
    stats = reader.get_feature_stats.remote(16751, sample_size=100000)
    print(f"Sampling: {stats['sampling']['description']}")
    print(f"Feature 16751 stats: activations={stats['total_activations']}, rate={stats['activation_rate']:.6f}")

    print("\n=== Testing get_top_tokens (CPU) ===")
    top_tokens_result = reader.get_top_tokens.remote(16751, top_k=10, max_samples=10000)
    print(f"Sampling: {top_tokens_result['sampling']['description']}")
    print(f"Top tokens for feature 16751:")
    for t in top_tokens_result['top_tokens'][:5]:
        print(f"  '{t['token']}': {t['count']} times, mean act={t['mean_activation']:.3f}")

    print("\n=== Testing get_feature_contexts (CPU, first N found) ===")
    contexts_result = reader.get_feature_contexts.remote(16751, n_samples=5)
    print(f"Sampling: {contexts_result['sampling']['description']}")
    print(f"Example contexts for feature 16751 (first found):")
    for i, ctx in enumerate(contexts_result['contexts'][:5], 1):
        print(f"\n  {i}. Activation: {ctx['activation']:.4f}")
        print(f"     Token: '{ctx['active_token']}'")
        print(f"     Context: {ctx['context'][:80]}...")

    print("\n=== Testing get_top_activations (CPU, globally strongest) ===")
    top_acts_result = reader.get_top_activations.remote(16751, top_k=10, max_scan=50000)
    print(f"Sampling: {top_acts_result['sampling']['description']}")
    print(f"Top 10 strongest activations for feature 16751:")
    for i, ctx in enumerate(top_acts_result['activations'][:10], 1):
        print(f"\n  {i}. Activation: {ctx['activation']:.4f}")
        print(f"     Token: '{ctx['active_token']}'")
        print(f"     Context: {ctx['context'][:80]}...")

    print("\n=== All tests passed! ===")


@app.local_entrypoint()
def test_top_activations():
    """Quick test for get_top_activations only (CPU)."""
    reader = SAEDataReader()

    print("\n=== Testing get_top_activations (CPU, globally strongest) ===")
    print("Scanning 10,000 active tokens to find top 10 strongest...")
    top_acts_result = reader.get_top_activations.remote(16751, top_k=10, max_scan=10000)
    print(f"Sampling: {top_acts_result['sampling']['description']}")
    print(f"Corpus coverage: {top_acts_result['sampling']['corpus_coverage']*100:.1f}%")
    print(f"\nTop 10 strongest activations for feature 16751:")
    for i, ctx in enumerate(top_acts_result['activations'][:10], 1):
        print(f"\n  {i}. Activation: {ctx['activation']:.4f}")
        print(f"     Token: '{ctx['active_token']}'")
        print(f"     Context: {ctx['context'][:100]}...")

    print("\n=== Done ===")
