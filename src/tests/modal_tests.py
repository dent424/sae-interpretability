"""GPU tests that run on Modal.

Usage:
    modal run src/tests/modal_tests.py                    # Run all tests
    modal run src/tests/modal_tests.py::test_pipeline     # Run specific test
    modal run src/tests/modal_tests.py::validate          # Run validation tests
"""

import modal

app = modal.App("sae-tests")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformer-lens",
        "transformers",
        "numpy",
        "pandas",
        "h5py",
    )
)

# Reference the data volume (create if doesn't exist for testing)
volume = modal.Volume.from_name("sae-data", create_if_missing=True)

VOLUME_MOUNT = "/data"


# =============================================================================
# Tier 2: Integration Tests (GPU, no real data)
# =============================================================================

@app.function(image=image, gpu="T4", timeout=600)
def test_gpt2_loads():
    """Test that GPT-2 loads correctly on GPU."""
    from transformer_lens import HookedTransformer

    print("Loading GPT-2...")
    model = HookedTransformer.from_pretrained("gpt2", device="cuda", dtype="float32")

    assert model is not None
    print(f"✓ GPT-2 loaded successfully")
    print(f"  Device: {next(model.parameters()).device}")
    return True


@app.function(image=image, gpu="T4", timeout=600)
def test_activation_extraction():
    """Test GPT-2 activation extraction via hooks."""
    import torch
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    print("Loading models...")
    model = HookedTransformer.from_pretrained("gpt2", device="cuda", dtype="float32")
    tok = AutoTokenizer.from_pretrained("gpt2")

    text = "Hello world"
    tokens = tok(text, return_tensors="pt").input_ids.cuda()

    activations = []

    def hook(act, hook):
        activations.append(act.detach().cpu())

    print("Running forward pass with hook...")
    model.run_with_hooks(
        tokens,
        fwd_hooks=[("blocks.8.hook_resid_pre", hook)],
        return_type=None
    )

    assert len(activations) == 1, f"Expected 1 activation, got {len(activations)}"
    assert activations[0].shape[-1] == 768, f"Expected d_model=768, got {activations[0].shape[-1]}"

    print(f"✓ Activation extraction passed")
    print(f"  Shape: {activations[0].shape}")
    return True


@app.function(image=image, gpu="T4", timeout=600)
def test_pipeline():
    """Test full text → activations → SAE pipeline."""
    import torch
    import math
    from torch import nn
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    # Local TopKSAE implementation for testing
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

    print("Loading models...")
    model = HookedTransformer.from_pretrained("gpt2", device="cuda", dtype="float32")
    tok = AutoTokenizer.from_pretrained("gpt2")

    # Create SAE (random weights for testing)
    D_MODEL, N_LATENTS, K_ACTIVE = 768, 24576, 32
    sae = TopKSAE(D_MODEL, N_LATENTS, K_ACTIVE, torch.zeros(D_MODEL))
    sae.eval().cuda()

    # Process text
    text = "The food at this restaurant was absolutely amazing!"
    tokens = tok(text, return_tensors="pt").input_ids.cuda()

    print(f"Input: '{text}'")
    print(f"Tokens: {tokens.shape}")

    # Extract activations
    activations = []

    def hook(act, hook):
        flat = act.detach().reshape(-1, D_MODEL)
        flat = flat - flat.mean(-1, keepdim=True)
        flat = flat / flat.norm(dim=-1, keepdim=True)
        activations.append(flat)

    model.run_with_hooks(
        tokens,
        fwd_hooks=[("blocks.8.hook_resid_pre", hook)],
        return_type=None
    )

    gpt2_acts = activations[0]
    print(f"GPT-2 activations: {gpt2_acts.shape}")

    # Run through SAE
    with torch.no_grad():
        x_hat, z = sae(gpt2_acts)

    print(f"SAE output: x_hat={x_hat.shape}, z={z.shape}")

    # Verify sparsity
    n_active_per_token = (z != 0).sum(dim=1)
    assert all(n == K_ACTIVE for n in n_active_per_token), "Sparsity violation"

    print(f"✓ Full pipeline test passed")
    print(f"  Active features per token: {n_active_per_token.tolist()}")
    return True


# =============================================================================
# Tier 3: Validation Tests (GPU + real data)
# =============================================================================

@app.function(image=image, gpu="T4", volumes={VOLUME_MOUNT: volume}, timeout=600)
def validate():
    """Validate outputs match notebook expectations.

    Requires real data files on the volume:
    - /data/sae_e32_k32_lr0.0003-final.pt
    - /data/mexican_national_metadata.npz
    """
    import os
    import torch
    import math
    from torch import nn
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    # Check if data files exist
    checkpoint_path = f"{VOLUME_MOUNT}/sae_e32_k32_lr0.0003-final.pt"
    if not os.path.exists(checkpoint_path):
        print("⚠ Validation skipped: checkpoint not found on volume")
        print(f"  Expected: {checkpoint_path}")
        print("  Upload data with: modal volume put sae-data /path/to/checkpoint.pt /")
        return {"status": "skipped", "reason": "no checkpoint"}

    # Local TopKSAE for loading
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

    D_MODEL, N_LATENTS, K_ACTIVE = 768, 24576, 32

    print("Loading models...")
    model = HookedTransformer.from_pretrained("gpt2", device="cuda", dtype="float32")
    tok = AutoTokenizer.from_pretrained("gpt2")

    print("Loading SAE checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sae = TopKSAE(D_MODEL, N_LATENTS, K_ACTIVE, torch.zeros(D_MODEL))
    sae.load_state_dict(ckpt["ema_sae"])
    sae.eval().cuda()

    # Test: Feature 16751 should activate on emphatic expressions
    test_cases = [
        ("Why in the world would anyone do that?", 16751, "emphatic"),
        ("How in the world did this happen?", 16751, "emphatic"),
        ("Never in my life have I seen this!", 16751, "emphatic"),
    ]

    results = []

    for text, expected_feature, description in test_cases:
        tokens = tok(text, return_tensors="pt").input_ids.cuda()

        activations = []

        def hook(act, hook):
            flat = act.detach().reshape(-1, D_MODEL)
            flat = flat - flat.mean(-1, keepdim=True)
            flat = flat / flat.norm(dim=-1, keepdim=True)
            activations.append(flat)

        model.run_with_hooks(
            tokens,
            fwd_hooks=[("blocks.8.hook_resid_pre", hook)],
            return_type=None
        )

        with torch.no_grad():
            _, z = sae(activations[0])

        # Check if expected feature activates anywhere
        feature_acts = z[:, expected_feature].cpu().numpy()
        max_activation = feature_acts.max()
        activates = max_activation > 0

        result = {
            "text": text,
            "feature": expected_feature,
            "description": description,
            "max_activation": float(max_activation),
            "activates": activates
        }
        results.append(result)

        status = "✓" if activates else "✗"
        print(f"{status} Feature {expected_feature} ({description}): max={max_activation:.4f}")
        print(f"   Text: '{text}'")

    all_passed = all(r["activates"] for r in results)

    if all_passed:
        print("\n✓ All validation tests passed!")
    else:
        print("\n✗ Some validation tests failed")

    return {"status": "passed" if all_passed, "results": results}


# =============================================================================
# Local Entrypoint
# =============================================================================

@app.local_entrypoint()
def main():
    """Run all Modal tests."""
    print("=" * 60)
    print("SAE Modal Tests")
    print("=" * 60)

    print("\n--- Tier 2: Integration Tests ---\n")

    print("Test 1: GPT-2 Loading")
    test_gpt2_loads.remote()

    print("\nTest 2: Activation Extraction")
    test_activation_extraction.remote()

    print("\nTest 3: Full Pipeline")
    test_pipeline.remote()

    print("\n--- Tier 3: Validation Tests ---\n")

    print("Test 4: Feature Validation")
    result = validate.remote()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
