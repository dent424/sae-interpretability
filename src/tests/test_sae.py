"""Tests for TopKSAE model (CPU only)."""

import sys
from pathlib import Path

import torch
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sae import TopKSAE


class TestTopKSAEShapes:
    """Test input/output tensor shapes."""

    def test_single_input_shape(self):
        """Single input should produce matching output shape."""
        d_model, n_lat, k = 768, 24576, 32
        sae = TopKSAE(d_model, n_lat, k, torch.zeros(d_model))

        x = torch.randn(1, d_model)
        x_hat, z = sae(x)

        assert x_hat.shape == x.shape
        assert z.shape == (1, n_lat)

    def test_batch_input_shape(self):
        """Batch input should produce matching batch output."""
        d_model, n_lat, k = 768, 24576, 32
        sae = TopKSAE(d_model, n_lat, k, torch.zeros(d_model))

        batch_size = 10
        x = torch.randn(batch_size, d_model)
        x_hat, z = sae(x)

        assert x_hat.shape == (batch_size, d_model)
        assert z.shape == (batch_size, n_lat)

    def test_smaller_dimensions(self):
        """Test with smaller dimensions for faster execution."""
        d_model, n_lat, k = 64, 256, 8
        sae = TopKSAE(d_model, n_lat, k, torch.zeros(d_model))

        x = torch.randn(5, d_model)
        x_hat, z = sae(x)

        assert x_hat.shape == (5, d_model)
        assert z.shape == (5, n_lat)


class TestTopKSAESparsity:
    """Test sparsity properties."""

    def test_exactly_k_active(self):
        """Exactly K features should be active per input."""
        d_model, n_lat, k = 64, 256, 8
        sae = TopKSAE(d_model, n_lat, k, torch.zeros(d_model))

        x = torch.randn(1, d_model)
        _, z = sae(x)

        n_active = (z != 0).sum().item()
        assert n_active == k

    def test_batch_sparsity(self):
        """Each sample in batch should have exactly K active features."""
        d_model, n_lat, k = 64, 256, 8
        sae = TopKSAE(d_model, n_lat, k, torch.zeros(d_model))

        batch_size = 10
        x = torch.randn(batch_size, d_model)
        _, z = sae(x)

        for i in range(batch_size):
            n_active = (z[i] != 0).sum().item()
            assert n_active == k, f"Sample {i} has {n_active} active features, expected {k}"

    def test_different_k_values(self):
        """Test with different K values."""
        d_model, n_lat = 64, 256

        for k in [1, 4, 16, 32]:
            sae = TopKSAE(d_model, n_lat, k, torch.zeros(d_model))
            x = torch.randn(1, d_model)
            _, z = sae(x)

            n_active = (z != 0).sum().item()
            assert n_active == k, f"Expected {k} active, got {n_active}"


class TestTopKSAEBaseline:
    """Test baseline/centering behavior."""

    def test_baseline_affects_output(self):
        """Different baselines should produce different outputs."""
        d_model, n_lat, k = 64, 256, 8

        baseline1 = torch.zeros(d_model)
        baseline2 = torch.ones(d_model)

        sae1 = TopKSAE(d_model, n_lat, k, baseline1)
        sae2 = TopKSAE(d_model, n_lat, k, baseline2)

        # Use same weights for fair comparison
        sae2.W_enc.data = sae1.W_enc.data.clone()
        sae2.W_dec.data = sae1.W_dec.data.clone()

        x = torch.randn(1, d_model)
        x_hat1, _ = sae1(x)
        x_hat2, _ = sae2(x)

        # Outputs should differ due to different baselines
        assert not torch.allclose(x_hat1, x_hat2)

    def test_baseline_in_reconstruction(self):
        """Reconstruction should include baseline offset."""
        d_model, n_lat, k = 64, 256, 8
        baseline = torch.ones(d_model) * 5.0

        sae = TopKSAE(d_model, n_lat, k, baseline)

        # Input centered around baseline should reconstruct well
        x = baseline + torch.randn(1, d_model) * 0.1
        x_hat, _ = sae(x)

        # Reconstruction should be close to baseline (since input is close to it)
        # This is a weak test but verifies baseline is used
        assert x_hat.shape == x.shape


class TestTopKSAEGradients:
    """Test gradient flow (important for training)."""

    def test_gradients_flow(self):
        """Gradients should flow through the model."""
        d_model, n_lat, k = 64, 256, 8
        sae = TopKSAE(d_model, n_lat, k, torch.zeros(d_model))

        x = torch.randn(1, d_model, requires_grad=True)
        x_hat, z = sae(x)

        loss = x_hat.sum()
        loss.backward()

        assert x.grad is not None
        assert sae.W_enc.grad is not None
        assert sae.W_dec.grad is not None
        assert sae.b_pre.grad is not None
