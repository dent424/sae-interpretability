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
    import sys
    from pathlib import Path

    # Add parent to path for config import
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import D_MODEL, N_LATENTS, K_ACTIVE

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sae = TopKSAE(D_MODEL, N_LATENTS, K_ACTIVE, torch.zeros(D_MODEL))
    sae.load_state_dict(ckpt["ema_sae"])
    sae.eval()
    return sae.to(device)
