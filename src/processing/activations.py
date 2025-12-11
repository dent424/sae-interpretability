"""GPT-2 activation extraction."""

import torch
from transformer_lens import HookedTransformer

import sys
from pathlib import Path

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import D_MODEL, HOOK


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
