"""Tests for config.py constants."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_d_model_value():
    """D_MODEL should be GPT-2 hidden dimension."""
    from config import D_MODEL
    assert D_MODEL == 768


def test_expansion_value():
    """EXPANSION factor should be 32."""
    from config import EXPANSION
    assert EXPANSION == 32


def test_k_active_value():
    """K_ACTIVE (top-k sparsity) should be 32."""
    from config import K_ACTIVE
    assert K_ACTIVE == 32


def test_n_latents_derived():
    """N_LATENTS should equal D_MODEL * EXPANSION."""
    from config import D_MODEL, EXPANSION, N_LATENTS
    assert N_LATENTS == D_MODEL * EXPANSION
    assert N_LATENTS == 24576


def test_layer_idx_value():
    """LAYER_IDX should be 8 (GPT-2 layer for activation extraction)."""
    from config import LAYER_IDX
    assert LAYER_IDX == 8


def test_hook_pattern():
    """HOOK should be the correct TransformerLens hook pattern."""
    from config import HOOK, LAYER_IDX
    expected = f"blocks.{LAYER_IDX}.hook_resid_pre"
    assert HOOK == expected
    assert HOOK == "blocks.8.hook_resid_pre"


def test_window_size_value():
    """WINDOW_SIZE should be 64 tokens."""
    from config import WINDOW_SIZE
    assert WINDOW_SIZE == 64
