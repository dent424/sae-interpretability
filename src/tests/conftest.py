"""Shared fixtures for SAE interpretability tests."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))


@pytest.fixture
def mock_h5_file(tmp_path):
    """Create a mock H5 file with known sparse values.

    Structure matches real H5:
    - z_idx: (n_tokens, K) active feature indices per token
    - z_val: (n_tokens, K) corresponding activation values
    - rev_idx: (n_tokens,) review ID for each token
    """
    import h5py

    n_tokens = 100
    k_active = 32

    path = tmp_path / "test_sae_features.h5"

    with h5py.File(path, 'w') as f:
        # Each token has features [i*32, i*32+1, ..., i*32+31] active
        # This makes it easy to test: token 0 has features 0-31,
        # token 1 has features 32-63, etc.
        z_idx = np.zeros((n_tokens, k_active), dtype=np.int32)
        z_val = np.zeros((n_tokens, k_active), dtype=np.float32)

        for i in range(n_tokens):
            z_idx[i] = np.arange(i * k_active, (i + 1) * k_active) % 24576
            z_val[i] = np.linspace(0.1, 1.0, k_active)  # Activation values 0.1-1.0

        f.create_dataset('z_idx', data=z_idx)
        f.create_dataset('z_val', data=z_val)

        # Review IDs - first 50 tokens are review_1, next 50 are review_2
        rev_ids = [b'review_1'] * 50 + [b'review_2'] * 50
        f.create_dataset('rev_idx', data=rev_ids)

    return path


@pytest.fixture
def mock_metadata_dict():
    """Create mock metadata similar to NPZ file contents."""
    return {
        'review_ids': np.array(['review_1', 'review_2', 'review_3']),
        'texts': np.array([
            'The tacos here are amazing!',
            'Worst service I have ever experienced.',
            'Pretty good for the price.'
        ]),
        'stars': np.array([5, 1, 3]),
        'useful': np.array([10, 5, 2]),
        'user_ids': np.array(['user_a', 'user_b', 'user_c']),
        'business_ids': np.array(['biz_1', 'biz_1', 'biz_2'])
    }


@pytest.fixture
def mock_npz_file(tmp_path, mock_metadata_dict):
    """Create a mock NPZ file with metadata."""
    path = tmp_path / "test_metadata.npz"
    np.savez(path, **mock_metadata_dict)
    return path


@pytest.fixture
def sample_text():
    """Sample text for tokenization tests."""
    return "The food at this restaurant was absolutely amazing!"


@pytest.fixture
def sample_texts():
    """Multiple sample texts for batch tests."""
    return [
        "Best tacos I've ever had!",
        "The service was slow but the food made up for it.",
        "Why in the world would anyone eat here?",
        "UPDATE: Came back and it was even better!"
    ]
