"""Tests for H5 sparse format utilities."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.h5_utils import (
    get_feature_activation_for_token,
    get_feature_activations_chunk,
    get_review_id_for_token
)


class TestGetFeatureActivationForToken:
    """Test single token/feature activation lookup."""

    def test_active_feature_returns_value(self, mock_h5_file):
        """Active feature should return its activation value."""
        with h5py.File(mock_h5_file, 'r') as f:
            # Token 0 has features 0-31 active with values 0.1-1.0
            # Feature 0 is the first one, should have value ~0.1
            val = get_feature_activation_for_token(f, 0, 0)
            assert val > 0
            assert isinstance(val, float)

    def test_inactive_feature_returns_zero(self, mock_h5_file):
        """Inactive feature should return 0.0."""
        with h5py.File(mock_h5_file, 'r') as f:
            # Token 0 has features 0-31 active
            # Feature 100 should NOT be active for token 0
            val = get_feature_activation_for_token(f, 0, 100)
            assert val == 0.0

    def test_different_tokens_different_features(self, mock_h5_file):
        """Different tokens should have different active features."""
        with h5py.File(mock_h5_file, 'r') as f:
            # Token 0: features 0-31
            # Token 1: features 32-63
            assert get_feature_activation_for_token(f, 0, 0) > 0
            assert get_feature_activation_for_token(f, 0, 32) == 0.0

            assert get_feature_activation_for_token(f, 1, 32) > 0
            assert get_feature_activation_for_token(f, 1, 0) == 0.0


class TestGetFeatureActivationsChunk:
    """Test chunked feature extraction."""

    def test_chunk_shape(self, mock_h5_file):
        """Chunk should return correct shape."""
        with h5py.File(mock_h5_file, 'r') as f:
            chunk = get_feature_activations_chunk(f, feature_idx=0, start_token=0, end_token=10)
            assert chunk.shape == (10,)
            assert chunk.dtype == np.float32

    def test_chunk_values(self, mock_h5_file):
        """Chunk should return correct activation values."""
        with h5py.File(mock_h5_file, 'r') as f:
            # Feature 0 is only active for token 0
            chunk = get_feature_activations_chunk(f, feature_idx=0, start_token=0, end_token=5)

            # Token 0 should have non-zero activation for feature 0
            assert chunk[0] > 0

            # Tokens 1-4 should have zero activation for feature 0
            # (they have features 32-63, 64-95, etc.)
            assert chunk[1] == 0.0
            assert chunk[2] == 0.0

    def test_full_range_chunk(self, mock_h5_file):
        """Test extracting full token range."""
        with h5py.File(mock_h5_file, 'r') as f:
            chunk = get_feature_activations_chunk(f, feature_idx=0, start_token=0, end_token=100)
            assert chunk.shape == (100,)

            # Only token 0 should have feature 0 active
            assert (chunk > 0).sum() == 1


class TestGetReviewIdForToken:
    """Test review ID lookup."""

    def test_returns_string(self, mock_h5_file):
        """Should return string review ID."""
        with h5py.File(mock_h5_file, 'r') as f:
            rev_id = get_review_id_for_token(f, 0)
            assert isinstance(rev_id, str)

    def test_correct_review_ids(self, mock_h5_file):
        """Should return correct review ID for token position."""
        with h5py.File(mock_h5_file, 'r') as f:
            # First 50 tokens are review_1
            assert get_review_id_for_token(f, 0) == 'review_1'
            assert get_review_id_for_token(f, 49) == 'review_1'

            # Next 50 tokens are review_2
            assert get_review_id_for_token(f, 50) == 'review_2'
            assert get_review_id_for_token(f, 99) == 'review_2'

    def test_handles_bytes(self, mock_h5_file):
        """Should handle bytes encoding from H5."""
        with h5py.File(mock_h5_file, 'r') as f:
            # The fixture stores bytes, function should decode
            rev_id = get_review_id_for_token(f, 0)
            assert 'review' in rev_id
            assert isinstance(rev_id, str)
