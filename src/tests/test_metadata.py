"""Tests for metadata loading utilities."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.metadata import load_metadata, create_review_lookup


class TestLoadMetadata:
    """Test NPZ metadata loading."""

    def test_returns_dataframe(self, mock_npz_file):
        """Should return a pandas DataFrame."""
        df = load_metadata(str(mock_npz_file))
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, mock_npz_file):
        """DataFrame should have all required columns."""
        df = load_metadata(str(mock_npz_file))

        required_columns = ['review_id', 'full_text', 'stars', 'useful', 'user_id', 'business_id']
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_correct_row_count(self, mock_npz_file, mock_metadata_dict):
        """DataFrame should have correct number of rows."""
        df = load_metadata(str(mock_npz_file))
        expected_rows = len(mock_metadata_dict['review_ids'])
        assert len(df) == expected_rows

    def test_data_values_preserved(self, mock_npz_file, mock_metadata_dict):
        """Data values should be preserved correctly."""
        df = load_metadata(str(mock_npz_file))

        # Check first review
        assert df.iloc[0]['full_text'] == mock_metadata_dict['texts'][0]
        assert df.iloc[0]['stars'] == mock_metadata_dict['stars'][0]


class TestCreateReviewLookup:
    """Test review ID to text lookup creation."""

    def test_returns_dict(self, mock_npz_file):
        """Should return a dictionary."""
        df = load_metadata(str(mock_npz_file))
        lookup = create_review_lookup(df)
        assert isinstance(lookup, dict)

    def test_lookup_by_review_id(self, mock_npz_file, mock_metadata_dict):
        """Should be able to look up text by review ID."""
        df = load_metadata(str(mock_npz_file))
        lookup = create_review_lookup(df)

        review_id = str(mock_metadata_dict['review_ids'][0])
        expected_text = str(mock_metadata_dict['texts'][0])

        assert review_id in lookup
        assert lookup[review_id] == expected_text

    def test_all_reviews_in_lookup(self, mock_npz_file, mock_metadata_dict):
        """All reviews should be in the lookup."""
        df = load_metadata(str(mock_npz_file))
        lookup = create_review_lookup(df)

        for review_id in mock_metadata_dict['review_ids']:
            assert str(review_id) in lookup

    def test_missing_review_raises_keyerror(self, mock_npz_file):
        """Looking up non-existent review should raise KeyError."""
        df = load_metadata(str(mock_npz_file))
        lookup = create_review_lookup(df)

        with pytest.raises(KeyError):
            _ = lookup['nonexistent_review']
