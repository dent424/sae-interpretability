"""Tests for text processing and tokenization."""

import sys
from pathlib import Path

import torch
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.text import get_tokenizer, tokenize_to_windows, get_token_strings
from config import WINDOW_SIZE


class TestGetTokenizer:
    """Test tokenizer initialization."""

    def test_returns_tokenizer(self):
        """Should return a tokenizer object."""
        tok = get_tokenizer()
        assert tok is not None

    def test_has_pad_token(self):
        """Tokenizer should have pad token configured."""
        tok = get_tokenizer()
        assert tok.pad_token is not None
        assert tok.pad_token_id is not None

    def test_encodes_text(self):
        """Should be able to encode text."""
        tok = get_tokenizer()
        result = tok("Hello world", add_special_tokens=False)
        assert 'input_ids' in result
        assert len(result.input_ids) > 0


class TestTokenizeToWindows:
    """Test window-based tokenization."""

    def test_returns_tensor(self, sample_text):
        """Should return a PyTorch tensor."""
        tok = get_tokenizer()
        windows = tokenize_to_windows(sample_text, tok)
        assert isinstance(windows, torch.Tensor)

    def test_window_size(self, sample_text):
        """Each window should have WINDOW_SIZE tokens."""
        tok = get_tokenizer()
        windows = tokenize_to_windows(sample_text, tok)
        assert windows.shape[1] == WINDOW_SIZE
        assert windows.shape[1] == 64

    def test_short_text_padded(self):
        """Short text should be padded to window size."""
        tok = get_tokenizer()
        windows = tokenize_to_windows("Hi", tok)

        assert windows.shape == (1, WINDOW_SIZE)
        # Should have pad tokens at the end
        assert (windows[0] == tok.pad_token_id).sum() > 0

    def test_long_text_multiple_windows(self):
        """Long text should produce multiple windows."""
        tok = get_tokenizer()
        # Create text longer than 64 tokens
        long_text = " ".join(["word"] * 100)
        windows = tokenize_to_windows(long_text, tok)

        assert windows.shape[0] > 1  # Multiple windows
        assert windows.shape[1] == WINDOW_SIZE

    def test_empty_text_returns_none(self):
        """Empty text should return None."""
        tok = get_tokenizer()
        result = tokenize_to_windows("", tok)
        assert result is None

    def test_first_token_not_pad(self, sample_text):
        """First token should not be a pad token (for non-empty text)."""
        tok = get_tokenizer()
        windows = tokenize_to_windows(sample_text, tok)
        assert windows[0, 0] != tok.pad_token_id


class TestGetTokenStrings:
    """Test token string extraction."""

    def test_returns_list(self, sample_text):
        """Should return a list of strings."""
        tok = get_tokenizer()
        tokens = get_token_strings(sample_text, tok)
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_correct_token_count(self):
        """Token count should match tokenizer output."""
        tok = get_tokenizer()
        text = "Hello world"

        tokens = get_token_strings(text, tok)
        ids = tok(text, add_special_tokens=False).input_ids

        assert len(tokens) == len(ids)

    def test_handles_special_characters(self):
        """Should handle GPT-2 special tokens (like Ġ prefix)."""
        tok = get_tokenizer()
        tokens = get_token_strings("Hello world", tok)

        # GPT-2 uses Ġ for tokens after whitespace
        # "Hello" -> "Hello", " world" -> "Ġworld"
        assert len(tokens) == 2
        # At least one should have the Ġ prefix
        assert any('Ġ' in t or t == 'world' for t in tokens)

    def test_empty_text_empty_list(self):
        """Empty text should return empty list."""
        tok = get_tokenizer()
        tokens = get_token_strings("", tok)
        assert tokens == []
