"""Text processing and tokenization."""

import torch
from transformers import AutoTokenizer

import sys
from pathlib import Path

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WINDOW_SIZE


def get_tokenizer() -> AutoTokenizer:
    """Get GPT-2 tokenizer with pad token configured."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': tok.eos_token})
    return tok


def tokenize_to_windows(text: str, tok: AutoTokenizer) -> torch.Tensor:
    """Tokenize text into 64-token windows.

    Args:
        text: Input text string
        tok: GPT-2 tokenizer

    Returns:
        Tensor of shape (n_windows, 64) with token IDs, or None if empty
    """
    ids = tok(text, add_special_tokens=False).input_ids

    if not ids:
        return None

    windows = []
    for i in range(0, len(ids), WINDOW_SIZE):
        window = ids[i:i + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            window = window + [tok.pad_token_id] * (WINDOW_SIZE - len(window))
        windows.append(window)

    return torch.tensor(windows)


def get_token_strings(text: str, tok: AutoTokenizer) -> list[str]:
    """Get list of token strings for text."""
    ids = tok(text, add_special_tokens=False).input_ids
    return tok.convert_ids_to_tokens(ids)
