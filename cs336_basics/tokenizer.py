
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    corpus = read_corpus(input_path, pct=1)
    pre_tokenized_corpus = pre_tokenize(corpus, special_tokens)
    vocab, merges = learn_bpe(pre_tokenized_corpus, vocab_size, special_tokens)
    return vocab, merges

def read_corpus(input_path: str | os.PathLike, pct: float = 1) -> list[str]:
    """Read the corpus from the given path and return a list of strings.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        pct (float, optional): Percentage of the corpus to read. Defaults to 1 (read all).

    Returns:
        list[str]: List of strings in the corpus.
    """
