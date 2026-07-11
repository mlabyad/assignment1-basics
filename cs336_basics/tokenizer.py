
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

    corpus_chuncks = read_corpus(input_path, nbr_chunks=4, pct=1, special_tokens=special_tokens)
    pre_tokenized_corpus = pre_tokenize(corpus_chuncks, special_tokens)
    vocab, merges = learn_bpe(pre_tokenized_corpus, vocab_size, special_tokens)
    return vocab, merges

def read_corpus(input_path: str, nbr_chunks: int = 3, pct: float = 1, special_tokens: list[bytes] = None) -> list[int]:
    """Read the corpus from the given path and return a list of strings.

    Args:
        input_path (str): Path to BPE tokenizer training data.
        pct (float, optional): Percentage of the corpus to read. Defaults to 1 (read all).

    Returns:
        list[str]: List of strings in the corpus.
    """
    file = open(input_path, "rb")
    file.seek(0, os.SEEK_END)
    total_size = file.tell()
    file.seek(0)

    last_byte = int(total_size * pct)
    chunk_size = int(last_byte // nbr_chunks)

    chunks = [i * chunk_size for i in range(nbr_chunks+1)]
    chunks[-1] = last_byte
    print(total_size)
    print(last_byte)
    print(chunks)

    mini_chunk_size = 4028
    for i in range(1, len(chunks)-1):
        curr_position = chunks[i]
        file.seek(chunks[i])
        while True:
            chunk_content = file.read(mini_chunk_size)

            if chunk_content == b"":
                chunks[i] = last_byte
                break

            best_pos = -1
            for token in special_tokens:
                pos = chunk_content.find(token)
                if pos != -1 and (best_pos == -1 or pos < best_pos):
                    best_pos = pos

            if best_pos != -1:
                chunks[i] = best_pos + curr_position
                break
            curr_position += mini_chunk_size
        print(i)
        print(chunks[i])
    return sorted(set(chunks))


if __name__ == "__main__":
    path = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    encoded_special_tokens = [token.encode("utf-8") for token in special_tokens]
    boundaries = read_corpus(path, 6, 0.01, encoded_special_tokens)

    with open(path, "rb") as file:
        file.seek(boundaries[0])
        print(file.read(boundaries[1] - boundaries[0]))
        print("\n")
        print("\n")
        print("\n")
        file.seek(boundaries[1])
        print(file.read(boundaries[2] - boundaries[1]))
