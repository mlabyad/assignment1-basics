
from __future__ import annotations

import os

import regex as re
from multiprocessing import Pool
from functools import partial

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
    encoded_special_tokens = [token.encode("utf-8") for token in special_tokens]
    corpus_chunks = read_corpus(input_path, nbr_chunks=4, pct=1, special_tokens=encoded_special_tokens)
    pre_tokenized_corpus = pre_tokenize(input_path, corpus_chunks, special_tokens)
    vocab, merges = learn_bpe(pre_tokenized_corpus, vocab_size, encoded_special_tokens)
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
    return sorted(set(chunks))

def pre_tokenizer(chunk, reg, special_tokens, path) -> dict[tuple[bytes, ...], int]:
    dic = {}
    file = open(path, "rb")
    file.seek(chunk[0])
    chunk_str = file.read(chunk[1]-chunk[0]).decode("UTF-8")
    pattern = "|".join(re.escape(token) for token in special_tokens)
    docs = re.split(pattern, chunk_str)
    for doc in docs:
        iter = re.finditer(reg, doc)
        c = next(iter, None)
        while c:
            token = c.group()
            tt = tuple(bytes([b]) for b in token.encode("UTF-8"))
            if tt in dic:
                dic[tt] += 1
            else:
                dic[tt] = 1
            c = next(iter, None)
    return dic

def pre_tokenize(path, boundaries, special_tokens):
    chunks = []
    for b in range(1, len(boundaries)):
        chunks.append((boundaries[b-1], boundaries[b]))
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    worker = partial(pre_tokenizer, reg=pat, special_tokens=special_tokens, path=path)

    with Pool(processes=4) as pool:
        results = pool.map(worker, chunks)
    
    res = {}
    for d in results:
        for k in d:
            if k in res:
                res[k] += d[k]
            else:
                res[k] = d[k]
    return res

def compute_pair_counts(corpus):
    pair_count = {}
    for k, v in corpus.items():
        for i in range(len(k) - 1):
            p = (k[i], k[i+1])
            pair_count[p] = pair_count.get(p, 0) + v
    return pair_count

def pair_count_delta(pair_count, changes):
    for old_tt, new_tt, count in changes:
        for i in range(len(old_tt) - 1):
            p = (old_tt[i], old_tt[i+1])
            pair_count[p] -= count
            if pair_count[p] <= 0:
                del pair_count[p]
        for i in range(len(new_tt) - 1):
            p = (new_tt[i], new_tt[i+1])
            pair_count[p] = pair_count.get(p, 0) + count
    return pair_count

def learn_bpe(pre_tokenized_corpus: dict[tuple[bytes, ...], int], vocab_size: int, special_tokens) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    init_vocab: dict[int, bytes] = {}
    pos = 0
    for i in special_tokens:
        init_vocab[pos] = i
        pos += 1
    pos = len(special_tokens)
    for i in range(256):
        init_vocab[pos + i] = bytes([i])
    pos += 256
    merges: list[tuple[bytes, bytes]] = []
    pair_count = compute_pair_counts(pre_tokenized_corpus)
    while pos < vocab_size:

        m = max(pair_count, key=lambda k: (pair_count[k], k[0], k[1]))
        pre_tokenized_corpus, changes = update_corpus(pre_tokenized_corpus, m)
        pair_count = pair_count_delta(pair_count, changes)
        init_vocab[pos] = m[0] + m[1]
        pos += 1
        merges.append(m)

    return init_vocab, merges

def update_corpus(corpus, pair):
    a, b = pair
    merged = a + b
    changes = []
    for tt, count in corpus.items():
        if a not in tt:
            continue
        new_tuple = []
        i = 0; n = len(tt); changed = False
        while i < n:
            if i < n - 1 and tt[i] == a and tt[i+1] == b:
                new_tuple.append(merged); i += 2; changed = True
            else:
                new_tuple.append(tt[i]); i += 1
        if changed:
            changes.append((tt, tuple(new_tuple), count))
    for old, new, count in changes:
        del corpus[old]
        corpus[new] = corpus.get(new, 0) + count
    return corpus, changes


if __name__ == "__main__":

    input_path = "C:/Users/D641771/Desktop/projects/AI/assignment1-basics/data/owt_train.txt"
    input_path = "C:/Users/D641771/Desktop/projects/AI/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    import cProfile, pstats


    profiler = cProfile.Profile()
    profiler.enable()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    profiler.disable()

    import pickle

    with open("ts_vocab.pkl", "wb") as f:
        pickle.dump(merges, f)

    with open("ts_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    stats = pstats.Stats(profiler)
    stats.dump_stats("profile_out.prof")
    stats.sort_stats("cumulative").print_stats(20)
