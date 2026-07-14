from typing import Iterable, Iterator

import regex as re

class Tokenizer:
    def __init__(self, vocab:dict[int, bytes] , merges:list[tuple[bytes, bytes]] , special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        # Class method  that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the 
        # same format that your BPE training code output) and (optionally) a list of special tokens. 
        import pickle
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        # Encode an input text into a sequence of token IDs.
        token_ids: list[int] = []

        for piece, is_special in self._split_special_tokens(text):
            if is_special:
                token_ids.append(self.byte_to_id[piece.encode("utf-8")])
                continue

            for pretoken in re.finditer(self.pat, piece):
                token_bytes = pretoken.group().encode("utf-8")
                symbols = [bytes([byte]) for byte in token_bytes]
                merged_symbols = self._apply_merges(symbols)
                token_ids.extend(self.byte_to_id[symbol] for symbol in merged_symbols)

        return token_ids

    def _split_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        if not self.special_tokens:
            return [(text, False)]

        pattern = "|".join(
            re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
        )
        pieces: list[tuple[str, bool]] = []
        start = 0

        return pieces

    def _apply_merges(self, symbols: list[bytes]) -> list[bytes]:
        merged = symbols[:]



        return merged

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is 
        #required for memory-efficient tokenization of large files that we cannot directly load into memory.
        for piece in iterable:
            for token_id in self.encode(piece):
                yield token_id

    def decode(self, ids: list[int]) -> str: 
        # Decode a sequence of token IDs into text. To test your Tokenizer against our provided tests, you will first need to implement the test 
        # adapter at [adapters.get_tokenizer] . Then, run uv run pytest tests/test_tokenizer.py.
        decoded_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return decoded_bytes.decode("utf-8", errors="replace")