from functools import lru_cache
from typing import Iterable
from idna import encode
from networkx import multi_source_dijkstra_path_length
import regex as re
import json
import os

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        """
        Args:
            vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
            special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
                be split into multiple tokens, and will always be kept as a single token.
        """
        self.vocab = vocab
        self.vocab_inverse = {v: k for k, v in self.vocab.items()}
        # 将 list 转为 dict，记录每个 pair 的 rank (索引)
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        '''
        Load a BPETokenizer from vocab and merges files.
        Args:
            vocab_filepath (str): Path to the vocabulary file (JSON format).
            merges_filepath (str): Path to the merges file (text format).
            special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
                be split into multiple tokens, and will always be kept as a single token.
        Returns:
            BPETokenizer: An instance of BPETokenizer loaded from the specified files.
        '''
        byte_encoder = cls._bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        with open(vocab_filepath, 'r', encoding="utf-8") as f:
            vocab = json.load(f)
        vocab = {v: bytes([byte_decoder[ch] for ch in k])
                 for k, v in vocab.items()}

        with open(merges_filepath, 'r', encoding="utf-8") as f:
            merges = f.read().split("\n")[:-1]  # remove the last empty line
            # remove the space between two str
            merges = [tuple(merge.split()) for merge in merges]
            merges = [
                (bytes([byte_decoder[ch] for ch in first]),
                 bytes([byte_decoder[ch] for ch in second]))
                for first, second in merges
            ]
        return cls(vocab, merges, special_tokens)

    def encode(
        self,
        text: str
    ) -> list[int]:
        token_ids: list[int] = []
        pre_tokens: list[bytes] = self._process_chunk(text, self.special_tokens)
        # If text is "Hello world! It's 42."
        # pre-tokens is [b'Hello', b' world', b'!', b' It', b"'s", b' 42', b'.']
        for token in pre_tokens:
            # print(f"Enter encoding loop, now token is {token}...")
            if self.special_tokens is not None and token.decode("utf-8") in self.special_tokens:
                token_ids.append(self.vocab_inverse[token])
                continue
            
            token_bytes: list[bytes] = list(bytes([b]) for b in token)
            while len(token_bytes) >= 2:
                pairs = list(zip(token_bytes, token_bytes[1:]))
                # print(f"Pairs of {token} now is {pairs}")
                pair_to_merge = None
                min_rank = float("inf")
                for pair in pairs:
                    rank = self.merges.get(pair, float("inf"))
                    if rank < min_rank:
                        min_rank = rank
                        pair_to_merge = pair
                if pair_to_merge is None:
                    break
                token_bytes = self._merge_pair(token_bytes, pair_to_merge)

            for t in token_bytes:
                token_ids.append(self.vocab_inverse[t])

        return token_ids

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(
        self,
        ids: list[int]
    ) -> str:
        text_bytes: bytes = bytes()
        for id in ids:
            text_bytes += self.vocab[id]
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    @staticmethod
    def _merge_pair(
        tokens: list[bytes],
        pair: tuple[bytes, bytes]
    ) -> list[bytes]:
        """Merge all occurrences of a specified byte pair in a list of byte tokens.
        Args:
            tokens (list[bytes]): A list of byte tokens representing a sequence.
            pair (tuple[bytes, bytes]): A tuple containing the byte pair to be merged.
        Returns:
            list[bytes]: A new list of byte tokens with all occurrences of the specified
                byte pair merged into single tokens.
        """
        new_token: list[bytes] = []

        i = 0
        while i < len(tokens):
            if i < (len(tokens) - 1) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_token.append(pair[0] + pair[1])
                i += 2
            else:
                new_token.append(tokens[i])
                i += 1
        return new_token

    @staticmethod
    @lru_cache()
    def _bytes_to_unicode():
        """
        **FROM https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gpt2/tokenization_gpt2.py#L37**
        Returns list of utf-8 byte and a mapping to unicode strings.
        We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

        The reversible bpe codes work on unicode strings.
        This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        This is a signficant percentage of your normal, say, 32K bpe vocab.
        To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        """
        # bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        bs = (
            list(range(ord("!"), ord("~") + 1)) +
            list(range(ord("¡"), ord("¬") + 1)) +
            list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    @staticmethod
    # Considering its a static method, so just pass the special tokens as a parameter
    def _process_chunk(
        text: str,
        special_tokens: list[str] | None
    ) -> list[bytes] :
        pre_tokens: list[bytes] = []
        pattern = re.compile(GPT2_PAT)
        if special_tokens is not None:
            sorted_special_tokens = sorted(
                special_tokens, key=len, reverse=True)
            spe_pattern = f"({"|".join(re.escape(special_token)
                                       for special_token in sorted_special_tokens)})"
            chunks = re.split(spe_pattern, text) if spe_pattern else [text]
            for c in chunks:
                if c in special_tokens:
                    pre_tokens.extend([c.encode("utf-8")])
                else:
                    pre_tokens.extend([match.group(0).encode("utf-8")
                              for match in pattern.finditer(c)])
        else:
            pre_tokens.extend([match.group(0).encode("utf-8")
                              for match in pattern.finditer(text)])
        return pre_tokens  
        
