from functools import lru_cache
from typing import Iterable
import regex as re
import json
import heapq

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
GPT2_REGEX = re.compile(GPT2_PAT)


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
        self.vocab_inverse: dict[bytes, int] = {
            v: k for k, v in self.vocab.items()}
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
        # If text is "Hello world! It's 42."
        # pre-tokens is [b'Hello', b' world', b'!', b' It', b"'s", b' 42', b'.']
        for token in self._process_chunk(text, self.special_tokens):
            # print(f"Enter encoding loop, now token is {token}...")
            if self.special_tokens is not None and token.decode("utf-8") in self.special_tokens:
                token_ids.append(self.vocab_inverse[token])
                continue
            token_ids.extend(self._bpe_merge_word(token))

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

    # a small number of words account for the vast majority of text occurrences.
    # using cache to store the most frequent tokens
    # avoid repeat merge
    # get 5x improvment
    @lru_cache(maxsize=5000)
    def _bpe_merge_word(self, token: bytes) -> list[int]:
        token_bytes: list[bytes] = [token[i:i+1] for i in range(len(token))]
        length = len(token_bytes)

        if length < 2:
            return [self.vocab_inverse[t] for t in token_bytes]

        next_idx = list(range(1, length + 1))
        next_idx[-1] = -1  # -1 indicates the end of the list
        prev_idx = list(range(-1, length - 1))  # -1 indicates the start

        # Stores tuples of (rank, index), where 'index' represents the pair (token_bytes[index], token_bytes[next_idx[index]])
        pq = []

        def get_rank(i):
            # Helper to get rank of pair starting at i
            if i == -1 or next_idx[i] == -1:
                return float('inf')
            pair = (token_bytes[i], token_bytes[next_idx[i]])
            return self.merges.get(pair, float('inf'))

        def push_pair(i):
            # Helper to push valid pairs to heap
            rank = get_rank(i)
            if rank != float('inf'):
                heapq.heappush(pq, (rank, i))

        for i in range(length - 1):
            push_pair(i)

        # Initial population of the heap
        while pq:
            rank, i = heapq.heappop(pq)
            # Check if the rank in the heap matches the ACTUAL current rank.
            # If token_bytes[i] or token_bytes[next_idx[i]] changed due to other merges, get_rank(i) will differ.
            if get_rank(i) != rank:
                continue

            j = next_idx[i]
            # A. Update token at 'i' (Merge j into i)
            token_bytes[i] += token_bytes[j]

            # delete and re-link the link list
            k = next_idx[j]
            next_idx[i] = k
            next_idx[j] = -1
            if k != -1:
                prev_idx[k] = i

            # The pair starting at 'i' has changed (it's now i+k instead of i+j)
            push_pair(i)
            # The pair ending at 'i' (starting at prev_idx[i]) has changed target
            if prev_idx[i] != -1:
                push_pair(prev_idx[i])

        result: list[int] = []
        curr = 0
        while curr != -1:
            result.append(self.vocab_inverse[token_bytes[curr]])
            curr = next_idx[curr]

        return result

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
    ) -> Iterable[bytes]:
        pre_tokens: list[bytes] = []
        if special_tokens is not None:
            sorted_special_tokens = sorted(
                special_tokens, key=len, reverse=True)
            spe_pattern = f"({"|".join(re.escape(special_token)
                                       for special_token in sorted_special_tokens)})"
            chunks = re.split(spe_pattern, text) if spe_pattern else [text]
            for c in chunks:
                if c in special_tokens:
                    yield c.encode("utf-8")
                else:
                    for match in GPT2_REGEX.finditer(c):
                        yield match.group(0).encode("utf-8")
        else:
            for match in GPT2_REGEX.finditer(text):
                yield match.group(0).encode("utf-8")
