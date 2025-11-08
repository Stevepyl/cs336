import os
import regex as re
from typing import BinaryIO
from collections import Counter
from typing import Iterable
from itertools import pairwise
from .constants import ENDOFTEXT, PAT
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],

) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given a path to an input text file, trains a (byte-level) BPE
    tokenizer. Your BPE training function should handle (at least) the following input parameters:

    Args:
        input_path (str | os.PathLike): 
            Path to a text file with BPE tokenizer training data.
        vocab_size (int): 
            A positive integer that defines the maximum final vocabulary size (including the
            initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens (list[str]):
            A list of strings to add to the vocabulary. 
            These special tokens do not otherwise affect BPE training

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
            merges: A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), 
                    representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
    """

    vocab: dict[int, bytes] = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    pattern = re.compile(PAT)

    pairs: dict[tuple[bytes, bytes], int] = {}
    merge_time = vocab_size - 256
    with open(input_path, "rb") as f:
        # with open("cs336_basics/bpe/test.txt", "rb") as f:

        num_processes = 1
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token

            pre_token_counts: dict[tuple[bytes, ...], int] = Counter(
                tuple(bytes([b]) for b in m.group().encode("utf-8")) for m in pattern.finditer(chunk)
            )
            for pre_token_byte, count in pre_token_counts.items():
                token_pair_count = get_pair_count_fast(pre_token_byte, count)
                for pair, count in token_pair_count.items():
                    pairs[pair] = pairs.get(pair, 0) + count

    merges = merge(pre_token_counts, pairs, merge_time)

    idx = len(vocab)
    for (t0, t1) in merges:
        # print(f"Merging {t0} and {t1} into {t0 + t1}")
        vocab[idx] = t0 + t1
        idx += 1

    # for k, v in vocab.items():
    #     print(f"{k}: {v}")
    return (vocab, merges)


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    for i in range(len(special_tokens)):
        vocab[i] = special_tokens[i].encode("utf-8")
    for i in range(256):
        vocab[i + len(special_tokens)] = bytes([i])
    return vocab


def get_pair_count(utf8_encoded_tokens: list[int], weight: int) -> dict[tuple[int, int], int]:
    counts = {}
    for pair in zip(utf8_encoded_tokens, utf8_encoded_tokens[1:]):
        counts[pair] = counts.get(pair, 0) + weight
    return counts


def get_pair_count_fast(
    utf8_encoded_tokens: Iterable[bytes],
    weight: int
) -> dict[tuple[bytes, bytes], int]:
    # 对于 bytes 和一般可迭代均适用；不创建额外列表
    # >= python 3.10
    pair_counts = Counter(pairwise(utf8_encoded_tokens))
    if weight != 1:
        for pair in pair_counts:
            pair_counts[pair] *= weight
    return pair_counts


def merge(
        pre_token_counts: dict[tuple[bytes, ...], int],
        pre_token_pairs: dict[tuple[bytes, bytes], int],
        merge_time: int
) -> list[tuple[bytes, bytes]]:
    """
    Merge the most common byte pairs until up to merge_time

    Args:
        pre_token_counts (dict[tuple[int, ...], int]): A dict whose key is byte sequence of pre-tokens, value is frequency
        pre_token_pairs (dict[tuple[int, int], int]): A dict whose key is byte pairs of pre-tokens, value is frequency
        merge_time (int): merge time(vocab_size - 256)

    Returns:
        list[tuple[bytes, bytes]]: 
            A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), 
            representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
    """
    merges: list[tuple[bytes, bytes]] = []

    for i in range(merge_time):
        # 这样会先按 count 降序（因为 max 取最大值），count 相同时再按 pair 本身的字典序升序（tuple 默认字典序比较）选择最大的 pair。
        top_pair = max(pre_token_pairs, key=lambda x: (pre_token_pairs[x], x))
        new_token_counts = {}

        for pre_token, count in pre_token_counts.items():
            j = 0
            new_token: list[bytes] = []
            while j < len(pre_token):
                if j < len(pre_token) - 1 and pre_token[j] == top_pair[0] and pre_token[j+1] == top_pair[1]:
                    # if pre_token[j] is the first element of top_pair, pre_token[j+1] is the second of it
                    new_token.append(top_pair[0] + top_pair[1])
                    j += 2
                else:  # if not one of top_pair
                    new_token.append(pre_token[j])
                    j += 1
            new_token_counts[tuple(new_token)] = count

        pre_token_counts = new_token_counts
        merges.append((top_pair[0], top_pair[1]))

        pre_token_pairs = {}
        for token, count in pre_token_counts.items():
            token_pair_count = get_pair_count_fast(token, count)
            for pair, pair_count in token_pair_count.items():
                pre_token_pairs[pair] = pre_token_pairs.get(
                    pair, 0) + pair_count
    # For debug
    #     print(
    #         f"Finish {i}-th merge, merging {merges[-1][0]} with {merges[-1][1]}")
    #     print(f"Now token seq is: ")
    #     for t, c in pre_token_counts.items():
    #         print(f"    {t}: {c}")
    # for merge in merges:
    #     print(merge)
    return merges


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token,
                      bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


if __name__ == "__main__":
    # text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    # utf_8_tokens = text.encode("utf-8")
    train_bpe("./", 1024, ["<|endoftext|>"])
