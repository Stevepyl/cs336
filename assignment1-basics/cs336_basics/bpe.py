import os
from secrets import token_bytes
import time
import regex as re
from typing import BinaryIO
from multiprocessing import get_context
from collections import defaultdict

import bpe_cpp

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8
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
    # 1. Init vocab and merges
    vocab: dict[int, bytes] = _init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    cpp_merges: list[tuple[bytes, bytes]] = [] # merges from cpp version merge function
    merge_time = vocab_size - len(vocab)
    # Attention: Here the merge_time is NOT `vocab_size - 256`, we also need to consider special_tokens,
    # cause they're also part of vocab, so using vocab_size - len(vocab)

    global_token_counts: defaultdict[bytes, int] = defaultdict(int)
    # 2. Pre-tokenization
    
    pre_tokenization_start_time = time.perf_counter()
    
    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))

    with get_context("forkserver").Pool(processes=num_processes) as pool:
        for chunk_counts in pool.starmap(_process_chunk, [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]):
            for token, count in chunk_counts.items():
                global_token_counts[token] += count

    pre_tokenization_end_time = time.perf_counter()
    print(f"Pre-Tokenization using: {(pre_tokenization_end_time - pre_tokenization_start_time):.2f}s")

    # 3. Compute merges
    pre_tokens: dict[bytes, list[bytes]] = {}
    for token in global_token_counts:
        pre_tokens[token] = list(bytes([b]) for b in token)
        
    pair_to_tokens, pair_counts = _get_pair_count(global_token_counts, pre_tokens)
    
    merge_start_time = time.perf_counter()
    # merges = _merge(pre_tokens, global_token_counts, pair_counts, pair_to_tokens, merge_time)
    cpp_merges = bpe_cpp.merge_cpp(pre_tokens, global_token_counts, pair_counts, pair_to_tokens, merge_time)
    merge_end_time = time.perf_counter()
    print(f"\nTime using of merges: {merge_end_time - merge_start_time:.2f}s")

    # 4. compute vocabs
    idx = len(vocab)
    for (t0, t1) in cpp_merges:
        vocab[idx] = t0 + t1
        idx += 1
    return (vocab, cpp_merges)


def _init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    for i in range(len(special_tokens)):
        vocab[i] = special_tokens[i].encode("utf-8")
    for i in range(256):
        vocab[i + len(special_tokens)] = bytes([i])
    return vocab


def _get_pair_count(
    pre_token_counts: defaultdict[bytes, int],
    pre_tokens: dict[bytes, list[bytes]]
) -> tuple[
    defaultdict[tuple[bytes, bytes], set],
    defaultdict[tuple[bytes, bytes], int]
]:
    """Count pair count and return a byte-pair to its token-index-set

    Args:
        pre_tokens (list[list[bytes]]): pre_tokens

    Returns:
        tuple[ defaultdict[tuple[bytes, bytes], set], defaultdict[tuple[bytes, bytes], int] ]: 
            pair_to_tokens: defaultdict[tuple[bytes, bytes], set]
                pair_to_tokens 维护了一个从字节对到包含该字节对的 token 集合的映射
                例如：
                {
                    (h, e): {b'hello', b'she', b'he', },  # 字节对 (h, e)出现在 token b'hello', b'she', b'he'中
                    (t, h): {b'the', b'they'},      # 字节对 (t, h) 出现在token the以及they 中
                    ...
                }
                没有 pair_to_tokens 的情况：
                    当找到最频繁的字节对 max_pair 时，需要遍历所有 token 来查找哪些包含这个字节对
                    时间复杂度：O(n)，其中 n 是 token 总数
                有了 pair_to_tokens 的情况：
                    直接获取包含 max_pair 的所有 token 索引，然后更新这一个pair所在的token
                    时间复杂度：O(k)，其中 k 是包含该字节对的 token 数量（通常 k << n）
            counts: defaultdict[tuple[bytes, bytes], int]
    """
    pair_to_tokens = defaultdict(set)
    pair_counts = defaultdict(int)
    for token, count in pre_token_counts.items():
        token_bytes = pre_tokens[token]
        for pair in zip(token_bytes, token_bytes[1:]):
            pair_counts[pair] += count
            pair_to_tokens[pair].add(token)
    return pair_to_tokens, pair_counts


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


def _merge(
        pre_tokens: dict[bytes, list[bytes]],
        pre_token_counts: defaultdict[bytes, int],
        pair_counts: defaultdict[tuple[bytes, bytes], int],
        pair_to_tokens: defaultdict[tuple[bytes, bytes], set],
        merge_time: int
) -> list[tuple[bytes, bytes]]:
    """
        Perform BPE merges by iteratively merging the most frequent byte pair.
        This function executes a specified number of byte pair merges on a collection of
        tokenized sequences. For each merge iteration, it identifies the most frequent
        byte pair (with ties broken by lexicographic order), merges all occurrences of
        that pair, and updates the pair frequency counts and location tracking.
        Args:
            pre_tokens: A list of tokenized sequences, where each sequence is represented
                as a list of bytes. This list is modified in-place during merging.
            pair_counts: A dictionary mapping byte pairs to their occurrence counts across
                all tokens. Updated in-place as merges are performed.
            pair_to_tokens: A dictionary mapping byte pairs to sets of indices indicating
                which tokens in pre_tokens contain that pair. Updated in-place as merges
                are performed.
            merge_time: The number of merge operations to perform.
        Returns:
            A list of merged byte pairs in the order they were merged. Each element is a
            tuple of two bytes representing a merged pair.
        Note:
            - The function modifies pre_tokens, pair_counts, and pair_to_tokens in-place.
            - When multiple pairs have the same frequency, the lexicographically smallest
              pair is selected for merging.
            - Pairs with zero count are removed from pair_counts and pair_to_tokens.
        """
    merges: list[tuple[bytes, bytes]] = []
    for i in range(merge_time):
        # 这样会先按 count 降序（因为 max 取最大值），count 相同时再按 pair 本身的字典序升序（tuple 默认字典序比较）选择最大的 pair。
        top_pair = max(pair_counts, key=lambda x: (pair_counts[x], x))
        print(f"merging :{top_pair}")
        merges.append(top_pair)

        # Update affected tokens
        # For example, the top-pair is ('h', 'e'), then we need to update the bytes list of "The", "They", "she"...
        affected_tokens = pair_to_tokens[top_pair].copy()
        for affected_token in affected_tokens:
            affected_token_bytes = pre_tokens[affected_token]
            # Single bytes needn't to update
            if len(affected_token_bytes) < 2:
                continue

            # Decrement all pair counts in affected token
            for pair in zip(affected_token_bytes, affected_token_bytes[1:]):
                pair_counts[pair] -= pre_token_counts[affected_token]
                pair_to_tokens[pair].discard(affected_token)
                if 0 == pair_counts[pair]:
                    del pair_counts[pair]
                    del pair_to_tokens[pair]
            affected_token_bytes = _merge_pair(affected_token_bytes, top_pair)
            # Increment all pair counts in affected token
            for pair in zip(affected_token_bytes, affected_token_bytes[1:]):
                pair_counts[pair] += pre_token_counts[affected_token]
                pair_to_tokens[pair].add(affected_token)
            pre_tokens[affected_token] = affected_token_bytes
    return merges


def _find_chunk_boundaries(
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

    chunk_size = max(1, file_size // desired_num_chunks)

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


def _process_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str]
) -> defaultdict[bytes, int]:
    """
    Process a chunk of the input file for BPE training. 
    Processing steps:
    1. Read the specified byte range from the input file.
    2. Decode the bytes to a UTF-8 string, ignoring decode errors.
    3. Remove any occurrences of the specified special tokens.
    4. Pre-tokenize the cleaned string using the GPT-2 BPE regex pattern.
    5. Count the occurrences of each pre-token byte sequence.
    Parameters:
        input_path (str):
            Path to the input file to read.
        start (int):
            Inclusive starting byte offset within the file.
        end (int):
            Exclusive ending byte offset within the file (must be >= start).
        special_tokens (list[str]):
            List of literal strings to be stripped from the chunk prior to
            regex tokenization. Each is treated as a plain sequence (no regex
            metacharacters, they are escaped).
    Returns:
        defaultdict[bytes, int]:
            A mapping from pre-token byte sequences to their occurrence counts
            within the processed chunk.
    Notes:
        - Decode errors are ignored, which may drop malformed byte sequences.
        - Special tokens are removed entirely (not left as placeholders).
        - The granularity of the returned structure (byte lists per token) is
          intended for subsequent byte-pair merge frequency analysis.
        - The function does not perform any normalization beyond removal of
          provided special tokens.
    """

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    # 1. Remove special tokens in chunk, split the document into pieces, splitter is special token
    pattern = "|".join(re.escape(special_token)
                       for special_token in special_tokens)
    chunks_without_spe_token = re.split(pattern, chunk)

    # 2. Pre-tokenize
    # using the pattern to split the chunks into 'pre-tokens' like "The", "  ", "hello"...
    pre_token_counts: defaultdict[bytes, int] = defaultdict(int)
    pattern = re.compile(GPT2_PAT)
    for c in chunks_without_spe_token:
        if not c.strip():
            continue
        tokens = [match.group(0).encode("utf-8")
                  for match in pattern.finditer(c)]
        for token in tokens:
            pre_token_counts[token] += 1
    return pre_token_counts
