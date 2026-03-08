import os
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from cs336_basics import BPETokenizer

TS_TOKENIZER_PATH = "./tokenizer/tinystories/"
OWT_TOKENIZER_PATH = "./tokenizer/owt/"
DATA_PATH = "./data/"

# Module-level global so each worker process holds its own tokenizer instance.
_worker_tokenizer: BPETokenizer | None = None


def _init_worker(vocab_path: str, merges_path: str, special_tokens: list[str]):
    """Called once per worker process to load the tokenizer into a global."""
    global _worker_tokenizer
    _worker_tokenizer = BPETokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )


def _encode_chunk(lines: list[str]) -> list[int]:
    """Encode a chunk of lines into a flat token ID list."""
    assert _worker_tokenizer is not None
    return list(_worker_tokenizer.encode_iterable(lines))


def _chunk_file(filepath: str, chunk_lines: int):
    """Generator that yields lists of lines from a file in fixed-size batches."""
    with open(filepath, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            batch.append(line)
            if len(batch) >= chunk_lines:
                yield batch
                batch = []
        if batch:
            yield batch


def _count_lines(filepath: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def encode_to_bin(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
    dataset: str,
    chunk_lines: int = 50000,
    num_workers: int = 8,
):
    txt_path = os.path.join(DATA_PATH, f"{dataset}.txt")
    bin_path = os.path.join(DATA_PATH, f"encoded/{dataset}.bin")

    total_lines = _count_lines(txt_path)
    total_chunks = (total_lines + chunk_lines - 1) // chunk_lines

    initargs = (vocab_path, merges_path, special_tokens)
    with open(bin_path, "wb") as out_f:
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=initargs) as pool:
            chunks = _chunk_file(txt_path, chunk_lines)
            for encoded_docs in tqdm(
                pool.imap(_encode_chunk, chunks),
                total=total_chunks,
                desc=f"Encoding {dataset}",
                unit=" chunks",
            ):
                arr = np.array(encoded_docs, dtype=np.uint16)
                arr.tofile(out_f)

    print(f"Finish encoding {dataset}")


def main():
    ts_vocab = os.path.join(TS_TOKENIZER_PATH, "vocab.json")
    ts_merges = os.path.join(TS_TOKENIZER_PATH, "merges.txt")
    ts_special = ["<|endoftext|>"]

    print(f"Loading tokenizer from {TS_TOKENIZER_PATH}")
    encode_to_bin(ts_vocab, ts_merges, ts_special, "TinyStoriesV2-GPT4-train")
    encode_to_bin(ts_vocab, ts_merges, ts_special, "TinyStoriesV2-GPT4-valid")

    owt_vocab = os.path.join(OWT_TOKENIZER_PATH, "vocab.json")
    owt_merges = os.path.join(OWT_TOKENIZER_PATH, "merges.txt")
    owt_special = ["<|endoftext|>"]
    encode_to_bin(owt_vocab, owt_merges, owt_special, "owt_train")
    encode_to_bin(owt_vocab, owt_merges, owt_special, "owt_valid")


if __name__ == "__main__":
    main()
