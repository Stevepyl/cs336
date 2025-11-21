import os
import time
import json
import argparse
from functools import lru_cache
from cs336_basics import train_bpe


def main():
    parser = argparse.ArgumentParser(
        description="bpe train"
    )

    parser.add_argument(
        "--output_dir",
        type=str
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/TinyStoriesV2-GPT4-train.txt",
    )

    args, unknown_args = parser.parse_known_args()
    print(f"{args=}")
    print(f"{unknown_args=}")
    vocab_size = args.vocab_size
    data_path = args.data_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    special_tokens = ["<|endoftext|>"]

    print(f"Train BPE tokenizer on {data_path}, vocab_size is {vocab_size}")
    start_time = time.perf_counter()
    vocab, merges = train_bpe(data_path, vocab_size, special_tokens, 8)
    end_time = time.perf_counter()
    save_tokenizer(vocab, merges, output_dir)
    print(f"total train time: {end_time - start_time:.2f} s")
    
    # For validate correctness of byte_to_unicode
    # byte_encoder = bytes_to_unicode()
    # byte_decoder = {v: k for k, v in byte_encoder.items()}
    
    # with open(os.path.join(output_dir, "vocab.json"), 'r', encoding="utf-8") as f:
    #     vocab_from_file = json.load(f)
    # vocab_from_file = {v: bytes([byte_decoder[ch] for ch in k])
    #                    for k, v in vocab_from_file.items()}
    
    # merges_from_file = []
    # with open(os.path.join(output_dir, "merges.txt"), 'r', encoding="utf-8") as f:
    #     merges_from_file = f.read().split("\n")[:-1] # remove the last empty line
    #     merges_from_file = [tuple(merge.split()) for merge in merges_from_file]
    #     merges_from_file = [
    #         (bytes([byte_decoder[ch] for ch in first]), 
    #          bytes([byte_decoder[ch] for ch in second])) 
    #         for first, second in merges_from_file]

            
def save_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str
):
    byte_encoder = bytes_to_unicode()
    
    serializable_vocab = {''.join(byte_encoder[b] for b in v): k for k, v in vocab.items()}
    serializable_merges = [
        (''.join(byte_encoder[b] for b in first),
         ''.join(byte_encoder[b] for b in second)) 
        for first, second in merges
    ]
    
    with open(os.path.join(output_dir, "vocab.json"), 'w', encoding="utf-8") as f:
        json.dump(serializable_vocab, f, indent=2, ensure_ascii=False)
        
    with open(os.path.join(output_dir, "merges.txt"), 'w', encoding="utf-8") as f:
        for first, second in serializable_merges:
            f.write(f"{first} {second}\n")
    

@lru_cache()
def bytes_to_unicode():
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


if __name__ == "__main__":
    main()
