import os
import numpy as np
from tqdm import tqdm
from cs336_basics import BPETokenizer

TS_TOKENIZER_PATH = "./model/tinystories/"
OWT_TOKENIZER_PATH = "./model/owt/"
DATA_PATH = "./data/"

def main():
    print(f"Loading tokenizer trained of tinystories from {TS_TOKENIZER_PATH}")
    ts_tokenizer = BPETokenizer.from_files(
        vocab_filepath=os.path.join(TS_TOKENIZER_PATH, "vocab.json"),
        merges_filepath=os.path.join(TS_TOKENIZER_PATH, "merges.txt"),
        special_tokens=["<|endoftext|>"]
    )
    encode_to_bin(ts_tokenizer, "TinyStoriesV2-GPT4-train")
    encode_to_bin(ts_tokenizer, "TinyStoriesV2-GPT4-valid")
    encode_to_bin(ts_tokenizer, "ts_test")
    
    # print(f"Loading tokenizer trained of owt from {OWT_TOKENIZER_PATH}")
    # owt_tokenizer = BPETokenizer.from_files(
    #     vocab_filepath=os.path.join(OWT_TOKENIZER_PATH, "vocab.json"),
    #     merges_filepath=os.path.join(OWT_TOKENIZER_PATH, "merges.txt"),
    #     special_tokens=["<|endoftext|>"]
    # )
    # encode_to_bin(owt_tokenizer, "owt_train")
    # encode_to_bin(owt_tokenizer, "owt_valid")
    # encode_to_bin(owt_tokenizer, "owt_test")

def encode_to_bin(
    tokenizer: BPETokenizer,
    dataset: str,
    chunk_lines: int = 50000
):
    encoded_output = open(os.path.join(DATA_PATH, f"encoded/{dataset}.bin"), 'ab')
    with open(os.path.join(DATA_PATH, f"{dataset}.txt"), 'r', encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
        
    with open(os.path.join(DATA_PATH, f"{dataset}.txt"), 'r', encoding="utf-8") as f:
        lines = []
        for line in tqdm(f, total=total_lines, desc=f"Encoding {dataset}", unit=" lines"):
            lines.append(line)
            if len(lines) >= chunk_lines:
                encodings = tokenizer.encode_iterable(lines)
                for enc in encodings:
                    arr = np.array(enc, dtype=np.uint16)
                    arr.tofile(encoded_output)
                lines = []
        if lines:
            encodings = tokenizer.encode_iterable(lines)
            for enc in encodings:
                arr = np.array(enc, dtype=np.uint16)
                arr.tofile(encoded_output)
    encoded_output.close()
    print(f"Finish encoding {dataset}")
    

if __name__ == "__main__":
    main()