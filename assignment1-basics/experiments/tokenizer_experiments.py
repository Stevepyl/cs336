from inspect import EndOfBlock
import os
import argparse
import time

from numpy import cross
from cs336_basics import BPETokenizer

def main():
    start_time = time.perf_counter()
    ts_tokenizer = BPETokenizer.from_files(
        vocab_filepath="./model/tinystories/vocab.json",
        merges_filepath="./model/tinystories/merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    owt_tokenizer = BPETokenizer.from_files(
        vocab_filepath="./model/owt/vocab.json",
        merges_filepath="./model/owt/merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    owt_text = ""
    ts_text = ""
    with open("./data/owt_test.txt", 'r', encoding="utf-8") as f:
        owt_text = f.read()
        
    with open("./data/TinyStoriesV2-GPT4-valid.txt", 'r', encoding="utf-8") as f:
        ts_text = f.read()
        
    owt_text_bytes = owt_text.encode("utf-8")
    ts_text_bytes = ts_text.encode("utf-8")
    
    owt_text_ids = owt_tokenizer.encode(owt_text)
    ts_text_ids = ts_tokenizer.encode(ts_text)
    cross_ids = ts_tokenizer.encode(owt_text)
    
    owt_compression_ratio = len(owt_text_bytes) / len(owt_text_ids)
    ts_compression_ratio = len(ts_text_bytes) / len(ts_text_ids)
    cross_compression_ratio = len(owt_text_bytes) / len(cross_ids)
    
    print(f"OWT compression ratio: {owt_compression_ratio:.4f}")
    print(f"TinyStories compression ratio: {ts_compression_ratio:.4f}")
    print(f"Cross compression ratio (OWT text with TinyStories tokenizer): {cross_compression_ratio:.4f}")
    end_time = time.perf_counter()
    print(f"Total experiment time: {end_time - start_time:.8f} seconds")
    
    

if __name__ == "__main__":
    main()
