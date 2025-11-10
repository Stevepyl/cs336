import json
import time
from pathlib import Path
from cs336_basics import train_bpe

def main():
    train_on = "tinystories"
    file = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 2000
    output_dir = Path(f"tokenizer/{train_on}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir/"tokenizer.json"
    vocab_path = output_dir/"vocab.json"
    merges_path = output_dir/"merges.txt"
    
    # Start Training
    start_time = time.time()
    vocab, merges = train_bpe(file, vocab_size, special_tokens, 16)
    end_time = time.time()
    print(f"Training time: {(end_time - start_time):.2f} seconds")
    
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {v.decode("utf-8", errors="ignore"): k for k, v in vocab.items()}, 
            f, 
            ensure_ascii=False, 
            indent=4)
    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(
                f"[{merge[0].decode("utf-8", errors="ignore")}, {merge[1].decode("utf-8", errors="ignore")}]\n")
    print("Results saved.")
    
    
if __name__ == "__main__":
    main()