#!/bin/bash

# 如果运行脚本时提供了第 1 个参数，就把它赋给 train_mode；否则使用默认值 "full"。
train_mode="${1:-full}"
comments="${2:-}"

case "$train_mode" in 
    test)
        data_path='tests/fixtures/corpus.en'
        vocab_size=500
        ;;
    mini)
        data_path='data/TinyStoriesV2-GPT4-valid.txt'
        vocab_size=1000
        ;;
    full)
        data_path='data/TinyStoriesV2-GPT4-train.txt'
        vocab_size=10000
        ;;
    *)
        echo "Unknown train_mode: $train_mode" >&2
        exit 1
        ;;
esac    

uv run train/train_tokenizer.py \
  --output_dir 'model/my_tokenizer' \
  --vocab_size "$vocab_size" \
  --data_path "$data_path" \
  > train_logs/train_tokenizer_${train_mode}_${comments}.log