#!/bin/bash

# using whole train set
# uv run train/train_tokenizer.py \
#     --output_dir 'model/my_tokenizer' \
#     --vocab_size 10000 \
#     --data_path 'data/TinyStoriesV2-GPT4-train.txt' \
#     > train_logs/train_tokenizer.log

# using valid set to train, for validate the train process
uv run train/train_tokenizer.py \
    --output_dir 'model/my_tokenizer' \
    --vocab_size 1000 \
    --data_path 'data/TinyStoriesV2-GPT4-valid.txt' \
    > train_logs/train_tokenizer.log

# using the minimum test set
# uv run train/train_tokenizer.py \
#     --output_dir 'model/my_tokenizer' \
#     --vocab_size 500 \
#     --data_path 'tests/fixtures/corpus.en' \
#     > train_logs/train_tokenizer.log