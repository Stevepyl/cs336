# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
The user is learning about implementing a Transformer-based language model from scratch as part of CS336 (Spring 2025) Assignment 1. So don't give the code directly, but guide him step-by-step.

## Project Overview

This is CS336 (Spring 2025) Assignment 1: Basics of Large Language Models. The project implements:
- A Byte Pair Encoding (BPE) tokenizer from scratch
- A Transformer-based Language Model with modern architecture (RMSNorm, SwiGLU, RoPE)
- Custom AdamW optimizer with cosine learning rate schedule

## Common Commands

```bash
# Environment setup and running
uv sync                          # Install dependencies
uv run <script.py>               # Run any Python file
uv run pytest                    # Run all tests
uv run pytest tests/test_X.py   # Run specific test file
uv run pytest -k "test_name"     # Run tests matching pattern

# Linting and formatting (uses ruff)
uv run ruff check .              # Check for lint errors
uv run ruff format .             # Format code

# Training (Hydra-based, outputs to outputs/)
uv run python train/train_model.py                          # Train with default config
uv run python train/train_model.py model=default training.max_iters=5000  # Override config keys
uv run python train/train_model.py --config-name train_muon_config  # Use Muon optimizer

# Tokenizer training
uv run python train/train_tokenizer.py
uv run python train/encode.py                              # Encode corpus to token IDs
```

## Architecture

### Core Library (`cs336_basics/`)
- `bpe.py` / `tokenizer.py`: BPE tokenizer training and inference
- `basic_block.py`: Foundational layers (`Linear`, `Embedding`, `softmax`, `cross_entropy_loss`)
- `pre_norm_block.py`: Modern Transformer components (`RMSNorm`, `SwiGLUFFN`, `RotaryPositionalEmbedding`, `MultiHeadSelfAttention`)
- `model.py`: High-level components (`TransformerBlock`, `TransformerLM`, `KVCache`)
- `generate.py`: Autoregressive generation with top-p sampling and optional KV cache (`generate()`)
- `optimizer.py`: `AdamW` optimizer and `cosine_learning_rate_schedule`
- `data_loader.py`: `get_batch()` for sampling training batches
- `checkpoint.py`: Model serialization (`save_checkpoint`, `load_checkpoint`)
- `config.py`: Dataclass configs (`ModelConfig`, `TrainingConfig`, `AdamWOptimizerConfig`, `TrainConfig`, etc.)

### Configuration System
Training uses [Hydra](https://hydra.cc/) for config composition. YAML files live in `conf/` with subdirectories for `model/`, `optimizer/`, `training/`, `logger/`, and `data/`. The entry point configs are `conf/train_config.yaml` (AdamW) and `conf/train_muon_config.yaml` (Muon optimizer). Override any field on the CLI with `key=value` syntax.

`ModelConfig` supports ablation flags: `ffn_type` (`swiglu`/`silu`), `use_post_norm`, `remove_rmsnorm`, `remove_rope`.

### Training Scripts (`train/`)
- `train_tokenizer.py`: Train BPE tokenizer on corpus
- `train_model.py`: Train TransformerLM
- `encode.py`: Encode text files to token IDs

### Test Adapter Pattern
The file `tests/adapters.py` bridges student implementations to the test suite. When adding new functionality, update the imports and adapter functions to connect your implementation to the tests.

## Key Implementation Details

- **Pre-norm architecture**: RMSNorm is applied before (not after) attention and FFN sublayers
- **No biases**: Linear layers use `bias=False` throughout
- **Tensor shapes**: Use `jaxtyping` annotations (e.g., `Float[Tensor, "batch seq d_model"]`)
- **RoPE**: Applied to Q and K after projection, before attention computation
- **Weight naming convention**: Match state dict keys in adapters.py (e.g., `q_proj.weight`, `ffn.w1.weight`)

## Data

Training data should be placed in `data/`:
- TinyStories: `TinyStoriesV2-GPT4-train.txt`, `TinyStoriesV2-GPT4-valid.txt`
- OpenWebText sample: `owt_train.txt`, `owt_valid.txt`

Download commands are in README.md.
