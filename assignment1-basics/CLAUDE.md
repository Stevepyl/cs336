# CLAUDE.md

## Teaching Protocol

NEVER write implementation code for the user. This is a learning assignment.
- Ask Socratic questions to guide toward the answer
- Show function signatures or type annotations only if needed to unblock
- Point to relevant papers, docs, or existing code patterns in the repo
- When the user is stuck, offer a hint about the concept, not the solution

## Verification

Task is complete when ALL of the following pass:
```bash
uv run pytest                  # all relevant tests green
uv run ruff check .            # zero lint errors
```

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

See `.claude/rules/ml-architecture.md` for full conventions (pre-norm, no-bias, RoPE placement, weight naming, jaxtyping).

## Data

Training data should be placed in `data/`:
- TinyStories: `TinyStoriesV2-GPT4-train.txt`, `TinyStoriesV2-GPT4-valid.txt`
- OpenWebText sample: `owt_train.txt`, `owt_valid.txt`

Download commands are in README.md.
