# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS336 Assignment 2 - Systems optimization for transformer language models. Implements benchmarking infrastructure, FlashAttention-2 (PyTorch + Triton), Distributed Data Parallel training (individual + bucketed), and optimizer state sharding.

## Teaching Protocol

NEVER write implementation code for the user. This is a learning assignment.
- Ask Socratic questions to guide toward the answer
- Show function signatures or type annotations only if needed to unblock
- Point to relevant papers, docs, or existing code patterns in the repo
- When the user is stuck, offer a hint about the concept, not the solution


## Setup

Uses `uv` as the package manager. Dependencies install automatically via `uv run`.

```bash
uv run python  # interactive Python with all deps
```

## Command Execution

All commands use `uv run` for dependency isolation(except for `ruff`):

```bash
uv run pytest        # run tests
uv run python        # interactive shell
ruff                 # linting
uv run nsys profile  # profiling with Nsight Systems
```

## Commands

```bash
# Run all tests
uv run pytest -v ./tests

# Run specific test files
uv run pytest tests/test_attention.py -v
uv run pytest tests/test_ddp_individual_parameters.py -v
uv run pytest tests/test_ddp.py -v
uv run pytest tests/test_sharded_optimizer.py -v

# Run a single test by name
uv run pytest -k test_flash_forward_pass_pytorch -v
uv run pytest -k test_flash_forward_pass_triton -v
uv run pytest -k test_flash_backward -v

# Lint
ruff check --line-length 120 cs336_systems/

# Build submission zip (runs all tests first)
./test_and_make_submission.sh

# Profile with Nsight Systems
uv run nsys profile -o result python benchmark.py
```

## Architecture

### Module Layout

- `cs336_systems/` - All implementations go here (currently empty)
- `tests/adapters.py` - **Interface contract**: all functions to implement; each raises `NotImplementedError`
- `tests/` - Test suite; tests are the specification
- `cs336-basics/` - Reference Assignment 1 implementation (Transformer model, AdamW, etc.)

### What to Implement (in `cs336_systems/`)

All implementations are registered via `tests/adapters.py`.

**Part 1 - Benchmarking (no tests, script deliverables):**
- `benchmarking_script`: Time forward/backward with `torch.cuda.synchronize()`, warmup steps, `timeit`
- `nsys_profile`: Profile with `nsys profile -o result python benchmark.py`, use NVTX ranges
- `mixed_precision_accumulation`: Use `torch.autocast(device="cuda", dtype=torch.bfloat16)`
- `memory_profiling`: Use `torch.cuda.memory._record_memory_history()` + `_dump_snapshot()`

**Part 2 - FlashAttention-2:**

1. `get_flashattention_autograd_function_pytorch()` → returns `torch.autograd.Function` subclass
   - Problem: `flash_forward` (a): pure PyTorch tiled FA2 forward (tile size ≥ 16×16)
   - Forward signature: `def forward(ctx, Q, K, V, is_causal=False)`
   - Saves Q, K, V, O, L for backward; backward raises `NotImplementedError` initially
   - Test: `pytest -k test_flash_forward_pass_pytorch`

2. `get_flashattention_autograd_function_triton()` → returns `torch.autograd.Function` subclass
   - Problem: `flash_forward` (b): Triton kernel `flash_fwd_kernel` following Algorithm 1
   - Launch grid: `(Tq, batch_size)` — one instance per query tile per batch element
   - Single loop over key tiles; advance block ptrs at end of loop
   - Problem: `flash_forward` (c): add `is_causal: tl.constexpr` flag, mask with `-1e6`
   - Problem: `flash_backward`: backward using `torch.compile` on PyTorch (Equations 13-19), compute D = rowsum(O ◦ dO)
   - Test: `pytest -k test_flash_forward_pass_triton`, `pytest -k test_flash_backward`

**Triton kernel signature to use:**
```python
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
):
```

**FlashAttention-2 Algorithm 1 (forward):** Online softmax with running `m` (max) and `l` (normalizer). For each query tile `i`, loop over key tiles `j`: compute S, update m, compute P̃ = exp(S - m), update l and O. Finalize: O /= l, L = m + log(l). On-chip buffers (O, l, m) in `tl.float32`. Cast P̃ to V dtype before matmul.

**Part 3 - DDP:**

3. `get_ddp_individual_parameters(module)` → DDP wrapper nn.Module
   - Broadcasts params from rank 0 at init
   - Uses `register_post_accumulate_grad_hook` to async all-reduce each param gradient as it's ready
   - `finish_gradient_synchronization()`: call `handle.wait()` on all pending handles
   - Test: `pytest tests/test_ddp_individual_parameters.py` (run multiple times)

4. `ddp_individual_parameters_on_after_backward(ddp_model, optimizer)` — call before `optimizer.step()`

5. `get_ddp_bucketed(module, bucket_size_mb)` → bucketed DDP wrapper
   - Allocates params to buckets in reverse `model.parameters()` order
   - All-reduces a bucket when all its params' gradients are ready
   - Test: `pytest tests/test_ddp.py` (run multiple times)

6. `ddp_bucketed_on_after_backward(ddp_model, optimizer)` and `ddp_bucketed_on_train_batch_start(ddp_model, optimizer)`

**Part 4 - Sharded Optimizer:**

7. `get_sharded_optimizer(params, optimizer_cls, **kwargs)` → sharded optimizer
   - Each rank handles ~1/world_size of params
   - After `step()`, each rank broadcasts its updated params to all others
   - Must call `torch.optim.Optimizer.__init__` in constructor
   - Must implement `add_param_group` for dynamic param group additions
   - Test: `pytest tests/test_sharded_optimizer.py` (run multiple times)

### Model Sizes (for benchmarking)

| Size  | d_model | d_ff  | num_layers | num_heads |
|-------|---------|-------|------------|-----------|
| small | 768     | 3072  | 12         | 12        |
| medium| 1024    | 4096  | 24         | 16        |
| large | 1280    | 5120  | 36         | 20        |
| xl    | 1600    | 6400  | 48         | 25        |
| 2.7B  | 2560    | 10240 | 32         | 32        |

Vocab size: 10,000. Batch size: 4. Context lengths: 128, 256, 512, 1024.

### Test Infrastructure

- `tests/conftest.py` - Snapshot fixtures (`NumpySnapshot`, `Snapshot`) for numerical correctness
- `tests/common.py` - DDP helpers: `ToyModel`, `ToyModelWithTiedWeights`, process group setup/teardown
- `tests/fixtures/` - Precomputed tensors (`ddp_test_data.pt`, `ddp_test_labels.pt`)

### cs336-basics Reference

Key files in `cs336-basics/cs336_basics/`:
- `model.py` - `Linear`, `Embedding`, `RMSNorm`, `Transformer`, `TransformerLM`, `scaled_dot_product_attention`
- `optimizer.py` - AdamW implementation
- `nn_utils.py`, `data.py` - Utilities

## Key Notes

- Python `>=3.11,<3.13`; PyTorch 2.6.0 (Linux/ARM Mac) or 2.2.2 (Intel Mac)
- Use NCCL backend for GPU distributed training, Gloo for CPU/local dev
- All distributed tests: run 5+ times to verify reliability (non-deterministic)
- Pytest: `-s` flag (stdout captured), `log_cli=true` at WARNING level
- Ruff line length: 120
- Submission: `writeup.pdf` + `code.zip` via `./test_and_make_submission.sh`
