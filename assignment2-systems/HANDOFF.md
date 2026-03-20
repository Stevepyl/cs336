# HANDOFF.md

Project progress tracker for CS336 Assignment 2 (Systems).

## Current Status

**No implementations started.** `cs336_systems/` contains only the empty `__init__.py`.

---

## Implementation Checklist

### Part 1 - Benchmarking & Profiling (script deliverables, no automated tests)

- [ ] `benchmarking_script` — timing script for forward/backward passes
  - Supports model size CLI args, w warmup steps, n timing steps
  - Calls `torch.cuda.synchronize()` after each step
  - Uses `timeit.default_timer()` for timing
  - Deliverable: timings table for all 5 model sizes + writeup responses

- [ ] `nsys_profile` — Nsight Systems profiling
  - Use `nsys profile -o result python benchmark.py`
  - Annotate with NVTX ranges to isolate warmup, forward, backward, per-attention-op
  - Deliverable: writeup responses (what kernel takes most time, softmax vs matmul, etc.)

- [ ] `mixed_precision_accumulation` — accumulation precision experiment
  - Run the 4 code snippets in the PDF (§1.1.5)
  - Add `torch.autocast(device="cuda", dtype=torch.bfloat16)` option to benchmark script
  - Deliverable: timing tables + writeup responses

- [ ] `memory_profiling` — PyTorch memory profiler
  - Use `torch.cuda.memory._record_memory_history()` + `_dump_snapshot()`
  - Profile 2.7B model at context lengths 128, 256, 512
  - Deliverable: memory timeline images + peak memory table

### Part 2 - FlashAttention-2

- [ ] `pytorch_attention` — benchmark vanilla PyTorch attention (script only)
  - Sweep: batch=8, d_head ∈ [16,32,64,128], seq_len ∈ [256,1024,4096,8192,16384]
  - Report OOM errors, timing table

- [ ] `torch_compile` — benchmark torch.compile on attention and full model (script only)

- [ ] **`flash_forward` (a)** — PyTorch FA2 forward pass  → `get_flashattention_autograd_function_pytorch()`
  - `torch.autograd.Function` subclass
  - Tile size ≥ 16×16; inputs guaranteed power-of-2 ≥ 16
  - Saves Q, K, V, O, L; backward raises `NotImplementedError`
  - Test: `uv run pytest -k test_flash_forward_pass_pytorch`

- [ ] **`flash_forward` (b)** — Triton FA2 forward kernel → `get_flashattention_autograd_function_triton()`
  - Kernel: `flash_fwd_kernel` with block ptrs, launch grid `(Tq, batch_size)`
  - On-chip buffers in `tl.float32`; cast P̃ to V dtype before matmul
  - Test: `uv run pytest -k test_flash_forward_pass_triton`

- [ ] **`flash_forward` (c)** — Causal masking flag
  - Add `is_causal: tl.constexpr` to kernel; mask with `-1e6`
  - `ctx.is_causal = is_causal` saved for backward

- [ ] **`flash_backward`** — FA2 backward pass (torch.compile, not Triton)
  - Compute D = rowsum(O ◦ dO); then Equations 13-19 from the PDF
  - Returns dQ, dK, dV
  - Test: `uv run pytest -k test_flash_backward`

- [ ] `flash_benchmarking` — compare FA2 vs vanilla attention (script deliverable)

### Part 3 - Distributed Data Parallel

- [ ] `naive_ddp` — naive DDP script (not tested by test suite)

- [ ] `naive_ddp_benchmarking` — benchmark naive DDP (script deliverable)

- [ ] `minimal_ddp_flat_benchmarking` — batched all-reduce (script deliverable)

- [ ] **`ddp_overlap_individual_parameters`** → `get_ddp_individual_parameters()`
  - Broadcast params from rank 0 at init
  - `register_post_accumulate_grad_hook` for async all-reduce per param
  - `finish_gradient_synchronization()` waits on all handles
  - Also implement `ddp_individual_parameters_on_after_backward()`
  - Test: `uv run pytest tests/test_ddp_individual_parameters.py` (run 5+ times)

- [ ] `ddp_overlap_individual_parameters_benchmarking` (script deliverable)

- [ ] **`ddp_overlap_bucketed`** → `get_ddp_bucketed()`
  - Buckets in reverse `model.parameters()` order
  - All-reduce bucket when all its grads ready
  - Also implement `ddp_bucketed_on_after_backward()` and `ddp_bucketed_on_train_batch_start()`
  - Test: `uv run pytest tests/test_ddp.py` (run 5+ times)

- [ ] `ddp_bucketed_benchmarking` (script deliverable)

- [ ] `communication_accounting` — written math problem (no code needed)

### Part 4 - Optimizer State Sharding

- [ ] **`optimizer_state_sharding`** → `get_sharded_optimizer()`
  - Subclass `torch.optim.Optimizer`; call super().__init__ in constructor
  - Shard params ~1/world_size per rank
  - After `step()`, broadcast each rank's updated params to all others
  - Implement `add_param_group()` for dynamic param groups
  - Test: `uv run pytest tests/test_sharded_optimizer.py` (run 5+ times)

- [ ] `optimizer_state_sharding_accounting` (script + writeup deliverable)

---

## Implementation Notes

### FlashAttention-2 Algorithm Summary

**Forward (Algorithm 1):**
- Outer loop: query tiles i (parallelized across Triton programs)
- Inner loop: key tiles j
- Running state per query tile: m (max), l (normalizer), O (output accumulator)
- Update rule:
  ```
  m_new = max(m_old, rowmax(S_ij))
  P_tilde = exp(S_ij - m_new)
  l_new = exp(m_old - m_new) * l_old + rowsum(P_tilde)
  O_new = diag(exp(m_old - m_new)) * O_old + P_tilde @ V_j
  ```
- Finalize: O /= l; L = m + log(l)

**Backward (Equations 13-19, PyTorch + torch.compile):**
- Pre-compute D = rowsum(O ◦ dO)
- Recompute S = QK^T / sqrt(d), P = exp(S - L)
- dV = P^T @ dO
- dP = dO @ V^T
- dS = P ◦ (dP - D)
- dQ = dS @ K / sqrt(d)
- dK = dS^T @ Q / sqrt(d)

### DDP Pattern

```python
# Register hook per param at init:
param.register_post_accumulate_grad_hook(
    lambda p: handles.append(dist.all_reduce(p.grad, async_op=True))
)

# Before optimizer.step():
for handle in handles:
    handle.wait()
handles.clear()
```

### Sharded Optimizer Pattern

```python
# Assign params: rank r gets params[r::world_size]
# After local optimizer.step():
for i, param in enumerate(all_params):
    owner_rank = i % world_size
    dist.broadcast(param.data, src=owner_rank)
```

---

## Benchmarking Configuration (Standard)

- Single node, 2 GPUs
- XL model (d_model=1600, d_ff=6400, 48 layers, 25 heads)
- Vocab size 10,000, batch size 4
- NCCL backend for GPU, Gloo for local dev
- 5 warmup steps, 10 measurement steps

---

## Submission

Files needed:
- `writeup.pdf` (from `writeup.typ`)
- `code.zip` (run `./test_and_make_submission.sh`)

Leaderboard: `github.com/stanford-cs336/assignment2-systems-leaderboard`
- Test config: batch=1, seq_len=16384, d_model=1024, n_heads=16, d_head=64, BF16, causal
