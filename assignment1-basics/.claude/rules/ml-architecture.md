# ML Architecture Conventions

Conventions for this CS336 Transformer implementation. Apply when reading or reasoning about cs336_basics/ code.

## Architecture Decisions

- **Pre-norm**: RMSNorm is applied BEFORE (not after) each attention and FFN sublayer
- **No biases**: ALL Linear layers use `bias=False`
- **Activations**: Default FFN uses SwiGLU; controlled by `ModelConfig.ffn_type` (`swiglu`/`silu`)
- **Normalization**: RMSNorm throughout; no LayerNorm

## Positional Encoding

- RoPE (Rotary Positional Embedding) is applied to Q and K AFTER projection, BEFORE attention computation
- Ablation flags: `remove_rope=True` disables RoPE; `remove_rmsnorm=True` disables all norms

## Weight Naming Convention

State dict keys must match `tests/adapters.py` exactly:
- Attention projections: `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `output_proj.weight`
- FFN (SwiGLU): `ffn.w1.weight`, `ffn.w2.weight`, `ffn.w3.weight`
- Norms: `ln1.weight`, `ln2.weight`

## Tensor Shape Conventions

Use `jaxtyping` annotations throughout:
```python
from jaxtyping import Float
from torch import Tensor

def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
```

## Test Adapter Pattern

`tests/adapters.py` bridges implementations to the test suite. When adding new components:
1. Implement in `cs336_basics/`
2. Add import and adapter function in `tests/adapters.py`
3. Run `uv run pytest tests/test_<component>.py` to verify
