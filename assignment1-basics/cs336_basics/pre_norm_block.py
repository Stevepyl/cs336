import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum, rearrange
from jaxtyping import Float, Bool, Int
from functools import lru_cache

from cs336_basics.basic_block import Linear
from cs336_basics.utils import (
    silu,
    softmax
)


def scaled_dot_product_attention(
    queries: Float[Tensor, "batch_size ... seq_len d_k"],
    key: Float[Tensor, "batch_size ... seq_len d_k"],
    values: Float[Tensor, "batch_size ... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None
) -> Float[Tensor, "batch_size ... d_v"]:
    d_k = queries.shape[-1]
    score = torch.matmul(queries, key.transpose(-2, -1)) / math.sqrt(d_k)
    # score = einsum(queries, key, "... i d_k, ... j d_k -> ... i j") / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == False, float('-inf'))
    score = softmax(score, dim=-1)
    attention: Float[Tensor, "batch_size, ..., d_v"] = torch.matmul(score, values)
    return attention


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weights = nn.Parameter(torch.empty(d_model, **factory_kwargs))

    def forward(self, x: Float[Tensor, " ... d_model"]) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.

        Args:
            x (Float[Tensor, "... d_model"]): input tensor of shape (batch_size, sequence_length, d_model) 

        Returns:
            torch.Tensor: _description_
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = x / self._rms(x) * self.weights
        return result.to(in_dtype)

    def _rms(self, a: torch.Tensor):
        # a is d_model dimension
        return torch.sqrt(self.eps + a.pow(2).mean(dim=-1, keepdim=True))


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_ff: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        # w1_weight(Float[Tensor, "d_ff d_model"]): Stored weights for W1
        # w2_weight(Float[Tensor, "d_model d_ff"]): Stored weights for W2
        # w3_weight(Float[Tensor, "d_ff d_model"]): Stored weights for W3
        self.w1 = Linear(in_features=d_model,
                         out_features=d_ff, **factory_kwargs)
        self.w2 = Linear(
            in_features=d_ff, out_features=d_model, **factory_kwargs)
        self.w3 = Linear(in_features=d_model,
                         out_features=d_ff, **factory_kwargs)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFFN(nn.Module):
    def __init__(
        self,
        d_ff: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = Linear(in_features=d_model,
                         out_features=d_ff, **factory_kwargs)
        self.w2 = Linear(in_features=d_ff,
                         out_features=d_model, **factory_kwargs)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return silu(x)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        angles = self._compute_rotate_angles()
        self.register_buffer(
            "angles", self._compute_rotate_angles(), persistent=False)

    def forward(
        self,
        x: Float[Tensor, " batch_size sequence_length d_k"],
        token_positions: Float[Tensor, " ... sequence_length"]
    ) -> torch.Tensor:
        x = x.view(*x.shape[:-1], -1, 2)
        x = torch.view_as_complex(x)
        self.angles.to(dtype=x.dtype)
        if self.angles[token_positions].ndim == 3: #type: ignore
            x_rotated = x * self.angles[token_positions].unsqueeze(1) # type: ignore
        else:
            x_rotated = x * self.angles[token_positions]  # type: ignore
        x_real = torch.view_as_real(x_rotated).flatten(-2)
        return x_real

    def _compute_rotate_angles(self):
        # holds the speed at which each dimension pair rotates.
        # shape: (self.d_k / 2, )
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, dtype=torch.float32).float() / self.d_k))
        positions = torch.arange(end=self.max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, freqs)
        # Creates a complex tensor where real part is cos and imag part is sin
        # Equal to torch.complex(torch.cos(freqs), torch.sin(freqs))
        # cis means 'c'os + 'is'in
        angles_cis = torch.polar(torch.ones_like(angles), angles)
        return angles_cis

# Not method of RoPE
@lru_cache(10)
def get_rope(theta: float, d_k: int, max_seq_len: int):
    return RotaryPositionalEmbedding(theta, d_k, max_seq_len)

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        # Following Attention is All You Need, set d_k = d_v = d_model/h
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        # o_proj is actually d_model in and h * d_v out
        self.o_proj = Linear(d_model, d_model, **factory_kwargs)
        if (theta is not None) and (max_seq_len is not None):
            self.rope = get_rope(theta, self.d_k, max_seq_len)

    def forward(
        self, 
        x: Float[Tensor, "batch_size seq_len d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> torch.Tensor: # Same shape of x
        seq_len = x.shape[-2]
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=x.dtype).tril()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # For every Batch and for every Head, independently calculate the interaction between Sequence tokens.
        # (batch_size, seq_len, d_model=num_heads*d_k) -> (batch_size, num_heads, seq_len, d_k)
        # q = q.view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        q = rearrange(q, "b s (h d) ->  b h s d", h=self.num_heads).contiguous()
        k = rearrange(k, "b s (h d) ->  b h s d", h=self.num_heads).contiguous()
        v = rearrange(v, "b s (h d) ->  b h s d", h=self.num_heads).contiguous()
        
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        multi_head = scaled_dot_product_attention(q, k, v, mask)
        # read multi_head = concat(head_1, ... , head_h)
        multi_head = rearrange(multi_head, "b h s d -> b s (h d)")
        return self.o_proj(multi_head)
    