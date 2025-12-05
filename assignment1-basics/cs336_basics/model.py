import torch
import torch.nn as nn

from .pre_norm_block import RMSNorm, SwiGLUFFN, MultiHeadSelfAttention
from .basic_block import Embedding, Linear
from .utils import softmax

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length # max_seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        factory_kwargs = {"device": device, "dtype": dtype}

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, **factory_kwargs)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, rope_theta, **factory_kwargs
            ) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, **factory_kwargs)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        assert seq_len <= self.context_length, "Sequence length exceeds model capacity"
        x = self.token_embeddings(x)
        
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        x = self.lm_head(x)
        # return softmax(x, self.vocab_size)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.eps = eps
        self.theta = theta
        self.max_seq_len = max_seq_len
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ln1 = RMSNorm(d_model=self.d_model,
                           eps=self.eps, **factory_kwargs)

        self.attn = MultiHeadSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            theta=self.theta,
            max_seq_len=self.max_seq_len,
            **factory_kwargs
        )
        self.ffn = SwiGLUFFN(
            d_ff=self.d_ff,
            d_model=self.d_model,
            **factory_kwargs
        )
        self.ln2 = RMSNorm(d_model=self.d_model,
                           eps=self.eps, **factory_kwargs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
