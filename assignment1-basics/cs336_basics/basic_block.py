import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum
from jaxtyping import Int, Float

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_in = in_features
        self.d_out = out_features
        self.weights = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(self.weights, x,  " d_out d_in, ... d_in -> ... d_out")

    def _custom_init_weights(self):
        '''
        Init weights as follows:
            Linear weights: N(u = 0, sigma^2 = 2 / (d_in + d_out)),trcuncated at (-3*sigma, 3*sigma)
        '''
        variance = 2.0 / (self.d_in + self.d_out)
        std_deviation = math.sqrt(variance)

        a = -3.0 * std_deviation
        b = 3.0 * std_deviation

        nn.init.trunc_normal_(self.weights, mean=0,
                              std=std_deviation, a=a, b=b)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weights = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        self._custom_init_weights()

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        '''
            If you passed a batch, e.g. token_ids.shape == (batch_size, seq_len), 
            then the output would be (batch_size, seq_len, embedding_dim).
        '''
        return self.weights[token_ids]

    def _custom_init_weights(self):
        nn.init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)
