import math
from turtle import forward
from sympy import Line
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum
from jaxtyping import Float

from cs336_basics.basic_block import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
    
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
        self.w1 = Linear(in_features=d_model, out_features=d_ff, **factory_kwargs)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, **factory_kwargs)
        
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return silu(x)