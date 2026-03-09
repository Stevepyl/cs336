import torch
from torch import Tensor
from jaxtyping import Float, Int
from collections.abc import Iterable

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(
    x: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    x = x.exp()
    return x / x.sum(dim=dim, keepdim=True)

def cross_entropy_loss(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    inputs = inputs - inputs.max(dim=-1, keepdim=True).values
    #  If your logits (o_i) are very large (e.g., 500), exp(500) will result in inf (infinity), crashing your program.
    log_probs = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True) 
    return -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).mean()

def get_perplexity(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    return cross_entropy_loss(inputs, targets).exp()


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> float:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    if total_norm >= max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(max_l2_norm / (total_norm + eps))
    return total_norm


def norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Functional RMS normalization without learnable parameters"""
    var = x.pow(2).mean(dim=-1, keepdim=True) + eps
    return x * var.rsqrt()


def compute_entropy_chunked(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """
        Memory-efficient implementation of `compute_entropy`.
    """
    
    num_chunks = (logits.shape[1] + chunk_size - 1) // chunk_size
    entropy_chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, logits.shape[1])
        chunk_logits = logits[:, start_idx:end_idx, :]
        # Use the numerically stable method for torch.bfloat16, do not use logsumexp
        chunk_probs = chunk_logits.softmax(dim=-1)
        chunk_log_probs = chunk_logits.log_softmax(dim=-1)
        chunk_entropy = -(chunk_probs * chunk_log_probs).sum(dim=-1)
        entropy_chunks.append(chunk_entropy)
    return torch.cat(entropy_chunks, dim=1)
