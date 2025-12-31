import torch
from torch import Tensor
from jaxtyping import Float, Int

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
    m = inputs.max(dim=-1, keepdim=True).values
    #  If your logits (o_i) are very large (e.g., 500), exp(500) will result in inf (infinity), crashing your program.
    log_sum_exp = m + torch.log(torch.sum((inputs - m).exp(), dim=-1, keepdim=True))
    # logits = inputs[range(inputs.shape[0]), targets]
    logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1))
    return torch.mean(-(logits - log_sum_exp))

def get_perplexity(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    return cross_entropy_loss(inputs, targets).exp()
