import torch
import numpy as np
import numpy.typing as npt

from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    start_indices = np.random.randint(low=0, high=len(dataset) - context_length, size=(batch_size,))
    
    # Create sequences of context_length + 1 tokens
    # (we need +1 to create input-target pairs)
    offsets = np.arange(context_length + 1)
    
    # Imagine start_indices = [10, 50] and offsets = [0, 1, 2].
    # The column "stretches" horizontally to match the length of offsets:
    # [10, 50] -> 
    #   [[10, 10, 10],
    #   [50, 50, 50]]
    # [0, 1, 2] ->
    #   [[0, 1, 2],
    #   [0, 1, 2]]
    # Element-wise addition:
    #   [[10 + 0, 10 + 1, 10 + 2],
    #   [50 + 0, 50 + 1, 50 + 2]]
    block_indices = start_indices[:, None] + offsets
    data_blocks = torch.from_numpy(dataset[block_indices].astype(np.int64))

    # Split into inputs (all but last token) and targets (all but first token)
    x = data_blocks[:, :-1]
    y = data_blocks[:, 1:]
    
    x, y = x.to(device), y.to(device)
    return x, y
    