import torch
from .utils import softmax
from .pre_norm_block import get_rope
from .model import TransformerLM, KVCache

def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    # if you want to sample from only the top 90% probability mass (top-p = 0.9), 
    # you'd look for where cumsum_probs first exceeds 0.9 and truncate everything after that point.
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    # Find a smallest set, let its sum >= top_p
    # cumsum_probs - sorted_probs means the all-left-probs sum
    mask = cumsum_probs - sorted_probs > top_p
    # For example, if your cumulative sum is [0.5, 0.8, 0.95, 1.0] and 
    # your sorted probabilities are [0.5, 0.3, 0.15, 0.05], 
    # then cumsum_probs - sorted_probs yields [0.0, 0.5, 0.8, 0.95]. 
    # This represents "all the probability mass accumulated before reaching this token."
    sorted_probs[mask] = 0.0
    # Re-normalize
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    # 根据概率大小来抽。如果某个词的概率是 0.6，那么它有 60% 的几率被抽中；如果概率是 0.1，则有 10% 的几率被抽中。
    next_token = torch.gather(sorted_indices, 1, next_token)
    # 得到该采样词在原始词表（Vocabulary）中真实的 Token ID。
    return next_token

@torch.inference_mode()
def generate(
    model: TransformerLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    use_kv_cache: bool = False,
) -> torch.Tensor:
    """

    Args:
        model (TransformerLM): _description_
        idx (torch.Tensor): (B, T) array of indices in the current context
        max_new_tokens (int): _description_
        block_size (int | None, optional): _description_. Defaults to None.
        temperature (float, optional): _description_. Defaults to 1.0.
        top_p (float, optional): _description_. Defaults to 1.0.
        use_kv_cache (bool, optional): _description_. Defaults to False.

    Returns:
        torch.Tensor: _description_
    """
    for i in range(max_new_tokens):
        if use_kv_cache:
            if i == 0:
                token_positions = torch.arange(idx.size(1), dtype=torch.long, device=idx.device)
                logits= model(idx, token_positions)
            else:
                # Wrong: torch.Tensor (uppercase T) — legacy constructor, no dtype/device kwargs
                token_positions = torch.tensor([idx.size(1) - 1], dtype=torch.long, device=idx.device)
                logits = model(idx[:, -1:], token_positions)
        else:
            idx_cond = idx[:, -block_size:]
            # if not use KVCache, the model is called on all (or last block_size) tokens
            logits = model(idx_cond)
            
        # Even if the model returns logits for every position, 
        # for generation you only need the logits predicting the next token after the current sequence
        logits = logits[:, -1, :]  # becomes (B, C)
        
        if 0.0 == temperature:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # temperature > 1: flatter distribution so more randomness
            # temperature < 1: sharper distribution so less randomness
            # Very small temperature approaches greedy behavior.
            logits = logits / temperature
            probs= softmax(logits, dim=-1)
            idx_next = sample_top_p(probs, top_p)
        
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def install_kv_cache(model: TransformerLM, batch_size: int, total_len: int):
    """
    Install KV cache in all attention layers of the model
    """
    for layer in model.layers:
        layer_dtype = layer.attn.q_proj.weight.dtype
        layer_device = layer.attn.q_proj.weight.device
        layer.attn.cache = KVCache(
            batch_size=batch_size,
            num_heads=layer.attn.num_heads,
            max_seq_len=total_len,
            head_dim=layer.attn.d_k,
            dtype=layer_dtype,
            device=layer_device,
        )
        if layer.attn.rope.max_seq_len < total_len:
            layer.attn.rope = get_rope(
                theta=layer.attn.theta,
                d_k=layer.attn.d_k,
                max_seq_len=total_len,
            ).to(device=layer_device, dtype=layer_dtype)


def remove_kv_cache(model: TransformerLM):
    """Remove KV cache in all attention layers of the model"""
    for layer in model.layers:
        layer_dtype = layer.attn.q_proj.weight.dtype
        layer_device = layer.attn.q_proj.weight.device
        layer.attn.cache = None
        layer.attn.rope = get_rope(
            theta=layer.attn.theta,
            d_k=layer.attn.d_k,
            max_seq_len=model.max_seq_len,
        ).to(device=layer_device, dtype=layer_dtype)
