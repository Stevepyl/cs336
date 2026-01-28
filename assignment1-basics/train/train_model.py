import wandb
import torch
import hydra
import argparse
import numpy as np
from omegaconf import DictConfig, OmegaConf

from cs336_basics import (
    BPETokenizer,
    Linear,
    Embedding,
    RMSNorm,
    SwiGLUFFN,
    RotaryPositionalEmbedding,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerLM,
    silu,
    softmax,
    cross_entropy_loss,
    gradient_clipping,
    scaled_dot_product_attention,
    train_bpe,
    AdamW,
    cosine_learning_rate_schedule,
    get_batch,
    load_checkpoint,
    save_checkpoint,
)

VOCAB_PATH = "model/owt/vocab.json"
MERGES_PATH = "model/owt/merges.txt"

SPECIAL_TOKENS = [
    "<|endoftext|>",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else DEVICE
DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 1

@hydra.main(version_base="1.3", config_path="../conf", config_name="train_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
if __name__ == "__main__":
    main()
