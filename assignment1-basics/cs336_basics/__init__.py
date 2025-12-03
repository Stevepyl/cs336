from .bpe import train_bpe
from .tokenizer import BPETokenizer
from .basic_block import (
    Linear,
    Embedding
)
from .pre_norm_block import (
    RMSNorm,
    SwiGLUFFN,
    SiLUFFN,
    RotaryPositionalEmbedding,
    MultiHeadSelfAttention,
    scaled_dot_product_attention
)
from .utils import (
    silu,
    softmax
)

import importlib.metadata
__version__ = importlib.metadata.version("cs336_basics")
