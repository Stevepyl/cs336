from .bpe import train_bpe
from .tokenizer import BPETokenizer
from .basic_block import (
    Linear,
    Embedding,
)
from .pre_norm_block import (
    RMSNorm,
    SwiGLUFFN,
    SiLUFFN,
    RotaryPositionalEmbedding,
    MultiHeadSelfAttention,
    scaled_dot_product_attention,
)
from .model import (
    TransformerBlock,
    TransformerLM,
)
from .utils import (
    silu,
    softmax,
    cross_entropy_loss,
    gradient_clipping,
)

from .optimizer import (
    AdamW,
    cosine_learning_rate_schedule,
)

import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")
