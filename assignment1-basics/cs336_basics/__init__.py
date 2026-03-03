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
    KVCache,
)
from .utils import (
    silu,
    softmax,
    cross_entropy_loss,
    gradient_clipping,
    compute_entropy_chunked,
)

from .optimizer import (
    AdamW,
    cosine_learning_rate_schedule,
)

from .data_loader import (
    get_batch,
)

from .checkpoint import (
    load_checkpoint,
    save_checkpoint,
)

from .generate import (
    install_kv_cache,
    remove_kv_cache,
    generate,
)

from .logger import Logger
from .config import TrainConfig
import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")
