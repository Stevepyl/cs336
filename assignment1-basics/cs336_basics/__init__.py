from .bpe import train_bpe
from .tokenizer import BPETokenizer
from .model import (
    Linear,
    Embedding,
    RMSNorm
)

import importlib.metadata
__version__ = importlib.metadata.version("cs336_basics")
