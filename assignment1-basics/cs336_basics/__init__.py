from .bpe import train_bpe
from .tokenizer import BPETokenizer

import importlib.metadata
__version__ = importlib.metadata.version("cs336_basics")
