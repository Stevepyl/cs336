from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    rope_theta: float = 10000
    vocab_size: int = 10000
    context_length: int = 256


@dataclass
class TrainConfig:
    """Training loop configuration."""
    seed: int = 42
    is_compile: bool = False  # torch.compile the model or not
    batch_size: int = 4
    max_iters: int | None = None  # 5000
    dataset_len: int = 65536
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 200
    resume_from: str | None = None
    out_dir: str = "outputs"  # From training/default.yaml
    save_checkpoint: bool = False

@dataclass
class BenchConfig:
    model: ModelConfig
    train: TrainConfig