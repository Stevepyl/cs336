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
    max_iters: int = 100  # 5000
    dataset_len: int = 65536
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 200
    resume_from: str | None = None
    out_dir: str = "outputs"  # From training/default.yaml
    save_checkpoint: bool = False

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    max_lr: float = 1e-3
    min_lr: float = 1e-4  # 3e-5
    warmup_iters: int = 5  # 500
    max_l2_norm: float = 1.0  # For gradient clipping
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
@dataclass
class BenchConfig:
    model: ModelConfig
    train: TrainConfig
    optim: OptimizerConfig
    is_only_forward: bool = False
    warmup_steps: int = 5