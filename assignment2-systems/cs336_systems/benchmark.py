import torch
import timeit
import cs336_basics
from .bench_config import (
    ModelConfig,
    TrainConfig,
    BenchConfig,
)
    
# Model configurations by size
MODEL_CONFIGS = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}
RANDOM_SEED = 42
DATASET_LEN = 65536

bench_config = BenchConfig(
    model=MODEL_CONFIGS["small"],
    train=TrainConfig(max_iters=1000)
)

def get_batch(
    data: torch.Tensor,
    context_length: int,
    batch_size: int,
    device: torch.device | str,
):
    start_indices = torch.randint(low=0, high=DATASET_LEN - context_length, size=(batch_size,))
    offsets = torch.arange(context_length + 1)
    block_indices = start_indices[:, None] + offsets
    print(f"Block indices shape is: {block_indices.shape}")
    x = data[block_indices][:, :-1].to(device=device)
    y = data[block_indices][:, 1:].to(device=device)
    return x, y


def main():
    print("Benchmarking Start...")
    print("=" * 20)

    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")
    
    # model = cs336_basics.model.BasicsTransformerLM(
    #     vocab_size=model_cfg.vocab_size,
    #     context_length=model_cfg.context_length,
    #     d_model=model_cfg.d_model,
    #     num_layers=model_cfg.num_layers,
    #     num_heads=model_cfg.num_heads,
    #     d_ff=model_cfg.d_ff,
    #     rope_theta=model_cfg.rope_theta,
    # ).to(device)
    # model = torch.compile(model)

    data = torch.randint(low=0, high=10000, size=(DATASET_LEN, ), dtype=torch.long)

    x, y = get_batch(data, bench_config.model.context_length, bench_config.train.batch_size, device)
    print(x)
    print(y)


if __name__ == "__main__":
    main()
