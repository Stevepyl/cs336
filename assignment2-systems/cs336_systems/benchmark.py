import torch
import timeit
import argparse
import numpy as np
from tqdm import tqdm

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import (
    cross_entropy,
    clip_gradient,
)
from cs336_basics.optimizer import (
    AdamW,
    get_cosine_lr,
)
from cs336_systems.bench_config import (
    ModelConfig,
    TrainConfig,
    BenchConfig,
    OptimizerConfig,
)

# Model configurations by size
MODEL_CONFIGS = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

bench_configs = {
    model_size: BenchConfig(
        model=MODEL_CONFIGS[model_size],
        train=TrainConfig(),
        optim=OptimizerConfig(),
    )
    for model_size in MODEL_CONFIGS
}


def get_batch(
    data: torch.Tensor,
    context_length: int,
    batch_size: int,
    dataset_len: int,
    device: torch.device | str,
):
    start_indices = torch.randint(low=0, high=dataset_len - context_length, size=(batch_size,))
    offsets = torch.arange(context_length + 1)
    block_indices = start_indices[:, None] + offsets
    # print(f"Block indices shape is: {block_indices.shape}")
    x = data[block_indices][:, :-1].to(device=device)
    y = data[block_indices][:, 1:].to(device=device)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_only_forward",
        action="store_true",
        help="Only do the forward passes",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Warmup iters which is not included in time measuring",
    )
    parser.add_argument("--measurement_steps", type=int, default=10)
    args = parser.parse_args()

    for _, cfg in bench_configs.items():
        cfg.is_only_forward = args.is_only_forward
        cfg.warmup_steps = args.warmup_steps
        cfg.train.max_iters = args.measurement_steps + cfg.warmup_steps

    print("Benchmarking Start...")
    print("=" * 20)
    for model_size, cfg in bench_configs.items():
        torch.manual_seed(cfg.train.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {device}")

        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("high")

        model = BasicsTransformerLM(
            vocab_size=cfg.model.vocab_size,
            context_length=cfg.model.context_length,
            d_model=cfg.model.d_model,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            d_ff=cfg.model.d_ff,
            rope_theta=cfg.model.rope_theta,
        ).to(device)
        # model = torch.compile(model)
        optimizer = AdamW(
            params=model.parameters(),
        )
        train_data = torch.randint(low=0, high=10000, size=(cfg.train.dataset_len,), dtype=torch.long)
        start_iter = 0

        train_time: list[float] = []
        forward_time: list[float] = []
        backpropagation_time: list[float] = []

        # ================= Training =================
        print("Training Start...")
        print("=" * 20)
        train_start = 0.0
        for it in tqdm(range(start_iter, cfg.train.max_iters), desc="training"):
            if it >= cfg.warmup_steps:
                train_start = timeit.default_timer()

            lr = get_cosine_lr(
                it,
                cfg.optim.max_lr,
                cfg.optim.min_lr,
                cfg.optim.warmup_iters,
                cfg.train.max_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            x, y = get_batch(train_data, cfg.model.context_length, cfg.train.batch_size, cfg.train.dataset_len, device)

            # Forward Pass
            forward_start = timeit.default_timer()
            logits = model(x)
            loss = cross_entropy(logits, y)
            torch.cuda.synchronize()
            forward_end = timeit.default_timer()

            if not cfg.is_only_forward:
                # Backpropagation
                backpropagation_start = timeit.default_timer()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.cuda.synchronize()
                backpropagation_end = timeit.default_timer()
                if it >= cfg.warmup_steps:
                    backpropagation_time.append(backpropagation_end - backpropagation_start)

                # Gradient Clipping
                grad_norm = clip_gradient(model.parameters(), max_norm=1.0)  # noqa: F841
                optimizer.step()

            if it >= cfg.warmup_steps:
                train_end = timeit.default_timer()
                forward_time.append(forward_end - forward_start)
                train_time.append(train_end - train_start)

        tqdm.write(f"Benchmarking with size {model_size} finished...")
        print(f"    Train uses per step:                 {np.mean(train_time):.3f} ± {np.std(train_time):.3f}s")
        print(f"    Forward pass per step:               {np.mean(forward_time):.3f} ± {np.std(forward_time):.3f}s")
        if not cfg.is_only_forward:
            print(
                f"    Backpropagation per step:            {np.mean(backpropagation_time):.3f} ± {np.std(backpropagation_time):.3f}s"
            )


if __name__ == "__main__":
    main()
