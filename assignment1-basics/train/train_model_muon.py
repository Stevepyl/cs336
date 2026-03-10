from sympy import use
from torch.optim import Adam
import time
import wandb
import torch
import hydra
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
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
    compute_entropy_chunked,
    scaled_dot_product_attention,
    train_bpe,
    AdamW,
    Muon,
    cosine_learning_rate_schedule,
    get_batch,
    load_checkpoint,
    save_checkpoint,
    Logger,
    install_kv_cache,
    remove_kv_cache,
    generate,
)

SPECIAL_TOKENS = [
    "<|endoftext|>",
]


@torch.no_grad()
def evaluate(model: TransformerLM, data, cfg, device):
    """
    Estimates the loss over a number of batches.
    """
    model.eval()
    losses = []
    entropies = []
    for k in tqdm(range(cfg.training.eval_iters), desc="Evaluating", leave=False):
        x, y = get_batch(data, cfg.training.batch_size, cfg.model.context_length, device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        losses.append(loss.item())
        entropies.append(compute_entropy_chunked(logits).mean().item())
    model.train()
    mean_loss = np.mean(losses)
    return {
        "val/loss": mean_loss,
        "val/perplexity": np.exp(mean_loss),
        "val/entropy": np.mean(entropies),
    }


def setup(cfg: DictConfig):
    if cfg.optimizer.min_lr is None:
        cfg.optimizer.min_lr = cfg.optimizer.max_lr * 0.1
    if cfg.training.eval_interval is None:
        cfg.training.eval_interval = cfg.training.max_iters // 10
    if cfg.training.max_iters is None:
        cfg.training.max_iters = 327_680_000 // cfg.training.batch_size // cfg.model.context_length
        # 32*1024=32,768 tokens per step
        # 327,680,000 / 32,768=10,000 iterations
        # iters = total tokens / (batch_size * context_length)
    if cfg.optimizer.warmup_iters is None:
        cfg.optimizer.warmup_iters = cfg.training.max_iters // 10
    if cfg.optimizer.mm_warmup_steps is None:
        cfg.optimizer.mm_warmup_steps = cfg.optimizer.warmup_iters


@hydra.main(version_base="1.3", config_path="../conf", config_name="train_config")
def main(cfg: DictConfig):
    setup(cfg)

    logger = Logger(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"Configuration:\n {OmegaConf.to_yaml(cfg, resolve=False)}")
    print("=" * 20)
    print(f"Running Start...")
    print("=" * 20)
    print(f"Output directory: {output_dir}")

    torch.manual_seed(cfg.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")
    print("Using float32 matrix multiplications...")

    # ================= Loading data =================
    print(f"Loading data...")
    data_path = Path(cfg.data.path)
    train_data = np.memmap(data_path / "train.bin", dtype=np.uint16, mode="r")
    valid_data = np.memmap(data_path / "valid.bin", dtype=np.uint16, mode="r")
    print(f"Train data size: {len(train_data)}")
    print(f"Valid data size: {len(valid_data)}")

    # ================= Loading model =================
    model = TransformerLM(**cfg.model).to(device)
    model = torch.compile(model) if cfg.training.is_compile else model

    # Collect parameters for different optimizers
    hidden_matrix_params = []  # Muon is used for the main hidden weight matrices in the transformer blocks
    other_params = []  # AdamW is used for everything else (embeddings, layer norms, biases, final head)
    print("Assigning parameters to optimizers:")
    for n, p in model.named_parameters():
        # n is the parameter name, like a path inside the module tree.
        # p is the tensor for that parameter.
        # Check if it's a 2D weight matrix in the transformer layers (attention and FFN weights)
        if p.ndim >= 2 and ("layers" in n) and (".weight" in n) and ("ln" not in n):  # Exclude layer norm weights
            hidden_matrix_params.append(p)
            print(f"    Muon: {n} - {p.shape}")
        else:
            other_params.append(p)
            print(f"    AdamW: {n} - {p.shape}")

    optimizer_adamw = AdamW(
        other_params,
        lr=cfg.optimizer.max_lr,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.eps,
    )
    optimizer_muon = Muon(
        hidden_matrix_params,
        lr=cfg.optimizer.max_lr,
        momentum=0.95,
        weight_decay=cfg.optimizer.weight_decay,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_adamw, optimizer_muon]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    start_iter = 0
    # ================= Loading Checkpoint =================
    if cfg.training.resume_from is not None:
        print(f"Resuming from checkpoint: {cfg.training.resume_from}")
        # load_checkpoint might need adjustment if it only supports one optimizer.
        # Assuming it can handle a list or needs to be called per optimizer.
        # For simplicity, we'll assume it loads the state for the whole model,
        # and we can load optimizer states separately if needed.
        start_iter = load_checkpoint(cfg.training.resume_from, model, optimizers)
        print(f"Resumed from iteration: {start_iter}")

    # ================= Training =================
    print("Start training...")
    start_time = time.time()
    for it in tqdm(range(start_iter, cfg.training.max_iters), desc="Training"):
        lr = cosine_learning_rate_schedule(
            it,
            cfg.optimizer.max_lr,
            cfg.optimizer.min_lr,
            cfg.optimizer.warmup_iters,
            cfg.training.max_iters,
        )
        lr_scale = lr / cfg.optimizer.max_lr
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group["lr"] = param_group["initial_lr"] * lr_scale

        # Momentum warmup
        if cfg.optimizer.mm_warmup is not None:
            frac = min(it / cfg.optimizer.mm_warmup_steps, 1.0)
            for group in optimizer_muon.param_groups:
                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

        x, y = get_batch(train_data, cfg.training.batch_size, cfg.model.context_length, device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = gradient_clipping(
            model.parameters(),  # ty: ignore
            max_l2_norm=cfg.optimizer.max_l2_norm,
        )
        for opt in optimizers:
            opt.step()

        # ================= Logging =================
        if it % cfg.training.log_interval == 0 or it == cfg.training.max_iters - 1:
            duration = time.time() - start_time
            entropy = compute_entropy_chunked(logits).mean()
            lr_adamw = optimizer_adamw.param_groups[0]["lr"]
            lr_muon = optimizer_muon.param_groups[0]["lr"]
            tqdm.write(f"iter {it}: train loss = {loss.item():.4f}, lr = {lr_adamw:.6f}, time = {duration:.2f}s")
            logger.log_metrics(
                {
                    "train/loss": loss.item(),
                    "train/perplexity": loss.exp().item(),
                    "train/lr_adamw": lr_adamw,
                    "train/lr_muon": lr_muon,
                    "train/entropy": entropy.item(),
                    "train/grad_norm": grad_norm,
                },
                step=it,
            )
        # ================= Eval and Checkpointing =================
        if it > 0 and (it % cfg.training.eval_interval == 0 or it == cfg.training.max_iters - 1):
            metrics = evaluate(model, valid_data, cfg, device)
            tqdm.write(f"iter {it}: val loss={metrics['val/loss']:.4f}")
            logger.log_metrics(metrics, it)

            if cfg.training.save_checkpoint:
                checkpoint_path = output_dir / f"checkpoint_{it}.pt"
                tqdm.write(f"saving checkpoint {it} to {checkpoint_path}")
                raw_model = model._orig_mod if cfg.training.is_compile else model
                save_checkpoint(raw_model, optimizers, it, checkpoint_path)
    tqdm.write("Training finished")
    
    tokenizer_path = Path(cfg.data.tokenizer_path)
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=tokenizer_path / "vocab.json",
        merges_filepath=tokenizer_path / "merges.txt",
        special_tokens=SPECIAL_TOKENS,
    )
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("Begining Generation")
    if cfg.training.is_compile:
        # The attribute _orig_mod is a reference to that original, uncompiled model hidden inside the wrapper.
        model = model._orig_mod  # ty: ignore
        install_kv_cache(model, batch_size=1, total_len=cfg.model.context_length + 1000)
    generated_output = tokenizer.decode(
        generate(
            model,
            context,
            max_new_tokens=1000,
            block_size=cfg.model.context_length,
            temperature=0.6,
            top_p=0.95,
            use_kv_cache=True,
        )[0].tolist(),
    )
    tqdm.write("\n--- Generated Text ---")
    tqdm.write(generated_output)
    # Log generated text
    logger.log_text("Generated Text", generated_output, step=cfg.training.max_iters)
    logger.close()
    OmegaConf.save(cfg, output_dir / "config.yaml")
    
if __name__ == "__main__":
    main()