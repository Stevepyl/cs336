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
    entropies= []
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

@hydra.main(version_base="1.3", config_path="../conf", config_name="train_config")
def main(cfg: DictConfig):
    setup(cfg)
    logger = Logger(cfg)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"Configuration:\n {OmegaConf.to_yaml(cfg, resolve=False)}")
    print("="*20)
    print(f"Running Start...")
    print("="*20)
    print(f"Output directory: {output_dir}")    
    
    torch.manual_seed(cfg.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    print(f"Using device: {device}")

    # ================= Loading data =================
    print(f"Loading data...")
    data_path = Path(cfg.data.path)
    train_data = np.memmap(data_path / 'train.bin', dtype=np.uint16, mode='r')
    valid_data = np.memmap(data_path / "valid.bin", dtype=np.uint16, mode='r')
    print(f"Train data size: {len(train_data)}")
    print(f"Valid data size: {len(valid_data)}")
    
    # ================= Loading model =================
    model = TransformerLM(**cfg.model).to(device)
    model = torch.compile(model) if cfg.training.is_compile else model
    
    optimizer = AdamW(
        model.parameters(),  # ty:ignore[possibly-missing-attribute]
        lr=cfg.optimizer.max_lr,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.eps,
    )
    
    start_iter = 0
    if cfg.training.resume_from is not None:
        print(f"Resuming from checkpoint: {cfg.training.resume_from}")
        start_iter = load_checkpoint(cfg.training.resume_from, model, optimizer)
        print(f"Resumed from iteration: {start_iter}")

    # ================= Training =================
    print("Start training...")
    start_time = time.time()
    for it in tqdm(range(start_iter, cfg.training.max_iters), desc="Training"):
        # Learning rate schedule
        lr = cosine_learning_rate_schedule(
            it,
            cfg.optimizer.max_lr,
            cfg.optimizer.min_lr,
            cfg.optimizer.warmup_iters,
            cfg.training.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # Get a batch of data
        x, y = get_batch(train_data, cfg.training.batch_size, cfg.model.context_length, device)
        
        # Forward pass
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient Clipping
        grad_norm = gradient_clipping(model.parameters(), max_l2_norm=1.0)
        
        optimizer.step()

        # ================= Logging =================
        if it % cfg.training.log_interval == 0 or it == cfg.training.max_iters - 1:
            duration = time.time() - start_time
            entropy = compute_entropy_chunked(logits).mean()
            tqdm.write(f"iter {it}: train loss = {loss.item():.4f}, lr = {lr:.6f}, time = {duration:.2f}s")
            logger.log_metrics({
                "train/loss": loss.item(),
                "train/perplexity": loss.exp().item(),
                "train/lr": lr,
                "train/entropy": entropy.item(),
                'train/grad_norm': grad_norm,
            }, step=it)

        # ================= Eval and Checkpointing =================
        if it > 0 and (it % cfg.training.eval_interval == 0 or it == cfg.training.max_iters - 1):
            metrics = evaluate(model, valid_data, cfg, device)
            tqdm.write(f"iter {it}: val loss={metrics['val/loss']:.4f}")
            logger.log_metrics(metrics, it)
            
            if cfg.training.save_checkpoint:
                checkpoint_path = output_dir / f"checkpoint_{it}.pt"
                tqdm.write(f"saving checkpoint {it} to {checkpoint_path}")
                raw_model = model._orig_mod if cfg.training.is_compile else model
                save_checkpoint(raw_model, optimizer, it, checkpoint_path)
        
    tqdm.write("Training finished...")
        
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
        model = model._orig_mod # ty: ignore
        install_kv_cache(model, batch_size=1, total_len=cfg.model.context_length+1000)
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
