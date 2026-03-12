# Use https://github.com/donglinkang2021/cs336-assignment1-basics/blob/main/scripts/leaderboard/leaderboard_20251019_1.sh
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model_muon.py -m \
    +model.add_qknorm=True \
    +model.tie_embeddings=True \
    model.vocab_size=32000 \
    model.num_layers=12 \
    model.num_heads=12 \
    model.d_model=768 \
    model.d_ff=2048 \
    data.path=data/owt \
    data.tokenizer_path=tokenizer/owt \
    training.batch_size=128 \
    training.max_iters=30000 \
    optimizer.warmup_iters=1000,2000,3000 \
    optimizer.max_lr=1e-2 \
    optimizer.min_lr=0 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.95,0.999]" \
    'logger.run_name=final_qknorm_muon_betas1_gpt2_tie_embed_ws${optimizer.warmup_iters}'
