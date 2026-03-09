CUDA_VISIBLE_DEVICES=0 uv run train/train_model.py \
    training.batch_size=128 \
    optimizer.max_lr=1e-2 \
    optimizer.min_lr=1e-3 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=baseline-ts'