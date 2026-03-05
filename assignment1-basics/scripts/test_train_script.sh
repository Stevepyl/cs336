WANDB_MODE=offline uv run train/train_model.py \
    training.max_iters=5 \
    training.eval_iters=2 \
    training.log_interval=1 \
    training.is_compile=false \
    logger.run_name="test_train_script"