CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.max_iters=500 \
    'logger.run_name=test_train_script'