CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    model.ffn_type=silu \
    model.d_ff=2048 \
    optimizer.max_lr=1e-2 \
    optimizer.min_lr=1e-3 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_swiglu2silu'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    model.use_post_norm=true \
    optimizer.max_lr=1e-2 \
    optimizer.min_lr=1e-3 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_prenorm2postnorm'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    model.remove_rmsnorm=true \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_wo_rmsnorm'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    model.remove_rope=true \
    optimizer.max_lr=1e-2 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_wo_rope'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    optimizer.max_lr=1e-3 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_lr${optimizer.max_lr}'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    model.ffn_type=silu \
    model.d_ff=2048 \
    optimizer.max_lr=1e-3 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_lr${optimizer.max_lr}_swiglu2silu'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    model.use_post_norm=true \
    optimizer.max_lr=1e-3 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_lr${optimizer.max_lr}_prenorm2postnorm'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    model.remove_rmsnorm=true \
    optimizer.max_lr=1e-3 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_lr${optimizer.max_lr}_wo_rmsnorm'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    model.remove_rope=true \
    optimizer.max_lr=1e-3 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_lr${optimizer.max_lr}_wo_rope'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    optimizer.max_lr=3e-4 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_lr${optimizer.max_lr}'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    model.ffn_type=silu \
    model.d_ff=2048 \
    optimizer.max_lr=3e-4 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_lr${optimizer.max_lr}_swiglu2silu'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    model.use_post_norm=true \
    optimizer.max_lr=3e-4 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_lr${optimizer.max_lr}_prenorm2postnorm'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    model.remove_rmsnorm=true \
    optimizer.max_lr=3e-4 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_lr${optimizer.max_lr}_wo_rmsnorm'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=128 \
    model.remove_rope=true \
    optimizer.max_lr=3e-4 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_lr${optimizer.max_lr}_wo_rope'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py -m \
    +model.add_qknorm=True \
    training.batch_size=128 \
    optimizer.max_lr=1e-2,1e-3,3e-4 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_baseline_lr${optimizer.max_lr}_qknorm'