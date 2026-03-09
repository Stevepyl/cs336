CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py -m \
    training.batch_size=256,128,64,32,16,8,4,2,1 \
    optimizer.max_lr=1e-3 \
    optimizer.min_lr=1e-4 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_batchsize${training.batch_size}'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=512 \
    optimizer.max_lr=1e-3 \
    optimizer.min_lr=1e-4 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_batchsize${training.batch_size}'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=768 \
    optimizer.max_lr=1e-3 \
    optimizer.min_lr=1e-4 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_batchsize${training.batch_size}'

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline uv run train/train_model.py \
    training.batch_size=1024 \
    optimizer.max_lr=1e-3 \
    optimizer.min_lr=1e-4 \
    optimizer.weight_decay=0.01 \
    optimizer.max_l2_norm=2.0 \
    optimizer.betas="[0.9,0.95]" \
    'logger.run_name=ts_batchsize${training.batch_size}'

# run large batch size experiments seperately for 
# 1. OOM risk isolation. Large batch sizes (512, 768, 1024) are more likely to hit GPU OOM. If they were in the multirun sweep and one
#    crashes, Hydra's -m would either abort the whole sweep or leave you with inconsistent results. Running them separately means a failure
#    doesn't cascade. 
# 3. Sequential execution concern — Hydra -m runs jobs one after another in the same process. For very large batch sizes that stress memory,
#    it's safer to run them in fresh processes so GPU memory is fully released between runs.
