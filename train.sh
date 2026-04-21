export WANDB_MODE=offline
export WANDB_DIR=/lpai/output/models/wandb_logs

export CUDA_VISIBLE_DEVICES=2,3

NUM_GPUS=2

# # Multi-GPU DDP launch
# torchrun --nproc_per_node=8 -m train.align_training_clip --clip_model_name ViT-B-16 --pretrained openai --dataset imagenet \
#     --imagenet_root /lpai/dataset/imagenet-1k/0-1-0 --template std --output_normalize False \
#     --total_epochs 2 \
#     --steps 40000  \
#     --warmup 175 \
#     --batch_size 128 --loss l2 --loss_clean l2 --opt adamw --lr 1e-5 --wd 1e-4 --inner_loss l2 --wandb False \
#     --output_dir /lpai/output/models --clean_weight 1. --penalty_weight 0.5 --kernel_dino polynomial \
#     --kernel_clip polynomial --gamma 0.0032 --coef0 0.191623 --experiment_name exp_1 --log_freq 1 --eval_freq 10

# Single GPU fallback (still works):
# python -m train.align_training_clip --clip_model_name ViT-B-16 ...

# CC3M WebDataset example:
torchrun --nproc_per_node=$NUM_GPUS -m train.align_training_clip --clip_model_name ViT-B-16 --pretrained openai \
    --dataset cc3m --wds_path "/lpai/dataset/cc3m-webdataset/0-1-0/cc3m/cc3m-train-{0000..0575}.tar" \
    --wds_train_length 3000000 \
    --imagenet_root /lpai/dataset/imagenet-1k/0-1-0 --template std --output_normalize False \
    --total_epochs 2 --warmup 410 \
    --batch_size 128 --loss l2 --loss_clean l2 --opt adamw --lr 1e-5 --wd 1e-4 --inner_loss l2 --wandb False \
    --output_dir /lpai/output/models --clean_weight 1. --penalty_weight 0.5 --kernel_dino polynomial \
    --kernel_clip polynomial --gamma 0.0032 --coef0 0.191623 --experiment_name exp_cc3m --log_freq 1 --eval_freq 10