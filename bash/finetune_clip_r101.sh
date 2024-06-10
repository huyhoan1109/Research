# CUDA_VISIBLE_DEVICES=$1 python \
#     -u finetune_clip.py \
#     --config tuning/config.yaml \
#     --clip_pretrain pretrain/RN101.pt \
#     --prefix_name endo-RN101 \
#     --early_stop $2 \
#     --resume $3 \
#     --run_id $4 \
#     --continue_training $5

CUDA_VISIBLE_DEVICES=1 python \
    -u finetune_clip.py \
    --config tuning/config.yaml \
    --clip_pretrain pretrain/RN101.pt \
    --prefix_name endo-RN101 \
    --early_stop 100 