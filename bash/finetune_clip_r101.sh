CUDA_VISIBLE_DEVICES=$1 python \
    -u finetune_clip.py \
    --config tuning/config.yaml \
    --clip_pretrain pretrain/RN101.pt \
    --prefix_name endo-RN101 \
    --resume $2 \
    --run_id $3 \
    --continue_training $4