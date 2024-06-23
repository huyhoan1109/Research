CUDA_VISIBLE_DEVICES=0 python \
    -u finetune_clip.py \
    --config tuning/config.yaml \
    --clip_pretrain pretrain/RN101.pt \
    --prefix_name endo-RN101 \
    --early_stop 35 \
    --step pretrain_main \
