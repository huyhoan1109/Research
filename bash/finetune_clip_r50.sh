CUDA_VISIBLE_DEVICES=0 python \
    -u finetune_clip.py \
    --config tuning/config.yaml \
    --clip_pretrain pretrain/RN50.pt \
    --prefix_name endo-RN50 \
    --early_stop 35 \
    --step pretrain_main