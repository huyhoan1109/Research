CUDA_VISIBLE_DEVICES=0 python \
    -u finetune_clip.py \
    --config tuning/config.yaml \
    --clip_pretrain pretrain/ViT-B-16.pt \
    --prefix_name endo-ViT-B-16 \
    --early_stop 35 \
    --step pretrain_main