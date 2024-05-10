CUDA_VISIBLE_DEVICES=$1 python \
    -u finetune_clip.py \
    --config tuning/config.yaml \
    --clip_pretrain pretrain/ViT-B-16.pt \
    --prefix_name endo-ViT-B-16 \
    --resume $2