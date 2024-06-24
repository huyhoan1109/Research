CUDA_VISIBLE_DEVICES=0 python \
    -u test_endo_single.py \
    --config config/endo/cris_vit16.yaml \
    --sg 0 \
    --step colon \
    --opts TRAIN.clip_pretrain exp/endo-ViT-B-16_pretrain_main_best_clip.pth TRAIN.resume exp/endo/CRIS_VIT16/main/best_model_base.pth