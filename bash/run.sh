CUDA_VISIBLE_DEVICES=1 python \
    -u test_endo_single.py \
    --config config/endo/cris_vit16.yaml \
    --sg 1 \
    --step main \
    --opts TRAIN.clip_pretrain exp/endo-ViT-B-16_pretrain_main_best_clip.pth TRAIN.resume exp/endo/CRIS_VIT16/main/best_model_sg.pth