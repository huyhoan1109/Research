CUDA_VISIBLE_DEVICES=0 python \
    -u train_endo_single.py \
    --config config/endo/cris_r101.yaml \
    --sg 0 \
    --early_stop 100 \
    --step main \
    --opts TRAIN.clip_pretrain exp/endo-RN101_pretrain_main_best_clip.pth