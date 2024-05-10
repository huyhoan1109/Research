CUDA_VISIBLE_DEVICES=$1 python \
    -u train_endo.py \
    --config config/endo/cris_vit16.yaml \
    --root_data /mnt/tuyenld/data/endoscopy/full_endo_data \
    --tsg $2 \
    --opts TRAIN.exp_name CRIS_VIT16_ENDO TRAIN.clip_pretrain exp/endo-ViT-B-16_best_model.pth