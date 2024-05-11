CUDA_VISIBLE_DEVICES=$1 python \
    -u train_endo.py \
    --config config/endo/cris_r50.yaml \
    --root_data /mnt/tuyenld/data/endoscopy/full_endo_data \
    --tsg $2 \
    --early_stop $3 \
    --opts TRAIN.clip_pretrain exp/endo-RN50_best_model.pth