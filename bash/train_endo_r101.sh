CUDA_VISIBLE_DEVICES=$1 python \
    -u train_endo.py \
    --config config/endo/cris_r101.yaml \
    --root_data /mnt/tuyenld/data/endoscopy/full_endo_data \
    --tsg $2 \
    --opts TRAIN.exp_name CRIS_R101_ENDO \
           TRAIN.clip_pretrain pretrain/RN101.pt