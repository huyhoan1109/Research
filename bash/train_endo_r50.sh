CUDA_VISIBLE_DEVICES=$1 python \
    -u train_endo.py \
    --config config/endo/cris_r50.yaml \
    --root_data /mnt/tuyenld/data/endoscopy/full_endo_data \
    --tsg $2 \
    --opts TRAIN.exp_name CRIS_R50_ENDO TRAIN.clip_pretrain pretrain/RN50.pt