CUDA_VISIBLE_DEVICES=$1 python \
    -u fine_tune_clip.py \
    --config tuning/config.yaml \
    --clip_pretrain pretrain/RN50.pt \
    --prefix_name endo-RN50 \
    --resume $2