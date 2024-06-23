# I. Preparation
1. Environment
    - [PyTorch](www.pytorch.org) (e.g. 1.10.0)
    - Other dependencies in `requirements.txt`
    - This code was implemented for `wandb` so before running the code make sure to create `.env` file that has `API_KEY` of your account.
2. Datasets
    - The model was experimented on 4 datasets: RefCOCO, RefCOCO+, G-Ref and Endo. 
    - The first 3 datasets API can be found in this [repository](https://github.com/lichengunc/refer.git) while Endo is our undisclosable dataset. The detailed instruction of the first 3 dataset is in [prepare_datasets.md](tools/prepare_datasets.md)
3. Pretrained weight for CLIP
    - Use `download_weight.py` in `pretrain` folder to download the specific CLIP weight.
    - Example:
        ```
        # Download ResNet-50 weight
        python download_weight.py --model RN50 --root pretrain
        ```
# II. Training 
To train the model for referring segmentation , please change the following parameters in `config` folder. The code is implemented for multiple GPU training so if you want single GPU, please add specific GPU device before the training code.
## RefCOCO, RefCOCO+, G-Ref
```
# Training RefCOCO dataset with ResNet-50 config on multiple GPU
python -u train.py --config config/refcoco/cris_r50.yaml
```
```
# Training RefCOCO dataset with ResNet-50 config on single GPU (device 0)
CUDA_VISIBLE_DEVICES=0 python -u train.py --config config/refcoco/cris_r50.yaml
```
```
# Add scale gate to the above model
CUDA_VISIBLE_DEVICES=0 python -u train.py --config config/refcoco/cris_r50.yaml --sg 1
```

## Endo
```
# Training Endo dataset faster with ResNet-50 config on single GPU (device 0)
CUDA_VISIBLE_DEVICES=0 python -u train_endo_single.py --config config/endo/cris_r50.yaml
```

# III. INFERENCE 
Trained model weight will be stored in `exp` folder. The code is implemented for single GPU evaluation so you must add specific GPU device before the evaluation code.
```
# Evaluating RefCOCO dataset with `exp/RN50-best_weight.pth` on single GPU (device 0)
CUDA_VISIBLE_DEVICES=0 python -u test.py --config config/refcoco/cris_r50.yaml --opts TRAIN.resume exp/RN50-best_weight.pth
```
```
# Evaluating RefCOCO dataset with `exp/RN50-best_weight.pth` on single GPU (device 0) without drawing mask
CUDA_VISIBLE_DEVICES=0 python -u test.py --config config/refcoco/cris_r50.yaml --opts TRAIN.resume exp/RN50-best_weight.pth TEST.visualize False
```

# Credited

The baseline was credited by [CRIS](https://arxiv.org/pdf/2111.15174) and some improvements that i made.