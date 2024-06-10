import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def init_transform(input_size, split='train'):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    if split == 'train' or split == 'val':
        return A.Compose([
            A.Resize(input_size, input_size),
            A.RandomResizedCrop(size=(input_size, input_size), scale=(0.2, 1.0), interpolation=3),
            A.HorizontalFlip(),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])