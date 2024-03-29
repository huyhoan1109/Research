import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def init_transform(size, split='train'):
    if split == 'train' or split == 'val':
        return A.Compose([
            A.Resize(size, size),
            A.RandomResizedCrop(size, size, scale=(0.2, 1.0), interpolation=3),
            A.HorizontalFlip(),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])