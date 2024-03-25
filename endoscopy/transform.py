import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def init_transform(size):
    return A.Compose([
        A.RandomCrop(width=size, height=size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2)
    ])