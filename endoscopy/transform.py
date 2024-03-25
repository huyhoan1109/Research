import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def init_transform(size):
    return A.Compose([
        A.Resize(size, size), 
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ), 
        ToTensorV2()
    ]
)