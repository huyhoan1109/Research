import torch
from torchvision.transforms import v2

def init_transform(size, split='train'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if split == 'train' or split == 'val':
        return v2.Compose([
            v2.RandomResizedCrop(size=(size, size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
    else:
        return v2.Compose([
            v2.Resize(size=(size, size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])