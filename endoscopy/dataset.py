import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms
import json
import os
from utils.simple_tokenizer import Tokenizer
import numpy as np
import cv2
from endoscopy.transform import *

dataset_path = 'endoscopy'
root_endo = '/mnt/tuyenld/data/endoscopy/full_endo_data'
# root_endo = '/home/Downloads/full_endo_data'
image_path = root_endo + '/images'
mask_path = root_endo + '/mask_images'
labels_path = root_endo + '/label_images'

class EndosDataset(Dataset):
    def __init__(
            self, 
            input_size,
            word_length,
            image_path=image_path,
            mask_path=mask_path,
            file_path='endoscopy/metadata.json', 
            split = 'train', 
            add_lesion = True,
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        ):
        self.image_path = image_path
        self.mask_path = mask_path
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.tokenizer = Tokenizer()
        self.split = split
        self.device = device
        self.add_lesion = add_lesion
        self.transform = init_transform(input_size)

        with open(f'{dataset_path}/{split}.txt', 'r') as f:
            ids = f.readlines()
            f.close()
        metadata = json.load(open(file_path))
        self.data = dict()
        for id in ids:
            id = id.strip('\n')
            self.data[id] = metadata[id]

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        sample = self.data[f"{idx}"]
        
        ori_img = cv2.imread(os.path.join(self.image_path, sample['image']))
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(os.path.join(self.mask_path, sample['mask']), cv2.IMREAD_GRAYSCALE)
                
        prompt = f'This is an image of {sample["position"]}. The image is taken with {sample["color"]} mode.'
        
        if sample['lesion'] and self.add_lesion:
            prompt += f'Some diagnosed lesions are {", ".join(sample['lesion'])}.'

        transformed = self.transform(image=img, mask=mask)

        return {
            'image': transformed['image'],
            'mask': transformed['mask'],
            'prompt': prompt,
            'word': self.tokenizer.tokenize(prompt, self.word_length, True).squeeze(0),
            'img_name': sample['image'],
        }
