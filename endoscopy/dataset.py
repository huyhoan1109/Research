import torch
from torch.utils.data import Dataset
import json
import os
from utils.simple_tokenizer import Tokenizer
import cv2
from endoscopy.transform import *

ROOT_PATH = '/mnt/tuyenld/data/endoscopy/full_endo_data'

class EndosDataset(Dataset):
    def __init__(
            self, 
            input_size,
            word_length,
            root_path=ROOT_PATH,
            dataset='./endoscopy', 
            split = 'train', 
            add_lesion = True,
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        ):
        super().__init__()
        self.image_path = root_path + '/images'
        self.mask_path = root_path + '/mask_images'
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.tokenizer = Tokenizer()
        self.split = split
        self.device = device
        self.add_lesion = add_lesion
        self.transform = init_transform(input_size)

        with open(f'{dataset}/{split}.txt', 'r') as f:
            ids = f.readlines()
            f.close()
        metadata = json.load(open(f'{dataset}/metadata.json'))
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
        
        if ('lesion' in sample.keys()) and self.add_lesion:
            lesions = ", ".join(sample['lesion'])
            prompt += f' Some diagnosed lesions are {lesions}.'

        transformed = self.transform(image=img, mask=mask)

        return {
            'image': transformed['image'],
            'mask': transformed['mask'] / 255.0, # change mask from [0, 255] => [0, 1]
            'prompt': prompt,
            'word': self.tokenizer.tokenize(prompt, self.word_length, True).squeeze(0),
            'img_id': sample['image'].rsplit('.')[0],
        }
