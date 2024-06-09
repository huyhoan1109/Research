import torch
from torch.utils.data import Dataset
import json
import os
from utils.simple_tokenizer import Tokenizer
import cv2
import numpy as np
from endoscopy.transform import *

ENDO_ROOT = '/mnt/tuyenld/data/endoscopy/full_endo_data'

DATASETS = {
    0: 'full_endo_data',
    1: 'polyp',
    2: 'ung_thu_da_day_20230620', 
    3: 'ung_thu_thuc_quan_20230620', 
    4: 'viem_da_day_20230620', 
    5: 'viem_thuc_quan_20230620', 
    6: 'viem_loet_hoanh_ta_trang_20230620',
}

class EndosDataset(Dataset):
    def __init__(
            self, 
            input_size,
            word_length,
            task = 0,
            root_path = ENDO_ROOT,
            dataset='./endoscopy', 
            split = 'train', 
            add_color = True,
            add_hp = True,
            add_lesion = True,
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        ):
        super().__init__()
        self.split = split
        self.task = task
        self.image_path = root_path + '/images'
        self.mask_path = root_path + '/label_images'
        if DATASETS[task] == 'full_endo_data':
            image_file = json.load(open(f'{dataset}/split_full_endo.json'))[split]['images']
        elif task == 1:
            image_file = json.load(open(f'{dataset}/split_polyp.json'))[split]['images']
        else:
            image_file = json.load(open(f'{dataset}/split_lesion.json'))[DATASETS[task]][split]['images']
        metadata = json.load(open(f'{dataset}/metadata.json'))
        self.data = {}
        for idx, key in enumerate(image_file):
            new_key = key.rsplit('/')[-1].rsplit('.')[0]
            self.data[idx] = metadata[new_key]
        self.input_size = input_size
        self.word_length = word_length
        self.tokenizer = Tokenizer()
        self.device = device
        self.add_color = add_color
        self.add_hp = add_hp
        self.add_lesion = add_lesion
        self.transform = init_transform(input_size, split=split)

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        sample = self.data[idx]
                
        prompt = f'This is an image of {sample["position"]}.'
        if ('lesion' in sample.keys()) and self.add_lesion:
            lesions = ", ".join(sample['lesion'])
            prompt += f'Some diagnosed lesions are {lesions}.'
        if ('color' in sample.keys()) and self.add_color:
            prompt += f'The image is taken with {sample["color"]} mode.'
        if ('HP' in sample.keys()) and self.add_hp:
            prompt += f'This person is {sample["HP"]} to HP.'
        
        ori_img = cv2.imread(os.path.join(self.image_path, sample['image']), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img).permute(2, 0, 1)

        # ori_mask = cv2.imread(os.path.join(self.mask_path, sample['mask']))
        # for i in range(len(gts)):
        #     mask = gts[i].astype(np.uint8) 
        #     cv2.imwrite(f'dump/{idx}-{i}-mask.png', mask)
        
        # cv2.imwrite(f'dump/mask.png', pred * 255.0)
        # mask_tensor = torch.tensor(ori_mask).permute(2, 0, 1)
        # gts = np.array([mask_img == c for c in self.colors])
        # mask_tensor = torch.tensor(gts).permute(0, 3, 1, 2)
        # for i in range(len(mask_tensor)):
        #     mask_tensor[i] = self.transform(mask_tensor[i])
        # mask_tensor = self.transform(mask_tensor)

        return {
            'image': self.transform(img_tensor),
            # 'mask': mask_tensor, # change mask from [0, 255] => [0, 1]
            'prompt': prompt,
            'word': self.tokenizer.tokenize(prompt, self.word_length, True).squeeze(0),
            'img_id': sample['image'].rsplit('.')[0],
        }