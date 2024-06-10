import torch
from torch.utils.data import Dataset
import json
import os
from utils.simple_tokenizer import Tokenizer
import cv2
import numpy as np
from endoscopy.transform import *

ENDO_ROOT = '/mnt/tuyenld/data/endoscopy/full_endo_data'

STEPS = {
    'pretrain': 'pretrain_lesion_split.json',
    'main': 'main_lesion_split.json',
    'pretrain_clip': 'pretrain_clip_split.json'
}

class EndosDataset(Dataset):
    def __init__(
            self, 
            input_size,
            word_length,
            root_path = ENDO_ROOT,
            dataset='./endoscopy', 
            split = 'train',
            step = 'main', 
            add_color = True,
            add_lesion = True
        ):
        super().__init__()
        self.split = split
        self.image_path = root_path + '/images'
        self.mask_path = root_path + '/mask_images'
        self.input_size = input_size
        self.word_length = word_length
        self.tokenizer = Tokenizer()
        self.add_color = add_color
        self.add_lesion = add_lesion
        self.transform = init_transform(input_size, split=split)
        self.step = step
        self.dataset = dataset
        self.data = self._process_data()

    def _load_classes(self):
        classes = json.load(open('./endoscopy/lesion_classes.json'))
        return classes

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.step == 'pretrain_clip':
            return self._get_clip_data(sample)
        else:
            return self._get_seg_data(sample)
    
    def _get_seg_data(self, sample):
        prompt = f'This is an image of {sample["position"]}.'
        if ('color' in sample.keys()) and self.add_color:
            prompt += f'The image is taken with {sample["color"]} mode.'
        if ('lesions' in sample.keys()) and self.add_lesion:
            lesions = ",".join(sample['lesions'])
            prompt += f'The patient has {lesions}'
            labels = ','.join(sample['labels'])
            labels = f'({labels})'
        else:
            labels = '()'
        ori_img = cv2.imread(os.path.join(self.image_path, sample['image']), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori_mask = cv2.imread(os.path.join(self.mask_path, sample['mask']), cv2.IMREAD_GRAYSCALE)
        tf = self.transform(image=rgb_img, mask=ori_mask)

        return {
            'image': tf['image'],
            'mask': tf['mask'] / 255.0, # change mask from [0, 255] => [0, 1]
            'prompt': prompt,
            'word': self.tokenizer.tokenize(prompt, self.word_length, True).squeeze(0),
            'img_id': sample['image'].rsplit('.')[0],
            'label': labels
        }

    def _get_clip_data(self, sample):
        prompt = f'This is an image of {sample["position"]}.'
        if ('color' in sample.keys()) and self.add_color:
            prompt += f'The image is taken with {sample["color"]} mode.'
        if ('lesions' in sample.keys()) and self.add_lesion:
            lesions = ",".join(sample['lesions'])
            prompt += f'The patient has {lesions}'
        ori_img = cv2.imread(os.path.join(self.image_path, sample['image']), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        tf = self.transform(image=rgb_img)
        return {
            'image': tf['image'],
            'prompt': prompt,
            'word': self.tokenizer.tokenize(prompt, self.word_length, True).squeeze(0),
        }
    
    def _process_data(self):
        data = {}
        self.classes = self._load_classes()
        metadata = json.load(open(f'{self.dataset}/meta.json'))
        split_data = json.load(open(f'{self.dataset}/{STEPS[self.step]}'))[self.split]
        if self.step == 'pretrain_clip':
            for idx, img_id in enumerate(split_data):
                data[idx] = {}
                data[idx]['image'] = img_id+'.jpeg'
                data[idx]['position'] = metadata[img_id]['position']
                label_ids = metadata[img_id]['labels']
                for label_id in label_ids:
                    data[idx]['lesions'].append(self.classes[label_id])
                if 'color' in metadata[img_id]:
                    data[idx]['color'] = metadata[img_id]['color']
        else:
            for idx, file_name in enumerate(split_data):
                data[idx] = {}
                split_name = file_name[:-4].rsplit('-')
                img_id = split_name[0]
                data[idx]['image'] = img_id+'.jpeg'
                data[idx]['mask'] = img_id+'.png'
                data[idx]['position'] = metadata[img_id]['position']
                if len(split_name) == 2:
                    label_ids = split_name[1:]
                    data[idx]['lesions'] = []
                    data[idx]['labels'] = []
                    for label_id in label_ids:
                        data[idx]['lesions'].append(self.classes[label_id])
                        data[idx]['labels'].append(label_id)
                if 'color' in metadata[img_id]:
                    data[idx]['color'] = metadata[img_id]['color']
        return data