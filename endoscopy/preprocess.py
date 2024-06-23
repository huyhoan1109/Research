import json
import os
import random as rd
import itertools

def prepare_position_split(
    meta_file='./endoscopy/meta.json',
    lesion_mask_folder='/mnt/tuyenld/data/endoscopy/full_endo_data/lesion_mask_images',
    train_size=0.9
):
    lesion_mask_files = os.listdir(lesion_mask_folder)
    meta = json.load(open(meta_file))
    position_data = {}
    for idx, mask_file in enumerate(lesion_mask_files):
        name = mask_file.rsplit('-')
        img = name[0]
        img_id = img.rsplit('.')[0]
        position = meta[img_id]['position']
        if position not in position_data.keys():
            position_data[position] = []
        position_data[position].append(mask_file)
    all_data = {
        'train': [],
        'test': [] 
    }
    for position in position_data.keys():
        data = position_data[position]
        
        single_lesion = []
        multi_lesion = []

        for idx, mask_file in enumerate(data):
            name = mask_file.rsplit('-')
            img = name[0]
            img_id = img.rsplit('.')[0]
            if img_id not in single_lesion:
                single_lesion.append(img_id)
            else:
                if img_id not in multi_lesion:
                    multi_lesion.append(img_id)
        
        for value in multi_lesion:
            single_lesion.remove(value)

        train_single_lesion_data = rd.sample(single_lesion, int(len(single_lesion) * train_size))   
        train_multi_lesion_data = rd.sample(multi_lesion, int(len(multi_lesion) * train_size))
        
        train_data = list(train_single_lesion_data) + list(train_multi_lesion_data)

        train_position_data = []
        test_position_data = []

        for idx, mask_file in enumerate(data):
            name = mask_file.rsplit('-')
            img = name[0]
            img_id = name[1].rsplit('.')[0]
            if img_id in train_data:
                train_position_data.append(mask_file)
            else:
                test_position_data.append(mask_file)            

        store_data = {
            'train': train_position_data,
            'test': test_position_data
        }

        all_data['train'].append(train_position_data)
        all_data['test'].append(test_position_data)
        os.makedirs('./endoscopy/new', exist_ok=True)
        with open(f'./endoscopy/new/{position}_split.json', 'w') as fp:
            json.dump(store_data, fp, sort_keys=True, indent=4)
    
    all_data['train'] = list(itertools.chain.from_iterable(all_data['train']))
    all_data['test'] = list(itertools.chain.from_iterable(all_data['test']))
    with open(f'./endoscopy/new/all_position_split.json', 'w') as fp:
        json.dump(all_data, fp, sort_keys=True, indent=4)


if __name__ == '__main__':
    prepare_position_split()