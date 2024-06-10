import json
import os
import random as rd

def prepare_pretrain_split(
        src_file='./endoscopy/main_lesion_split.json', 
        dest_file='./endoscopy/pretrain_lesion_split.json',
        process_percent=0.8
    ):
    all_split =  json.load(open(src_file))
    train_data = all_split['train']
    test_data = all_split['test']
    pretrain_ids = rd.sample(range(len(train_data)), int(len(train_data) * process_percent))
    pretest_ids = rd.sample(range(len(test_data)), int(len(test_data) * process_percent))
    pretrain_split = {
        'train': [],
        'test': []
    }
    for i in range(len(pretrain_ids)):
        pretrain_split['train'].append(train_data[pretrain_ids[i]])
    for i in range(len(pretest_ids)):
        pretrain_split['test'].append(test_data[pretest_ids[i]])
    with open(dest_file, 'w') as fp:
        json.dump(pretrain_split, fp, sort_keys=True, indent=4)

def prepare_lesion_split(
        lesion_mask_folder='/mnt/tuyenld/data/endoscopy/full_endo_data/lesion_mask_images',
        meta_file='./endoscopy/meta.json',
        dest='./endoscopy/main_lesion_split.json',
        train_percent=0.8
    ):
        lesion_mask_files = os.listdir(lesion_mask_folder)
        single_lesion = {}
        multi_lesion = {}
        max_lesion = 0
        for idx, mask_file in enumerate(lesion_mask_files):
            name = mask_file.rsplit('-')
            img = name[0]
            if len(name) == 2:
                label = name[1].rsplit('.')[0]
            else:
                label = None
            if img not in single_lesion:
                single_lesion[img] = label
            else:
                if img not in multi_lesion.keys():
                    multi_lesion[img] = [single_lesion[img], label]
                else:
                    multi_lesion[img].append(label)
                    if len(multi_lesion[img]) > max_lesion:
                        max_lesion = len(multi_lesion[img])
        
        for key in multi_lesion.keys():
            del single_lesion[key]
        
        train_single_lesion_data = rd.sample(single_lesion.keys(), int(len(single_lesion.keys()) * train_percent))
        test_single_lesion_data = list(set(single_lesion.keys()) - set(train_single_lesion_data))
        
        train_multi_lesion_data = rd.sample(multi_lesion.keys(), int(len(multi_lesion.keys()) * train_percent))
        test_multi_lesion_data = list(set(multi_lesion.keys()) - set(train_multi_lesion_data))
        
        train_lesion_data = list(train_single_lesion_data) + list(train_multi_lesion_data)
        test_lesion_data = list(test_single_lesion_data) + list(test_multi_lesion_data)

        
        rd.shuffle(train_lesion_data)
        rd.shuffle(test_lesion_data)

        split_lesion = {
            'train': [],
            'test': []
        }

        metadata = json.load(open(meta_file))
        for i in range(len(train_lesion_data)):
            img_id = train_lesion_data[i].rsplit('.')[0]
            current_meta = metadata[img_id]
            if len(current_meta['labels']) > 0:
                for label in current_meta['labels']:
                    file_name = f'{img_id}-{label}.png'
                    split_lesion['train'].append(file_name)
            else:
                file_name = f'{img_id}.png'
                split_lesion['train'].append(file_name)
        for i in range(len(test_lesion_data)):
            img_id = test_lesion_data[i].rsplit('.')[0]
            current_meta = metadata[img_id]
            if len(current_meta['labels']) > 0:
                for label in current_meta['labels']:
                    file_name = f'{img_id}-{label}.png'
                    split_lesion['test'].append(file_name)
            else:
                file_name = f'{img_id}.png'
                split_lesion['test'].append(file_name)
        
        with open(dest, 'w') as fp:
            json.dump(split_lesion, fp, sort_keys=True, indent=4)

def prepare_pretrain_clip_split(
        src_file='./endoscopy/pretrain_lesion_split.json', 
        dest_file='./endoscopy/pretrain_clip_split.json',
    ):
    src_split =  json.load(open(src_file))
    train_data = src_split['train']
    test_data = src_split['test']
    train_ids = []
    test_ids = []
    for data in train_data:
        img_id = data.rsplit('.')[0].rsplit('-')[0]
        if img_id not in train_ids:
            train_ids.append(img_id)
    for data in test_data:
        img_id = data.rsplit('.')[0].rsplit('-')[0]
        if img_id not in test_ids:
            test_ids.append(img_id)
    pretrain_clip_split = {
        'train': train_ids,
        'test': test_ids
    }
    with open(dest_file, 'w') as fp:
        json.dump(pretrain_clip_split, fp, sort_keys=True, indent=4)


if __name__ == '__main__':
    prepare_lesion_split()
    prepare_pretrain_split()
    prepare_pretrain_clip_split()