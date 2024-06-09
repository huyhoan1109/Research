import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from endoscopy.dataset import DATASETS, ENDO_ROOT

MODEL = {
    'r50': 'CRIS_R50',
    'r101': 'CRIS_R101',
    'vit16': 'CRIS_VIT16',
}

def draw_masked_image(image, mask, name='drawed.png'):
    _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    # copy where we'll assign the new values
    drawed = np.copy(image)
    drawed[(mask==255).all(-1)] = [0,255,0]
    drawed = cv2.addWeighted(drawed, 0.3, image, 0.7, 0, drawed)
    cv2.imwrite(
        filename=name,
        img=drawed
    )

def get_args():
    parser = argparse.ArgumentParser(description='Draw mask')
    parser.add_argument('--task', default=0, choices=DATASETS.keys(), type=str, help='config file')
    parser.add_argument('--model', type=str, choices=MODEL.keys(), help='add transformer scale gate.')

if __name__ == '__main__':
    args = get_args()
    vis_folder = f'exp/endo/{MODEL[args.model_id]}/{DATASETS[args.task]}/vis'
    draw_folder = f'exp/endo/{MODEL[args.model_id]}/{DATASETS[args.model]}/draw'
    os.makedirs(draw_folder, exist_ok=True)
    file_list = [f for f in os.listdir(vis_folder) if os.path.isfile(os.path.join(vis_folder, f))]
    for file_name in tqdm(file_list):
        names = file_name.rsplit('-')
        img_id = names[0]
        image_path = ENDO_ROOT+f'/images/{img_id}.jpeg'
        image = cv2.imread(image_path)
        mask = cv2.imread(vis_folder+f'/{file_name}')
        height, width = mask.shape[0], mask.shape[1]
        image = cv2.resize(image, (height, width), interpolation = cv2.INTER_CUBIC)
        if names[-1] == 'mask.png':
            drawed_name = draw_folder+f'/{img_id}-GT.png'
        else:
            drawed_name = draw_folder+f'/{img_id}-Pred.png'
        draw_masked_image(image, mask, drawed_name)  




