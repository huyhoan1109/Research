import cv2
import numpy as np
import os
from tqdm import tqdm
import torch

ENDO_PATH = '/mnt/tuyenld/data/endoscopy/full_endo_data'
MODEL_NAME = 'CRIS_VIT16'
VIS_FOLDER = f'exp/endo/{MODEL_NAME}/vis'
DRAWED_FOLDER = f'exp/endo/{MODEL_NAME}/draw'

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

if __name__ == '__main__':
    os.makedirs(DRAWED_FOLDER, exist_ok=True)
    file_list = [f for f in os.listdir(VIS_FOLDER) if os.path.isfile(os.path.join(VIS_FOLDER, f))]
    for file_name in tqdm(file_list):
        names = file_name.rsplit('-')
        img_id = names[0]
        image_path = ENDO_PATH+f'/images/{img_id}.jpeg'
        image = cv2.imread(image_path)
        mask = cv2.imread(VIS_FOLDER+f'/{file_name}')
        height, width = mask.shape[0], mask.shape[1]
        image = cv2.resize(image, (height, width), interpolation = cv2.INTER_CUBIC)
        if names[-1] == 'mask.png':
            drawed_name = DRAWED_FOLDER+f'/{img_id}-GT.png'
        else:
            drawed_name = DRAWED_FOLDER+f'/{img_id}-Pred.png'
        draw_masked_image(image, mask, drawed_name)  




