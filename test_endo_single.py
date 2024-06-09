import argparse
import os
import warnings

import cv2
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
from loguru import logger

import utils.config as config
from engine.engine_endo_single import inference
from model import build_segmenter
from endoscopy.dataset import EndosDataset, DATASETS
from utils.misc import setup_logger, count_parameters

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch CLIP Endoscopy Segmentation')
    parser.add_argument('--config', default='path to xxx.yaml', type=str, help='config file')
    parser.add_argument('--tsg', default=0, type=int, help='add transformer scale gate.')
    parser.add_argument('--jit', default=0, type=int, help='jit mode.')
    parser.add_argument('--task', default=0, choices=DATASETS.keys(), type=int, help='Choose task.')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER, help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.__setattr__('tsg', args.tsg)
    cfg.__setattr__('jit', args.jit)
    cfg.__setattr__('task', args.task)
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, DATASETS[args.task], "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    logger.info(args)

    # build dataset & dataloader
    test_data = EndosDataset(
        task=args.task,
        input_size=args.input_size,
        word_length=args.word_len,
        split='test'
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model, _ = build_segmenter(args)
    model = model.cuda()
    logger.info(model)
    logger.info(f'Total parameters: {count_parameters(model)}')
    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.resume))
    else:
        raise ValueError(f"=> resume failed! no checkpoint found at '{args.model_dir}'. Please check args.resume again!")
    # build model
    inference(test_loader, model, args)

if __name__ == '__main__':
    main()