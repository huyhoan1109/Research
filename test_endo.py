import argparse
import os
import warnings

import cv2
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
from loguru import logger

import utils.config as config
from engine.engine_endo import inference
from model import build_segmenter
from endoscopy.dataset import EndosDataset
from utils.misc import setup_logger

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch CLIP Endoscopy Segmentation')
    parser.add_argument('--config', default='path to xxx.yaml', type=str, help='config file')
    parser.add_argument('--tsg', default=0, type=int, help='add transformer scale gate.')
    parser.add_argument('--jit', default=0, type=int, help='jit mode.')
    parser.add_argument('--root_data', type=str, help='load root path for endoscopy data')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER, help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.__setattr__('tsg', args.tsg)
    cfg.__setattr__('jit', args.jit)
    cfg.__setattr__('root_data', args.root_data)
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    # logger
    setup_logger(
        args.output_dir,
        distributed_rank=0,
        filename="test.log",
        mode="a"
    )
    logger.info(args)

    # build dataset & dataloader
    test_data = EndosDataset(
        input_size=args.input_size,
        word_length=args.word_len,
        mode='test'
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # build model
    model, _ = build_segmenter(args)
    model = torch.nn.DataParallel(model).cuda()
    logger.info(model)

    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.resume))
    else:
        raise ValueError(f"=> resume failed! no checkpoint found at '{args.model_dir}'. Please check args.resume again!")

    # inference
    inference(test_loader, model, args)


if __name__ == '__main__':
    main()
