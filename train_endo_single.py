import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
import wandb

import cv2
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR

from endoscopy.dataset import EndosDataset, STEPS
import utils.config as config
from utils.misc import WandbLogger, count_parameters
from engine.engine_endo_single import train, validate
from model import build_segmenter
from torch.utils.data import RandomSampler

from dotenv import dotenv_values

env_var = dotenv_values(".env")

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

def get_sampler(dataset, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator)
    return sampler

def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch CLIP Endoscopy Segmentation')
    parser.add_argument('--config', default='path to xxx.yaml', type=str, help='config file')
    parser.add_argument('--sg', default=0, type=int, help='add scale gate.')
    parser.add_argument('--jit', default=0, type=int, help='jit mode.')
    parser.add_argument('--early_stop', default=50, type=int, help='set early stop epoch')
    parser.add_argument('--step', choices=STEPS.keys(), help='Choose step')
    parser.add_argument('--use_relu', default=0, type=int, help='use relu scale gate.')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER, help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.__setattr__('sg', args.sg)
    cfg.__setattr__('jit', args.jit)
    cfg.__setattr__('early_stop', args.early_stop)
    cfg.__setattr__('step', args.step)
    return cfg


@logger.catch
def main():
    args = get_parser()
    main_worker(args)

def main_worker(args):
    args.output_dir = os.path.join(args.output_folder, args.exp_name)

    # wandb
    wandb.login(key=env_var['API_KEY'])
    wlogger = WandbLogger(args)
    wlogger.init_logger(
        project="CRIS-ENDO",
        mode="online"
    )

    # build model
    model, param_list = build_segmenter(args)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    logger.info(model)
    logger.info(f'Total parameters: {count_parameters(model)}')

    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(
        param_list,
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )

    scaler = amp.GradScaler()
    
    train_data = EndosDataset(
        input_size=args.input_size,
        word_length=args.word_len,
        split='train',
        step=args.step
    )
    val_data = EndosDataset(
        input_size=args.input_size,
        word_length=args.word_len,
        split='test',
        step=args.step
    )

    train_sampler = get_sampler(train_data, args.manual_seed)
    val_sampler = get_sampler(val_data, args.manual_seed)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        num_workers=args.workers_val,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max = len(train_loader) * args.epochs,
        eta_min = args.base_lr / 1000,
    )

    best_IoU = 0.0
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["iou"]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            raise ValueError(f"=> resume failed! no checkpoint found at '{args.resume}'. Please check args.resume again!")
    
    early_epoch = args.early_stop
    # start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # train
        train(train_loader, model, optimizer, scheduler, scaler, epoch_log, args, wlogger)

        # evaluation
        # use marco score
        iou, dice, prec_dict = validate(val_loader, model, epoch_log, args)

        # loggin
        val_log = dict({
            'eval/iou': iou,
            'eval/dice': dice,
            'eval/step': epoch_log
        })
        for key in prec_dict.keys():
            log_key = key.lower()
            val_log[f'eval/{log_key}'] = prec_dict[key]
        wlogger.logging(val_log)
        
        # save model
        if iou >= best_IoU and early_epoch > 0:
            best_IoU = iou
            early_epoch = args.early_stop
            model_name = f"best_model_sg.pth" if args.sg else f"best_model_base.pth"
            os.makedirs(os.path.join(args.output_dir, args.step), exist_ok=True)
            model_path = os.path.join(args.output_dir, args.step, model_name)
            torch.save(
                {
                    'epoch': epoch_log,
                    'iou': iou,
                    'prec': prec_dict,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, 
                model_path
            )
        else:
            early_epoch -= 1
            if early_epoch == 0:
                break
    torch.cuda.empty_cache()

    wlogger.finish()

    logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)