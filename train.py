import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
from functools import partial

import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler 
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config
import wandb
from utils.misc import WandbLogger
from utils.dataset import RefDataset
from engine.engine import train, validate
from model import build_segmenter
from utils.misc import (init_random_seed, set_random_seed, setup_logger, worker_init_fn)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config', default='path to xxx.yaml', type=str, help='config file')
    parser.add_argument('--tsg', default=0, type=int, help='add transformer scale gate.')
    parser.add_argument('--jit', default=0, type=int, help='jit mode.')
    parser.add_argument('--early_stop', default=10, type=int, help='set early stop epoch')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER, help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.__setattr__('tsg', args.tsg)
    cfg.__setattr__('jit', args.jit)
    cfg.__setattr__('early_stop', args.early_stop)
    return cfg


@logger.catch
def main():
    args = get_parser()

    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size

    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ))


def main_worker(gpu, args):
    args.output_dir = os.path.join(args.output_folder, args.exp_name)

    # local rank & global rank
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    # logger
    setup_logger(
        args.output_dir,
        distributed_rank=args.gpu,
        filename="train.log",
        mode="a"
    )

    # dist init
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )

    # wandb
    if args.rank == 0:
        wlogger = WandbLogger(args)
        wlogger.init_logger(
            project="CRIS",
            mode="online"
        )
    else:
        wlogger = None
    dist.barrier()

    # build model
    model, param_list = build_segmenter(args)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info(model)
    model = nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[args.gpu],
        find_unused_parameters=True
    )

    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(param_list, lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)
    scaler = amp.GradScaler()

    # build dataset
    args.batch_size = int(args.batch_size / args.ngpus_per_node)
    args.batch_size_val = int(args.batch_size_val / args.ngpus_per_node)
    args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    train_data = RefDataset(
        lmdb_dir=args.train_lmdb,
        mask_dir=args.mask_root,
        dataset=args.dataset,
        split=args.train_split,
        mode='train',
        input_size=args.input_size,
        word_length=args.word_len
    )
    val_data = RefDataset(
        lmdb_dir=args.val_lmdb,
        mask_dir=args.mask_root,
        dataset=args.dataset,
        split=args.val_split,
        mode='val',
        input_size=args.input_size,
        word_length=args.word_len
    )

    # build dataloader
    init_fn = partial(
        worker_init_fn,
        num_workers=args.workers,
        rank=args.rank,
        seed=args.manual_seed
    )
    train_sampler = DistributedSampler(train_data, shuffle=True)
    val_sampler = DistributedSampler(val_data, shuffle=False)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=init_fn,
        sampler=train_sampler,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers_val,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False
    )

    best_IoU = 0.0
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.rank}
            checkpoint = torch.load(args.resume, map_location=map_location)
            args.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["iou"]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            raise ValueError(f"=> resume failed! no checkpoint found at '{args.resume}'. Please check args.resume again!")

    # start training
    early_epoch = args.early_stop
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        train_sampler.set_epoch(epoch_log)

        # train
        train(train_loader, model, optimizer, scheduler, scaler, epoch_log, args, wlogger)

        # evaluation
        iou, prec_dict, dice_coef = validate(val_loader, model, epoch_log, args)

        if dist.get_rank() == 0:
            # loggin
            val_log = dict({
                'eval/iou': iou,
                'eval/dice_coef': dice_coef,
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
                model_name = f"best_model_sg.pth" if args.tsg else f"best_model_base.pth"
                model_path = os.path.join(args.output_dir, model_name)
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

        # update lr
        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    time.sleep(1)
    if dist.get_rank() == 0:
        wlogger.finish()

    logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)
