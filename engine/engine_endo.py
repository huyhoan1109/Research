import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, trainMetricGPU)

from endoscopy.transform import *

def train(train_loader, model, optimizer, scheduler, scaler, epoch, args, wlogger=None):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    dice_coef_meter = AverageMeter('Dice Coef', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter, dice_coef_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs)
    )
    model.train()
    time.sleep(2)
    end = time.time()

    # size_list = [320, 352, 384, 416, 448, 480, 512]
    # idx = np.random.choice(len(size_list))
    # new_size = size_list[idx]

    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data
        image = data['image'].cuda(non_blocking=True)
        text = data['word'].cuda(non_blocking=True)
        target = data['mask'].cuda(non_blocking=True).unsqueeze(1)

        # forward
        with amp.autocast():
            pred, target, loss = model(image, text, target)
        
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # metric
        iou, pr50, dice_coef = trainMetricGPU(pred, target, 0.35, 0.5)
        dist.all_reduce(loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr50)
        dist.all_gather(dice_coef)
        
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr50 = pr50 / dist.get_world_size()
        dice_coef = dice_coef / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr50.item(), image.size(0))
        dice_coef_meter.update(dice_coef.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if dist.get_rank() == 0 and wlogger != None:
                wlogger.logging(
                    {
                        "time/batch": batch_time.val,
                        "time/data": data_time.val,
                        "training/lr": lr.val,
                        "training/loss": loss_meter.val,
                        "training/iou": iou_meter.val,
                        "training/prec@50": pr_meter.val,
                        "training/dice_coef": dice_coef_meter.val,
                        "training/step": epoch * len(train_loader) + (i + 1)
                    }
                )

@torch.no_grad()
def validate(val_loader, model, epoch, args):
    iou_list = []
    dice_coef_list = []
    model.eval()
    time.sleep(2)
    for id, data in enumerate(val_loader):
        # data
        imgs = data['image'].cuda(non_blocking=True)
        texts = data['word'].cuda(non_blocking=True)
        target = data['mask'].cuda(non_blocking=True)
        # inference
        preds = model(imgs, texts)
        preds = torch.sigmoid(preds)
        if preds.shape[-2:] != target.shape[-2:]:
            preds = F.interpolate(
                preds,
                size=target.shape[-2:],
                mode='bicubic',
                align_corners=True
            ).squeeze(1)

        preds = torch.tensor(preds > 0.35)
        inters = torch.logical_and(preds, target)
        unions = torch.logical_or(preds, target)
        iou = torch.sum(inters) / (torch.sum(unions) + 1e-6)
        iou_list.append(iou)
        dice_coef = 2 * torch.sum(inters) / torch.sum(preds + target)
        dice_coef_list.append(dice_coef)

    iou_list = torch.stack(iou_list).to(imgs.device)
    iou_list = concat_all_gather(iou_list)
    dice_coef_list = torch.stack(dice_coef_list).to(imgs.device)
    dice_coef_list = concat_all_gather(dice_coef_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    dice_coef = dice_coef_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Prec@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}'.format(epoch, args.epochs, 100. * iou.item())
    logger.info(head + temp)
    return iou.item(), prec, dice_coef.item()

@torch.no_grad()
def inference(test_loader, model, args):
    iou_list = []
    dice_coef_list = []
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    for id, data in enumerate(tbar):
        # data
        img = data['image'].cuda(non_blocking=True)
        target = data['mask'].cuda(non_blocking=True)
        word = data['word'].cuda(non_blocking=True)
        prompts = data['prompt']
        img_ids = data['img_id']
        preds = model(img, word)
        preds = torch.sigmoid(preds)
        if preds.shape[-2:] != target.shape[-2:]:
            preds = F.interpolate(
                preds,
                size=target.shape[-2:],
                mode='bicubic',
                align_corners=True
            ).squeeze(1)
        for pred, mask, sent, img_id in zip(preds, target, prompts, img_ids):
            pred = torch.tensor(pred > 0.35)
            # iou
            inter = torch.logical_and(pred, mask)
            union = torch.logical_or(pred, mask)
            iou = torch.sum(inter) / (torch.sum(union) + 1e-6)
            dice_coef = 2 * torch.sum(inter) / torch.sum(pred + mask)
            iou_list.append(iou)
            dice_coef_list.append(dice_coef)
            if args.visualize:
                # dump image & mask
                mask_name = '{}-mask.png'.format(img_id)
                cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name), img=mask)
                pred = np.array(pred*255, dtype=np.uint8)
                pred_name = '{}-iou={:.2f}-{}.png'.format(img_id, iou * 100, sent)
                cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name), img=pred)
    
    logger.info('=> Metric Calculation <=')
    iou_list = torch.stack(iou_list).to(img.device)
    iou_list = concat_all_gather(iou_list)
    dice_coef_list = torch.stack(dice_coef_list).to(img.device)
    dice_coef_list = concat_all_gather(dice_coef_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    dice_coef = dice_coef_list.mean()
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('IoU={:.2f}'.format(100.*iou.item()))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))

    return iou.item(), prec, dice_coef.item()