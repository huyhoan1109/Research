import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from loguru import logger
from utils.misc import (AverageMeter, ProgressMeter, CalculateMetricGPU)
from endoscopy.transform import *

def train(train_loader, model, optimizer, scheduler, scaler, epoch, args, wlogger=None):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    dice_meter = AverageMeter('Dice', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, iou_meter, pr_meter, dice_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs)
    )
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data
        image = data['image'].cuda()
        text = data['word'].cuda()
        target = data['mask'].cuda()

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
        iou, dice, pr50 = CalculateMetricGPU(pred, target, 0.35, 0.5)

        loss_meter.update(loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        dice_meter.update(dice.item(), image.size(0))
        pr_meter.update(pr50.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if wlogger != None:
                wlogger.logging(
                    {
                        "time/batch": batch_time.val,
                        "time/data": data_time.val,
                        "training/lr": lr.val,
                        "training/loss": loss_meter.val,
                        "training/iou": iou_meter.val,
                        "training/prec@50": pr_meter.val,
                        "training/dice": dice_meter.val,
                        "training/step": epoch * len(train_loader) + (i + 1)
                    }
                )

@torch.no_grad()
def validate(val_loader, model, epoch, args):
    iou_list = []
    dice_list = []
    model.eval()
    for idx, data in enumerate(val_loader):
        # data
        if idx == 5:
            break
        imgs = data['image']
        texts = data['word']
        masks = data['mask']
        # inference
        preds = model(imgs, texts)
        preds = torch.sigmoid(preds)
        if preds.shape[-2:] != masks.shape[-2:]:
            preds = F.interpolate(
                preds,
                size=masks.shape[-2:],
                mode='bicubic',
                align_corners=True
            ).squeeze(1)
        preds = torch.tensor(preds > 0.35)
        inters = torch.logical_and(preds, masks)
        unions = torch.logical_or(preds, masks)
        iou = (torch.sum(inters)+ 1e-6) / (torch.sum(unions) + 1e-6)
        dice = 2 * (torch.sum(inters) + 1e-6) / (torch.sum(preds + masks) + 1e-6)
        iou_list.append(iou)
        dice_list.append(dice)
    
    iou_list = torch.stack(iou_list).to(imgs.device)
    dice_list = torch.stack(dice_list).to(imgs.device)
    if len(iou_list.size()) == 2:
        micro_iou = iou_list.permute(1, 0).mean(1)
        micro_dice = dice_list.permute(1, 0).mean(1)
    else:
        micro_iou = iou_list.mean()
        micro_dice = dice_list.mean()
    macro_iou = iou_list.mean()
    macro_dice = dice_list.mean()
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Prec@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  Macro: [IoU={:.2f} Dice={:.2f}] Micro: [IoU={:.2f} Dice={:.2f}]'.format(epoch, args.epochs, 100. * macro_iou.item(), 100. * macro_dice.item(), 100. * micro_iou.item(), 100. * micro_dice.item())
    logger.info(head + temp)
    return macro_iou.item(), macro_dice.item(), prec

@torch.no_grad()
def inference(test_loader, model, args):
    iou_list = []
    dice_list = []
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    for idx, data in enumerate(tbar):
        if idx == 10:
            break
        imgs = data['image']
        masks = data['mask']
        words = data['word']
        prompts = data['prompt']
        img_ids = data['img_id']
        labels = data['label']  
        preds = model(imgs, words)
        preds = torch.sigmoid(preds)
        if preds.shape[-2:] != masks.shape[-2:]:
            preds = F.interpolate(
                preds,
                size=masks.shape[-2:],
                mode='bicubic',
                align_corners=True
            ).squeeze(1)
        if len(preds.shape) == 3:
            preds = preds.unsqueeze(1)
            masks = masks.unsqueeze(1)
        for pred, mask, img_id, prompt, label in zip(preds, masks, img_ids, prompts, labels):
            iou_stack, dice_stack = single_inference(
                pred, 
                mask, 
                img_id, 
                prompt,
                label, 
                vis_dir=args.vis_dir, 
                visual=args.visualize
            )
            iou_list.append(iou_stack)
            dice_list.append(dice_stack)
    logger.info('=> Metric Calculation <=')
    iou_list = torch.stack(iou_list).to(imgs.device).permute(1, 0)
    dice_list = torch.stack(dice_list).to(imgs.device).permute(1, 0) 
    micro_iou = (iou_list.mean(1).numpy() * 100).round(3)  # Micro iou
    micro_dice = (dice_list.mean(1).numpy() * 100).round(3) # Micro dice
    macro_iou = round(micro_iou.mean(), 3)
    macro_dice = round(micro_dice.mean(), 3)
    # logger.info('Micro: IoU={}, Dice={}'.format(micro_iou, micro_dice))
    logger.info('Macro: IoU={}, Dice={}'.format(macro_iou, macro_dice))

def single_inference(
        pred, 
        mask, 
        img_id, 
        sentence, 
        label, 
        vis_dir, 
        visual=True
    ):
    output_shape = pred.shape
    iou_stack = []
    dice_stack = []
    os.makedirs(os.path.join(vis_dir, img_id), exist_ok=True)
    pred = torch.tensor(pred > 0.35)
    for i in range(output_shape[0]):
        inter_i = torch.logical_and(pred[i], mask[i])
        union_i = torch.logical_or(pred[i], mask[i])
        iou_i = (torch.sum(inter_i) + 1e-6) / (torch.sum(union_i) + 1e-6)
        dice_i = 2 * (torch.sum(inter_i) + 1e-6) / (torch.sum(pred[i] + mask[i]) + 1e-6)
        iou_stack.append(iou_i) 
        dice_stack.append(dice_i)
        if visual:
            if output_shape[0] > 1:
                mask_name = '{}/infer-{}-{}-mask.png'.format(img_id, i, label)
                pred_name = '{}/infer-{}-{}-iou={:.2f}-{}.png'.format(img_id, i, label, iou_i * 100, sentence)
            else:
                mask_name = '{}/infer-{}-mask.png'.format(img_id, label)
                pred_name = '{}/infer-{}-iou={:.2f}-{}.png'.format(img_id, label, iou_i * 100, sentence)            
            mask_i = np.array(mask[i].cpu() * 255, dtype=np.uint8)
            pred_i = np.array(pred[i].cpu() * 255, dtype=np.uint8)
            cv2.imwrite(filename=os.path.join(vis_dir, mask_name), img=mask_i)
            cv2.imwrite(filename=os.path.join(vis_dir, pred_name), img=pred_i)
    return torch.tensor(iou_stack), torch.tensor(dice_stack)