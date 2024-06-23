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
        image = data['image'].cuda(non_blocking=True)
        text = data['word'].cuda(non_blocking=True)
        target = data['mask'].cuda(non_blocking=True)
        
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
        imgs = data['image'].cuda(non_blocking=True)
        texts = data['word'].cuda(non_blocking=True)
        masks = data['mask'].cuda(non_blocking=True)
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
        for pred, mask in zip(preds, masks):
            pred = torch.tensor(pred > 0.35)
            inter = torch.logical_and(pred, mask)
            union = torch.logical_or(pred, mask)
            iou = (torch.sum(inter) + 1e-6) / (torch.sum(union) + 1e-6)
            dice = (2 * torch.sum(inter) + 1e-6) / (torch.sum(pred + mask) + 1e-6)
            iou_list.append(iou)
            dice_list.append(dice)

        # inter_list.append(torch.sum(inters))
        # union_list.append(torch.sum(unions))
        # combo_list.append(torch.sum(preds + masks))
    
    iou_list = torch.stack(iou_list).to(imgs.device)
    dice_list = torch.stack(dice_list).to(imgs.device)
    
    # inter_list = torch.tensor(inter_list).to(imgs.device)
    # union_list = torch.tensor(union_list).to(imgs.device)
    # combo_list = torch.tensor(combo_list).to(imgs.device)
    
    macro_iou = iou_list.mean()
    macro_dice = dice_list.mean()

    # micro_iou = (torch.sum(inter_list) + 1e-6) / (torch.sum(union_list) + 1e-6)
    # micro_dice = (2 * torch.sum(inter_list) + 1e-6) / (torch.sum(combo_list) + 1e-6)
    
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
    # head = 'Evaluation: Epoch=[{}/{}]  Macro: [IoU={:.2f} Dice={:.2f}] Micro: [IoU={:.2f} Dice={:.2f}]'.format(epoch, args.epochs, 100. * macro_iou.item(), 100. * macro_dice.item(), 100. * micro_iou.item(), 100. * micro_dice.item())
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f} Dice={:.2f}'.format(epoch, args.epochs, 100. * macro_iou.item(), 100. * macro_dice.item())
    logger.info(head + temp)
    return macro_iou.item(), macro_dice.item(), prec

@torch.no_grad()
def inference(test_loader, model, args):
    iou_list = []
    dice_list = []
    inter_list = []
    union_list = []
    combo_list = []
    tbar = tqdm(test_loader, desc='Inference', ncols=100)
    model.eval()
    for idx, data in enumerate(tbar):
        imgs = data['image'].cuda()
        masks = data['mask'].cuda()
        words = data['word'].cuda()
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
        for pred, mask, img_id, prompt, label in zip(preds, masks, img_ids, prompts, labels):
            iou_stack, dice_stack, inter_stack, union_stack = single_inference(
                pred, 
                mask, 
                img_id, 
                prompt,
                label, 
                args
            )
            iou_list.append(iou_stack)
            dice_list.append(dice_stack)
            inter_list.append(inter_stack)
            union_list.append(union_stack)
            combo_list.append(torch.sum(pred + mask))
    logger.info('=> Metric Calculation <=')
    
    iou_list = torch.stack(iou_list).to(imgs.device)
    dice_list = torch.stack(dice_list).to(imgs.device)
    
    inter_list = torch.tensor(inter_list).to(imgs.device)
    union_list = torch.tensor(union_list).to(imgs.device)
    combo_list = torch.tensor(combo_list).to(imgs.device)
    
    micro_iou = (torch.sum(inter_list) + 1e-6) / (torch.sum(union_list) + 1e-6)
    micro_dice = (2 * torch.sum(inter_list) + 1e-6) / (torch.sum(combo_list) + 1e-6)
    macro_iou = iou_list.mean()
    macro_dice = dice_list.mean()
    logger.info('Macro: IoU={}, Dice={}; Micro: IoU={}, Dice={}'.format(macro_iou, macro_dice, micro_iou, micro_dice))

def single_inference(
        pred, 
        mask, 
        img_id, 
        sentence, 
        label, 
        args
    ):
    output_shape = pred.shape
    iou_stack = []
    dice_stack = []
    inter_stack = []
    union_stack = []
    if args.visualize:
        os.makedirs(os.path.join(args.vis_dir, img_id), exist_ok=True)
    for i in range(output_shape[0]):
        pred_i = torch.tensor(pred[i] > 0.35)
        inter_i = torch.logical_and(pred_i, mask[i])
        union_i = torch.logical_or(pred_i, mask[i])
        iou_i = (torch.sum(inter_i) + 1e-6) / (torch.sum(union_i) + 1e-6)
        dice_i = (2 * torch.sum(inter_i) + 1e-6) / (torch.sum(pred_i + mask[i]) + 1e-6)
        inter_stack.append(torch.sum(inter_i))
        union_stack.append(torch.sum(union_i))
        iou_stack.append(iou_i) 
        dice_stack.append(dice_i)
        if args.visualize:
            if output_shape[0] > 1:
                mask_name = '{}/infer-{}-{}-mask.png'.format(img_id, i, label)
                pred_name = '{}/infer-{}-{}-iou={:.2f}-{}.png'.format(img_id, i, label, iou_i * 100, sentence)
            else:
                mask_name = '{}/infer-{}-mask.png'.format(img_id, label)
                pred_name = '{}/infer-{}-iou={:.2f}-{}.png'.format(img_id, label, iou_i * 100, sentence)            
            mask_i = np.array(mask[i].cpu() * 255, dtype=np.uint8)
            pred_i = np.array(pred_i.cpu() * 255, dtype=np.uint8)
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name), img=mask_i)
            cv2.imwrite(filename=os.path.join(args.vis_dir, pred_name), img=pred_i)
    return torch.tensor(iou_stack), torch.tensor(dice_stack), torch.tensor(inter_stack), torch.tensor(union_stack)