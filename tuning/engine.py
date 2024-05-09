import os
import torch
from tqdm import tqdm 
from utils.misc import AverageMeter 
from tuning.loss import clip_loss
from torch.utils.data import RandomSampler


def get_sampler(dataset, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator)
    return sampler

def train_batch(model, loaders, optimizer):
    model.train()
    loss_meter = AverageMeter('Train loss')
    train_tqdm = tqdm(loaders['train'], total=len(loaders['train']))
    for batch in train_tqdm:
        count = batch["image"].size(0)
        image, text = batch['image'], batch['word']
        loss = clip_loss(model, image, text)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), count)
        train_tqdm.set_postfix(train_loss=loss_meter.avg)    

def validate(model, loaders):
    model.eval()
    loss_meter = AverageMeter('Eval loss')
    valid_tqdm = tqdm(loaders['valid'], total=len(loaders['valid']))
    for batch in valid_tqdm:
        count = batch["image"].size(0)
        image, text = batch['image'], batch['word']
        loss = clip_loss(model, image, text)
        loss_meter.update(loss.item(), count)
        valid_tqdm.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter.avg

def load_checkpoint(cfg, model, optimizer, lr_scheduler):
    checkpoint = torch.load(cfg['resume'])
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'], strict=True)
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'], strict=True)
    return checkpoint['epoch'], checkpoint['best_loss']

def save_checkpoint(cfg, epoch_log, losses, model, optimizer, lr_scheduler, best=False):
    model_name = "best_model.pth" if best else "last_model.pth"
    model_path = os.path.join(cfg['output_dir'], f"{cfg['prefix_name']}_{model_name}")
    torch.save(
        {
            'epoch': epoch_log,
            'cur_loss': losses['cur'],
            'best_loss': losses['best'],
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict()
        }, 
        model_path
    )

def train_model(cfg, model, loaders, optimizer, lr_scheduler):
    if cfg['resume']:
        start_epoch, best_loss = load_checkpoint(cfg, model, optimizer, lr_scheduler)
    else:
        start_epoch = 0
        best_loss = float('inf')
    for epoch in range(start_epoch, cfg['epochs']):
        epoch_log = epoch + 1
        train_batch(model, loaders, optimizer)
        cur_loss = validate(model, loaders)
        best_loss = cur_loss if cur_loss <= best_loss else best_loss
        losses = {
            'cur': cur_loss,
            'best': best_loss
        }
        save_checkpoint(cfg, epoch_log, losses, model, optimizer, lr_scheduler)
        if best_loss == cur_loss:
            save_checkpoint(cfg, epoch_log, losses, model, optimizer, lr_scheduler, best=True)
        lr_scheduler.step(epoch_log)

