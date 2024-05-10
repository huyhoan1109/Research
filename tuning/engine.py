import os
import torch
from tqdm import tqdm 
from utils.misc import AverageMeter, ProgressMeter 
from tuning.loss import clip_loss
from torch.utils.data import RandomSampler


def get_sampler(dataset, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator)
    return sampler

def train_batch(cfg, model, train_loader, optimizer, meters):
    model.train()
    for i, batch in enumerate(train_loader):
        count = batch["image"].size(0)
        image, text = batch['image'].cuda(), batch['word'].cuda()
        loss = clip_loss(model, image, text)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        meters['train_loss'].update(loss.item(), count)
        if (i + 1) % cfg['print_freq'] == 0:
            meters['progress'].display(i + 1)


def validate(model, valid_loader, meters):
    model.eval()
    for i, batch in enumerate(valid_loader):
        count = batch["image"].size(0)
        image, text = batch['image'].cuda(), batch['word'].cuda()
        loss = clip_loss(model, image, text)
        meters['valid_loss'].update(loss.item(), count)
    return meters['valid_loss'].avg

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
    train_loss_meter = AverageMeter('Train loss')
    valid_loss_meter = AverageMeter('Eval loss')
    progress_meter = ProgressMeter(
        len(loaders['train']),
        [train_loss_meter, valid_loss_meter],
        prefix="Training: Epoch=[{}/{}] ".format(start_epoch, cfg['epochs'])
    )
    meters = {
        'train_loss': train_loss_meter,
        'valid_loss': valid_loss_meter,
        'progress': progress_meter
    }
    for epoch in range(start_epoch, cfg['epochs']):
        epoch_log = epoch + 1
        train_batch(cfg, model, loaders, optimizer, meters)
        cur_loss = validate(model, loaders, meters)
        best_loss = cur_loss if cur_loss <= best_loss else best_loss
        losses = {
            'cur': cur_loss,
            'best': best_loss
        }
        save_checkpoint(cfg, epoch_log, losses, model, optimizer, lr_scheduler)
        if best_loss == cur_loss:
            save_checkpoint(cfg, epoch_log, losses, model, optimizer, lr_scheduler, best=True)
        lr_scheduler.step(epoch_log)

