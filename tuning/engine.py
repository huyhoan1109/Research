import os
import torch
from tqdm import tqdm 
from utils.misc import AverageMeter, ProgressMeter 
from tuning.loss import clip_loss
from torch.utils.data import RandomSampler


def get_sampler(dataset, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator)
    return sampler

def train_batch(cfg, epoch, model, train_loader, optimizer, logger):
    model.train()
    loss_meter = AverageMeter('Train loss')
    progress_meter = ProgressMeter(
        len(train_loader),
        [loss_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, cfg['epochs'])
    )
    for i, batch in enumerate(train_loader):
        count = batch["image"].size(0)
        image, text = batch['image'].cuda(), batch['word'].cuda()
        loss = clip_loss(model, image, text)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), count)
        if (i + 1) % cfg['print_freq'] == 0:
            progress_meter.display(i + 1)
            logger.log(
                {
                    'training/loss': loss_meter.val,
                    'training/step': epoch * len(train_loader) + (i + 1)
                },
                commit=True
            )

def validate(model, valid_loader):
    model.eval()
    loss_meter = AverageMeter('Eval loss')
    for i, batch in enumerate(valid_loader):
        count = batch["image"].size(0)
        image, text = batch['image'].cuda(), batch['word'].cuda()
        loss = clip_loss(model, image, text)
        loss_meter.update(loss.item(), count)
    print(f"Valid loss: {loss_meter.avg}")
    return loss_meter.avg

def load_checkpoint(cfg, model, optimizer, lr_scheduler):
    checkpoint = torch.load(cfg['resume'])
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
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

def train_model(cfg, model, loaders, optimizer, lr_scheduler, logger):
    if cfg['resume']:
        start_epoch, best_loss = load_checkpoint(cfg, model, optimizer, lr_scheduler)
    else:
        start_epoch = 0
        best_loss = float('inf')
    for epoch in range(start_epoch, cfg['epochs']):
        epoch_log = epoch + 1
        train_batch(cfg, epoch_log, model, loaders['train'], optimizer, logger)
        cur_loss = validate(model, loaders['valid'])
        logger.log(
            {
                'eval/loss': cur_loss,
                'eval/step': epoch_log
            },
            commit=True
        )
        best_loss = cur_loss if cur_loss <= best_loss else best_loss
        losses = {
            'cur': cur_loss,
            'best': best_loss
        }
        save_checkpoint(cfg, epoch_log, losses, model, optimizer, lr_scheduler)
        if best_loss == cur_loss:
            save_checkpoint(cfg, epoch_log, losses, model, optimizer, lr_scheduler, best=True)
        lr_scheduler.step(epoch_log)