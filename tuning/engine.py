import os
import torch
from tqdm import tqdm 
from utils.misc import AverageMeter, ProgressMeter 
from tuning.metric import calculate_metric
from torch.utils.data import RandomSampler

def get_sampler(dataset, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator)
    return sampler

def train_batch(cfg, epoch, model, train_loader, optimizer, scheduler, logger):
    model.train()
    loss_meter = AverageMeter('Train loss')
    progress_meter = ProgressMeter(
        len(train_loader),
        [loss_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, cfg['epochs'])
    )
    for i, batch in enumerate(train_loader):
        count = batch["image"].size(0)
        # image = batch['image']
        # word = batch['word']
        image = batch['image'].cuda()
        word = batch['word'].cuda()
        result = calculate_metric(model, image, word)
        optimizer.zero_grad()
        result['loss'].backward()
        optimizer.step()
        loss_meter.update(result['loss'].item(), count)
        if (i + 1) % cfg['print_freq'] == 0:
            progress_meter.display(i + 1)
            logger.log(
                {
                    'training/lr': scheduler.get_last_lr()[-1],
                    'training/loss': loss_meter.val,
                    'training/step': epoch * len(train_loader) + (i + 1)
                },
                commit=True
            )
    scheduler.step()

def validate(model, valid_loader):
    model.eval()
    loss_meter = AverageMeter('Eval loss')
    tbar = tqdm(valid_loader, desc='Eval:', ncols=100)
    for i, batch in enumerate(tbar):
        count = batch["image"].size(0)
        # image = batch['image']
        # word = batch['word']
        image = batch['image'].cuda()
        word = batch['word'].cuda()
        result = calculate_metric(model, image, word)
        loss_meter.update(result['loss'].item(), count)
    print(f"Valid loss: {loss_meter.avg}")
    return loss_meter.avg

def load_checkpoint(cfg, model, optimizer, scheduler):
    checkpoint = torch.load(cfg['resume'])
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint['epoch'], checkpoint['loss']

def save_checkpoint(cfg, epoch_log, loss, model, optimizer, scheduler):
    model_path = os.path.join(cfg['output_dir'], f"{cfg['prefix_name']}_{cfg['step']}_best_clip.pth")
    torch.save(
        {
            'epoch': epoch_log,
            'loss': loss,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, 
        model_path
    )

def train_model(cfg, model, loaders, optimizer, scheduler, logger): 
    if cfg['resume']:
        start_epoch, best_loss = load_checkpoint(cfg, model, optimizer, scheduler)
    else:
        start_epoch = 0
        best_loss = float('inf')
    early_epoch = cfg['early_stop']
    for epoch in range(start_epoch, cfg['epochs']):
        epoch_log = epoch + 1
        train_batch(cfg, epoch_log, model, loaders['train'], optimizer, scheduler, logger)
        cur_loss = validate(model, loaders['valid'])
        logger.log(
            {
                'eval/loss': cur_loss,
                'eval/step': epoch_log
            },
            commit=True
        )
        best_loss = cur_loss if cur_loss <= best_loss else best_loss
        if best_loss == cur_loss and early_epoch > 0:
            save_checkpoint(cfg, epoch_log, best_loss, model, optimizer, scheduler)
            early_epoch = cfg['early_stop']
        else:
            early_epoch -= 1
            if early_epoch == 0:
                break