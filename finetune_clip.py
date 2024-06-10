import yaml
import wandb
import torch
import argparse
from endoscopy.dataset import EndosDataset
from tuning.clip import build_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tuning.engine import train_model, get_sampler
from dotenv import dotenv_values

ENV_VAR = dotenv_values(".env")

def load_yaml(path):
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError:
            print(f'Error when reading {path}')

def load_config(args, yaml_cfg):
    for key in yaml_cfg.keys():
        if yaml_cfg[key] == None and args.__getattribute__(key) != None:
            yaml_cfg[key] = args.__getattribute__(key)
    args_keys = args._get_args()
    for key in args_keys:
        if key in yaml_cfg.keys():
            continue
        else:
            yaml_cfg[key] = args.__getattribute__(key)
    return yaml_cfg

def get_args():
    parser = argparse.ArgumentParser(description='Finetuning CLIP')
    parser.add_argument('--config', default='path to xxx.yaml', type=str, help='config file')
    parser.add_argument('--clip_pretrain', default=None, type=str, help='load pretrained weight')
    parser.add_argument('--resume', nargs='?', help='load resume weight')
    parser.add_argument('--prefix_name', default=None, type=str, help='save weight prefix name')
    parser.add_argument('--run_id', nargs='?', help='log run id')
    parser.add_argument('--continue_training', nargs='?', help='continue logging')
    parser.add_argument('--early_stop', default=30, type=int, help='early stop')
    args = parser.parse_args()
    yaml_cfg = load_yaml(args.config)
    cfg = load_config(args, yaml_cfg)
    return cfg

def build_clip(cfg):
    weight = torch.jit.load(cfg['clip_pretrain'])
    return build_model(weight.state_dict()).float().cuda()

def init_logger(args):
    wandb.login(key=ENV_VAR['API_KEY'])
    wlogger = wandb.init(
        config=args,
        project='Finetuning CLIP for Endo',
        id=args['run_id'],
        name=args['prefix_name'],
        tags=args['clip_pretrain'],
        resume=args['continue_training']
    )
    wlogger.define_metric('training/step')
    wlogger.define_metric('eval/step')
    wlogger.define_metric(
        'training/loss', step_metric='training/step'
    )
    wlogger.define_metric(
        'eval/loss', step_metric='eval/step'
    )
    return wlogger

def finish_logger():
    wandb.finish()

if __name__ == '__main__':
    args = get_args()
    model = build_clip(args)
    logger = init_logger(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['base_lr'], weight_decay=args['weight_decay'])

    train_data = EndosDataset(
        input_size=args['input_size'],
        word_length=args['word_len'],
        split='train',
        step='pretrain'
    )

    valid_data = EndosDataset(
        input_size=args['input_size'],
        word_length=args['word_len'],
        split='test',
        step='pretrain'
    )

    train_sampler = get_sampler(train_data, args['seed'])
    valid_sampler = get_sampler(valid_data, args['seed'])

    train_loader = DataLoader(
        train_data,
        batch_size=args['train_batch'],
        num_workers=args['train_worker'],
        pin_memory=True,
        sampler=train_sampler,
        shuffle=False,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=args['valid_batch'],
        num_workers=args['valid_worker'],
        pin_memory=True,
        sampler=valid_sampler,
        shuffle=False,
        drop_last=False
    )
    
    loaders = {
        'train': train_loader,
        'valid': valid_loader
    }

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max = args['train_worker'] * len(train_loader) * args['epochs'],
        eta_min = args['base_lr'] * args['lr_decay']
    )

    train_model(args, model, loaders, optimizer, scheduler, logger)
    finish_logger()
