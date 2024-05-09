import yaml
import torch
import argparse
from endoscopy.dataset import EndosDataset
from tuning.clip import build_model
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tuning.engine import train_model, get_sampler

def load_yaml(path):
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def load_config(args, yaml_cfg):
    for config in yaml_cfg.keys():
        if yaml_cfg[config] == None and args.__getattribute__(config) != None:
            yaml_cfg[config] = args.__getattribute__(config)
    return yaml_cfg

def get_args():
    parser = argparse.ArgumentParser(description='Finetuning CLIP')
    parser.add_argument('--config', default='path to xxx.yaml', type=str, help='config file')
    parser.add_argument('--clip_pretrain', default=None, type=str, help='load pretrained weight')
    parser.add_argument('--resume', nargs='?', help='load resume weight')
    parser.add_argument('--prefix_name', default=None, type=str, help='save weight prefix name')
    args = parser.parse_args()
    yaml_cfg = load_yaml(args.config)
    cfg = load_config(args, yaml_cfg)
    return cfg

def build_clip(cfg):
    weight = torch.jit.load(cfg['clip_pretrain'])
    return build_model(weight.state_dict(), cfg['word_len']).float().cuda()

if __name__ == '__main__':
    args = get_args()
    model = build_clip(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['base_lr'], weight_decay=args['weight_decay'])
    scheduler = MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['lr_decay'])
    
    train_data = EndosDataset(
        input_size=args['input_size'],
        word_length=args['word_len'],
        split='train'
    )

    valid_data = EndosDataset(
        input_size=args['input_size'],
        word_length=args['word_len'],
        split='test'
    )

    train_sampler = get_sampler(train_data, args['seed'])
    valid_sampler = get_sampler(valid_data, args['seed'])

    train_loader = DataLoader(
        train_data,
        batch_size=args['train_batch'],
        num_workers=args['train_worker'],
        pin_memory=True,
        sampler=train_sampler
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=args['valid_batch'],
        num_workers=args['valid_worker'],
        pin_memory=True,
        sampler=valid_sampler
    )
    
    loaders = {
        'train': train_loader,
        'valid': valid_loader
    }
    train_model(args, model, loaders, optimizer, scheduler)
