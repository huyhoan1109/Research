import torch
import torch.nn as nn
from model.backbone.clip import build_model, ModifiedResNet

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), 
        nn.ReLU(True)
    )

def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias),
        nn.BatchNorm1d(out_dim), nn.ReLU(True)
    )

class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(
            in_channels + 2, 
            out_channels, 
            kernel_size,
            padding, 
            stride
        )

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        weight = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        self.clip = build_model(weight.state_dict(), cfg.word_len).float()
        self.multi_scale = isinstance(self.clip, ModifiedResNet)
        if not self.multi_scale:
            self.scale1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU(True)
            )
            self.scale2 = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward_visual(self, image):
        if self.multi_scale:
            return self.clip.encode_image(image) 
        else:
            x2, x3, x4 = self.clip.encode_image(image)
            x2 = self.scale1(x2)
            x3 = self.scale2(x3)
            return x2, x3, x4
    def forward_text(self, text):
        return self.clip.encode_text(text)

def build_backbone(args):
    return Backbone(args)
        