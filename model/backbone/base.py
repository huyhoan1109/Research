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
            
            num_layers = self.clip.visual.transformer.layers
            
            self.vis_channel = 768 

            assert num_layers >= 3 
            self.layer_indexes = [num_layers // 3, num_layers // 2 + 1]
            
            self.layers = []

            for l in self.layer_indexes:
                self.clip.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))
        
            self.scale1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                conv_layer(self.vis_channel, self.vis_channel, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                conv_layer(self.vis_channel, self.vis_channel, 3, padding=1),
            )
            
            self.scale2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                conv_layer(self.vis_channel, self.vis_channel, 3, padding=1),
            )
    
    def forward_visual(self, image):
        if self.multi_scale:
            return self.clip.encode_image(image) 
        else:
            self.layers = []
            x4 = self.clip.encode_image(image)
            batch, grid = x4.size(0), x4.size(-1)
            x2 = self.layers[0][1:, :, :]
            x3 = self.layers[1][1:, :, :]
            x2 = x2.permute(1, 2, 0).reshape(batch, self.vis_channel, grid, grid)
            x3 = x3.permute(1, 2, 0).reshape(batch, self.vis_channel, grid, grid)
            x2 = self.scale1(x2)
            x3 = self.scale2(x3)
            return x2, x3, x4
    
    def forward_text(self, text):
        return self.clip.encode_text(text)

def build_backbone(args):
    return Backbone(args)
        