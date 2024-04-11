import torch
import torch.nn as nn
from model.backbone.clip import build_model, VisionTransformer, AttentionPool2d
import torch.nn.functional as F

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), 
        nn.ReLU(True)
    )

def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias),
        nn.BatchNorm1d(out_dim), 
        nn.ReLU(True)
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

        self.use_transformer = isinstance(self.clip.visual, VisionTransformer)
            
        if self.use_transformer:
            final_channel = self.clip.visual.output_dim
            out_channels = cfg.fpn_in
            self.clip_resolution = 256 
            
            # num_layers = self.clip.visual.transformer.layers
            # assert num_layers >= 3 
            # self.vis_channel = self.clip.visual.width 
            # self.layer_indexes = [num_layers // 4, num_layers // 2 + 1]
            # self.layers = []
            # for l in self.layer_indexes:
            #     self.clip.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))

            self.proj1 = nn.Sequential(
                conv_layer(final_channel, out_channels[0], 3, 1),
                nn.Upsample(scale_factor=4, mode='bilinear'),
                conv_layer(out_channels[0], out_channels[0], 3, 1)
            )
            
            self.proj2 = nn.Sequential(
                nn.AvgPool2d(2),
                conv_layer(out_channels[0], out_channels[1], 3, 1),
                nn.BatchNorm2d(out_channels[1])
            )

            self.proj3 = AttentionPool2d(self.clip_resolution // 16, out_channels[2] * 2, 32, out_channels[2])
            
    
    def forward_visual(self, image):
        if self.use_transformer:
            # self.layers = [] 
            if image.size(-1) != self.clip_resolution:
                image = F.interpolate(image, self.clip_resolution, mode='bicubic', align_corners=False)
            # x3 = self.clip.encode_image(image)
            # batch, grid = x3.size(0), x3.size(-1)
            # x1 = self.layers[0][1:, :, :]
            # x2 = self.layers[1][1:, :, :]
            # x1 = x1.permute(1, 2, 0).reshape(batch, self.vis_channel, grid, grid)
            # x2 = x2.permute(1, 2, 0).reshape(batch, self.vis_channel, grid, grid)
            # x1 = self.proj1(x1)
            # x2 = self.proj2(x2)
            # x3 = self.proj3(x3)
            x = self.clip.encode_image(image)
            x1 = self.proj1(x)
            x2 = self.proj2(x1)
            x3 = self.proj3(x2)
            return x1, x2, x3
        else:
            return self.clip.encode_image(image)
    
    def forward_text(self, text):
        return self.clip.encode_text(text)

def build_backbone(args):
    return Backbone(args)
        