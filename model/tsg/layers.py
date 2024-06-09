import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.base import conv_layer, linear_layer, CoordConv

class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_stages,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                num_stages=num_stages,
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ffn,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.nr = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis_feats, txt, pad_mask):
        '''
            vis_feats = fusion, (vis1, vis2, vis3)
            vis: b, 512, h, w
            txt: b, L, 512
            pad_mask: b, L
        '''
        f_vis = vis_feats[0]
        r_feats = vis_feats[1]
        B, C, H, W = f_vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        f_vis = f_vis.reshape(B, C, -1).permute(2, 0, 1) # B, 512, HxW => HxW, B, 512
        txt = txt.permute(1, 0, 2)  # B, L, C => L, B, C  
        # forward
        output = f_vis
        intermediate = []
        for decoder in self.decoder_layers:
            output = decoder(output, txt, vis_pos, txt_pos, pad_mask, r_feats)
            if self.return_intermediate:
                # HxW, b, 512 -> b, 512, HxW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HxW, b, 512 -> b, 512, HxW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 num_stages,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.vis_norm = nn.LayerNorm(d_model)
        self.vis_txt_norm = nn.LayerNorm(d_model)
        
        # Attention Layer
        self.vis_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.vis_txt_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=d_model, vdim=d_model)
        
        # Scale gate
        self.scale_gate = ScaleGate(d_model, num_stages)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(True), 
            nn.Dropout(dropout),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, d_model)
        )
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask, r_feats):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L,
            r_feats: 26*26, b, 512 * 3
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2, _ = self.vis_attn(
            query=q, 
            key=k, 
            value=vis2,
        ) 
        # (26x26, B, 512)
        vis2 = self.vis_norm(vis2)
        vis = vis + self.dropout1(vis2)

        # Cross-Attention
        vis2 = self.norm2(vis)
        # (26x26, B, 512), (B, nhead, 26x26, 17)
        vis2, _ = self.vis_txt_attn(
            query=self.with_pos_embed(vis2, vis_pos),
            key=self.with_pos_embed(txt, txt_pos),
            value=txt,
            key_padding_mask=pad_mask,
        )
        vis2 = self.vis_txt_norm(vis2 + self.scale_gate(vis2, r_feats))
        # [676, B, 512]
        vis = vis + self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis

class FPN(nn.Module):
    def __init__(self,
                 state_dim,
                 in_channels,
                 out_channels):
        super(FPN, self).__init__()
        # text projection
        self.txt_proj = linear_layer(state_dim, out_channels[2])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU(True)
        )
        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1], out_channels[1], 1, 0)
        
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1], out_channels[1], 1, 0)
        
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1)
        )

    def forward(self, imgs, state, use_transformer=False):
        # text projection: b, 1024 -> b, 1024
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        
        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = imgs

        # first projection
        f5 = self.f1_v_proj(v5)
        f4 = self.f2_v_proj(v4)
        f3 = self.f3_v_proj(v3)

        # fusion 1
        f5 = self.norm_layer(f5 * state)

        # check transformer
        if use_transformer:
            # fusion 2
            f4 = self.f2_cat(torch.cat([f4, f5], dim=1))

            # fusion 3: b, 256, 26, 26
            f3 = self.f3_cat(torch.cat([f3, f4], dim=1))

            # middle projection
            fq5 = self.f4_proj5(f5)
            fq4 = self.f4_proj4(f4)
            fq3 = self.f4_proj3(f3)
        else:
            # fusion 2: b, 512, 26, 26
            f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
            f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
            
            # fusion 3: b, 256, 26, 26
            f3 = F.avg_pool2d(f3, 2, 2)
            f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
            
            # middle projection
            fq5 = self.f4_proj5(f5)
            fq4 = self.f4_proj4(f4)
            fq3 = self.f4_proj3(f3)
            
            # interpolate
            fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        
        # raw fusion
        r_fusion = torch.cat([fq3, fq4, fq5], dim=1)
        
        # fusion query
        fq = self.aggr(r_fusion)
        fq = self.coordconv(fq)
        
        # fusion query / raw fusion: b, 512, 26, 26 / b, 1536, 26, 26
        return fq, r_fusion

class Projector(nn.Module):
    def __init__(self, num_classes=1, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.inter_dim = in_dim // 2
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, self.inter_dim * self.num_classes , 1)
        )
        # textual projector
        text_out_dim = self.inter_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, text_out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        # B, C * num_classes, 104, 104
        B, _, H, W = x.size()
        x = x.transpose(1, 0)
        x = x.reshape(self.num_classes, B * self.inter_dim, H, W)
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, self.inter_dim, self.kernel_size, self.kernel_size)
        # combine image and text
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       bias=bias,
                       groups=weight.size(0))
        out = out.transpose(1, 0)
        return out

class ScaleGate(nn.Module):
    def __init__(self, d_model, num_stages):
        super().__init__()
        self.d_model = d_model
        self.num_stages = num_stages
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_stages * d_model),
        )
        self.aggr = conv_layer(num_stages * d_model, d_model, 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(d_model, d_model, 3, 1),
            conv_layer(d_model, d_model, 3, 1)
        )

    def forward(self, x, r_feats):
        _, B, C = x.size() # HxW, B, C
        r_shape = r_feats.size()
        x = self.layers(x)
        x = x.permute(1, 2, 0).reshape(r_shape)
        product = F.softmax(x, dim=1) * r_feats
        output = self.coordconv(self.aggr(product))
        return output.reshape(B, C, -1).permute(2, 0, 1)



