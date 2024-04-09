import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from loss import CELoss, FocalLoss, DiceLoss


from .layers import FPN, Projector, TransformerDecoder

class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        
        # Multi-Modal fusion
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        
        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
        loss_type = cfg.loss_type
        if loss_type == 'focal':
            self.loss_func = FocalLoss(cfg)
        elif loss_type == 'dice':
            self.loss_func = DiceLoss(cfg)
        else:
            self.loss_func = CELoss(cfg)
    
    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: x1, x2, x3, x4
        # word: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(img)
        word, state = self.backbone.encode_text(word)
         
        # fusion: b, 512, 26, 26 
        fusion = self.neck(vis, state)
        b, _, h, w = fusion.size()
        out = self.decoder(fusion, word, pad_mask)
        out = out.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        # pred: b, 1, 104, 104
        pred = self.proj(out, state)

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
            loss = self.loss_func(pred, mask) 
            return pred.detach(), mask, loss
        else:
            return pred.detach()
