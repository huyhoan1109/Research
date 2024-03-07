import torch 
import torch.nn as nn
import torch.nn.functional as F

# Have logits
class CELoss(nn.Module):
    def __init__(self, cfg):
        super(CELoss, self).__init__()
        self.cfg = cfg
    def forward(self, input, target):
        loss = F.binary_cross_entropy_with_logits(input, target)
        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, cfg):
        super(FocalLoss, self).__init__()
        self.cfg = cfg
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy_with_logits(input, target)
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss

class DiceLoss(nn.Module):
    def __init__(self, cfg):
        super(DiceLoss, self).__init__()
        self.smooth = cfg.smooth
    def forward(self, input, target):
        intersection = torch.sum(input * target, dim=2)  # (N, C)
        union = torch.sum(input, dim=2) + torch.sum(target, dim=2)  # (N, C)
        loss = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)
        return loss