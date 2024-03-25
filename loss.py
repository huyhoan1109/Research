import torch 
import torch.nn as nn
import torch.nn.functional as F

# Have logits
class CELoss(nn.Module):
    def __init__(self, cfg, reduction='mean'):
        super(CELoss, self).__init__()
        self.cfg = cfg
        self.reduction = reduction
    def forward(self, input, target):
        input = F.sigmoid(input)
        # flatten
        input = input.view(-1).float()
        target = target.view(-1).float()
        # loss
        loss = F.binary_cross_entropy(input, target, reduction=self.reduction)
        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, cfg, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.cfg = cfg
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.reduction = reduction
    def forward(self, input, target):
        input = F.sigmoid(input)
        # flatten
        input = input.view(-1).float()
        target = target.view(-1).float()
        # loss
        bce_loss = F.binary_cross_entropy(input, target, reduction='mean')
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss

class DiceLoss(nn.Module):
    def __init__(self, cfg, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = cfg.smooth
        self.reduction = reduction
    def forward(self, input, target):
        input = F.sigmoid(input)
        # flatten
        input = input.view(-1).float()
        target = target.view(-1).float()
        # loss
        inter = (input * target).sum()
        union = input.sum() + input.sum()                      
        loss = 1 - (2.*inter + self.smooth)/(union + self.smooth)
        return loss
    
    