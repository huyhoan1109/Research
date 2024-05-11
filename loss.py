import torch 
import torch.nn as nn
import torch.nn.functional as F

def build_loss(name):
    if name is None:
        name = 'ce'
    return {
        'ce': CELoss(reduction='mean'),
        'focal': FocalLoss(alpha=0.25, gamma=2, reduction='mean'),
        'dice': DiceLoss(smooth=1e-6, reduction='mean'),
        'bce_dice': BCEDiceLoss(alpha=0.5, beta=0.5, smooth=1e-6, reduction='mean')
    }[name]

class CELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CELoss, self).__init__()
        self.reduction = reduction
    def forward(self, input, target):
        # flatten
        input = input.view(-1).float()
        target = target.view(-1).float()
        # loss
        loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)
        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, input, target):
        # flatten
        input = input.view(-1).float()
        target = target.view(-1).float()
        # loss
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    def forward(self, input, target):
        # flatten
        input = input.view(-1).float()
        target = target.view(-1).float()
        # loss
        inter = (input * target).sum()
        union = input.sum() + input.sum()                      
        loss = 1 - (2.*inter + self.smooth)/(union + self.smooth)
        return loss
    
class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta, smooth=1e-6, reduction='mean'):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = CELoss(reduction)
        self.dice = DiceLoss(smooth, reduction)
    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)