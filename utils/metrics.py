def calculate_metrics(pred, target, dim, smooth=1e-6, ths=0.5):
    pred_b = pred.bool()
    target_b = target.bool()
    inter = (pred_b & target_b).sum(dim=dim)
    union = (pred.bool() | target.bool()).sum(dim=dim)  
    ious = (inter + smooth) / (union + smooth)
    iou = ious.mean()
    prec = (ious > ths).float().mean()
    dice = 2 * (
        ((pred * target).sum(dim=1) + smooth) / 
        ((pred + target).sum(dim=1) + smooth)
    ).mean()
    return {
        'iou': iou,
        'precision': prec,
        'dice': dice
    }