def calculate_metrics(pred, target, dim, smooth=1e-6, ths=0.5):
    pred_b = pred.bool()
    target_b = target.bool()
    tp = (pred_b & target_b).sum(dim=dim)
    fn = ((pred_b == False) & (target_b == True)).sum(dim=dim)
    union = (pred.bool() | target.bool()).sum(dim=dim)  
    
    ious = (tp + smooth) / (union + smooth)
    iou = ious.mean()

    prec = (ious > ths).float().mean()
    
    recs = (tp+smooth)/(tp+fn+smooth)
    recs = recs.mean()
    rec = (recs > ths).float().mean()

    return {
        'iou': iou,
        'precision': prec,
        'recall': rec,
    }