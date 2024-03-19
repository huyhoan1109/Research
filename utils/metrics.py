def calculate_metrics(pred, target, dim, smooth=1e-6, ths=0.5):
    union = (pred.bool() | target.bool()).sum(dim=dim)  
    
    tp_mask = (target == 1) & (pred == 1)
    tn_mask = (target == 0) & (pred == 0)
    fp_mask = (target == 1) & (pred == 0)
    fn_mask = (target == 0) & (pred == 1)
    
    tp = union[tp_mask].sum(dim=dim)
    tn = union[tn_mask].sum(dim=dim)
    fp = union[fp_mask].sum(dim=dim)
    fn = union[fn_mask].sum(dim=dim)
    
    ious = (tp+smooth)/(tp+tn+smooth)
    iou = ious.mean()

    precs = (tp+smooth)/(tp+fp+smooth)
    precs = precs.mean()
    prec = (precs > ths).float().mean()
    
    recs = (tp+smooth)/(tp+fn+smooth)
    recs = recs.mean()
    rec = (recs > ths).float().mean()

    return {
        'iou': iou * 100,
        'precision': prec * 100,
        'recall': rec * 100,
    }