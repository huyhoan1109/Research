import torch
import torch.nn.functional as F

def clip_loss(model, image, text):
    logits_per_image, logits_per_text = model(image, text)
    ground_truth = torch.arange(len(image),dtype=torch.long,device=image.device)
    loss_img = F.cross_entropy(logits_per_image, ground_truth)
    loss_text = F.cross_entropy(logits_per_text, ground_truth)
    total_loss = (loss_img + loss_text) / 2
    return total_loss