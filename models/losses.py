"""Loss functions for training the change-detection models."""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).reshape(logits.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined pixel-wise BCE + region-overlap Dice loss.
    Landslide pixels are typically a small minority of each patch, so Dice
    (which directly rewards mask overlap regardless of class balance)
    complements plain BCE and helps counter that imbalance.
    """

    def __init__(self, bce_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + (1 - self.bce_weight) * self.dice(logits, targets)


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=4.0 / 3.0, eps=1e-7):
        """
        Alpha controls penalty for false negatives.
        Beta controls penalty for false positives.
        Gamma > 1 focuses on harder examples.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).reshape(logits.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        
        true_pos = (probs * targets).sum(dim=1)
        false_neg = ((1 - probs) * targets).sum(dim=1)
        false_pos = (probs * (1 - targets)).sum(dim=1)
        
        tversky = (true_pos + self.eps) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.eps)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky.mean()


import torch.nn.functional as F

class BoundaryAwareLoss(nn.Module):
    def __init__(self, base_loss_fn, lambda_boundary=0.5):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.lambda_boundary = lambda_boundary

    def extract_boundaries(self, mask):
        # Mask shape: [B, 1, H, W]
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = 1 - F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
        return dilated - eroded

    def forward(self, logits, targets):
        base_loss = self.base_loss_fn(logits, targets)
        
        with torch.no_grad():
            target_boundaries = self.extract_boundaries(targets)
            
        pred_probs = torch.sigmoid(logits)
        pred_boundaries = self.extract_boundaries(pred_probs)
        
        boundary_loss = F.mse_loss(pred_boundaries, target_boundaries)
        
        return base_loss + self.lambda_boundary * boundary_loss


def get_loss_function(loss_type="bce_dice", pos_weight=None, lambda_boundary=0.5):
    """Factory to retrieve configured loss dynamically."""
    if loss_type == "bce_dice":
        return BCEDiceLoss(pos_weight=pos_weight)
    elif loss_type == "focal_tversky":
        return FocalTverskyLoss()
    elif loss_type == "boundary_aware":
        base_loss = BCEDiceLoss(pos_weight=pos_weight)
        return BoundaryAwareLoss(base_loss_fn=base_loss, lambda_boundary=lambda_boundary)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def coral_loss(source, target):
    """
    CORAL (CORrelation ALignment) loss for lightweight domain adaptation.
    source: [B, C, H, W] feature map from source domain (e.g. Kodagu)
    target: [B, C, H, W] feature map from target domain (e.g. Wayanad)
    """
    # Flatten spatial dimensions
    source = source.view(source.size(0), source.size(1), -1) # [B, C, H*W]
    target = target.view(target.size(0), target.size(1), -1)
    
    # Average over spatial dimensions to get [B, C]
    source = source.mean(dim=2)
    target = target.mean(dim=2)
    
    d = source.size(1)

    # Source covariance
    source_c = source - torch.mean(source, dim=0, keepdim=True)
    source_c = torch.mm(source_c.t(), source_c) / (source.size(0) - 1 + 1e-6)

    # Target covariance
    target_c = target - torch.mean(target, dim=0, keepdim=True)
    target_c = torch.mm(target_c.t(), target_c) / (target.size(0) - 1 + 1e-6)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))
    loss = loss / (4 * d * d)
    
    return loss
