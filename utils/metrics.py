"""Standard change-detection metrics used to benchmark both models."""
import torch


def compute_metrics(preds, targets, threshold=0.5, eps=1e-7):
    """
    preds:   [B, 1, H, W] probabilities in [0, 1] (apply sigmoid before calling this)
    targets: [B, 1, H, W] binary ground-truth mask (0 = background, 1 = landslide)

    Returns a dict with precision, recall, f1, iou, micro-averaged over the batch
    (pixels pooled across the whole batch before computing the ratios, which is
    the standard way to report these metrics for imbalanced segmentation masks).
    """
    preds_bin = (preds > threshold).float()
    targets = targets.float()

    tp = (preds_bin * targets).sum()
    fp = (preds_bin * (1 - targets)).sum()
    fn = ((1 - preds_bin) * targets).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "dice": dice.item(),
        "iou": iou.item(),
    }


def get_tp_fp_fn(preds, targets, threshold=0.5):
    """Returns raw (TP, FP, FN) scalar counts for global micro-averaging."""
    preds_bin = (preds > threshold).float()
    targets = targets.float()

    tp = (preds_bin * targets).sum().item()
    fp = (preds_bin * (1 - targets)).sum().item()
    fn = ((1 - preds_bin) * targets).sum().item()
    
    return tp, fp, fn
