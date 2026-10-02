"""
Evaluate a trained LandSentry checkpoint on a given region.

Run it against BOTH the matching region and the other region's data to fill
out the full cross-event generalization matrix, e.g.:

    python evaluate.py --checkpoint checkpoints/siamese_kodagu_best.pt --data_root data/ml/kodagu/test   # same-event
    python evaluate.py --checkpoint checkpoints/siamese_kodagu_best.pt --data_root data/ml/wayanad       # cross-event
    python evaluate.py --checkpoint checkpoints/siamese_wayanad_best.pt --data_root data/ml/wayanad      # same-event
    python evaluate.py --checkpoint checkpoints/siamese_wayanad_best.pt --data_root data/ml/kodagu/test  # cross-event

Repeat for the transformer checkpoints to compare both architectures.
"""
import argparse
import torch
from torch.utils.data import DataLoader

from data.dataset import NpyLandslideDataset
from models.siamese_cnn import SiameseChangeDetector
from models.transformer_cd import TransformerChangeDetector
from utils.metrics import compute_metrics


def build_model(name, in_channels):
    if name == "siamese":
        return SiameseChangeDetector(in_channels=in_channels)
    if name == "transformer":
        return TransformerChangeDetector(in_channels=in_channels)
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True, help="Region to evaluate on (may differ from training region)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = build_model(ckpt["model_name"], ckpt["in_channels"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = NpyLandslideDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    from utils.metrics import get_tp_fp_fn
    
    global_tp = 0.0
    global_fp = 0.0
    global_fn = 0.0
    
    with torch.no_grad():
        for pre, post, mask in loader:
            pre, post, mask = pre.to(device), post.to(device), mask.to(device)
            probs = torch.sigmoid(model(pre, post))
            
            tp, fp, fn = get_tp_fp_fn(probs, mask, threshold=0.5)
            global_tp += tp
            global_fp += fp
            global_fn += fn

    eps = 1e-7
    precision = global_tp / (global_tp + global_fp + eps)
    recall = global_tp / (global_tp + global_fn + eps)
    dice = 2 * precision * recall / (precision + recall + eps)
    iou = global_tp / (global_tp + global_fp + global_fn + eps)

    print(f"Model:        {ckpt['model_name']} (checkpoint from epoch {ckpt['epoch']})")
    print(f"Evaluated on: {args.data_root}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"Dice score:   {dice:.4f}")
    print(f"IoU:          {iou:.4f}")


if __name__ == "__main__":
    main()
