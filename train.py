"""
Train LandSentry's Siamese CNN or transformer-based bitemporal change detector.

Run once per region to later build the cross-event generalization matrix.
The dataset is assumed to be in .npy format with explicit train/ and validation/ splits.

Usage:
    python train.py --model siamese     --data_root data/ml/kodagu --run_name siamese_kodagu
    python train.py --model transformer --data_root data/ml/kodagu --run_name transformer_kodagu
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader

from data.dataset import NpyLandslideDataset
from models.siamese_cnn import SiameseChangeDetector
from models.transformer_cd import TransformerChangeDetector
from models.losses import get_loss_function, coral_loss
from utils.metrics import compute_metrics


def build_model(name, in_channels):
    if name == "siamese":
        return SiameseChangeDetector(in_channels=in_channels)
    if name == "transformer":
        return TransformerChangeDetector(in_channels=in_channels)
    raise ValueError(f"Unknown model: {name}")


def run_epoch(model, loader, criterion, device, optimizer=None, target_loader=None, use_coral=False, lambda_coral=0.1):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0
    agg = {"precision": 0.0, "recall": 0.0, "dice": 0.0, "iou": 0.0}

    # If domain alignment is enabled, zip iterators
    if train and target_loader is not None and use_coral:
        target_iter = iter(target_loader)
    else:
        target_iter = None

    with torch.set_grad_enabled(train):
        for pre, post, mask in loader:
            pre, post, mask = pre.to(device), post.to(device), mask.to(device)
            
            if train and target_iter is not None:
                try:
                    t_pre, t_post, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    t_pre, t_post, _ = next(target_iter)
                
                t_pre, t_post = t_pre.to(device), t_post.to(device)
                
                logits, src_features = model(pre, post, return_features=True)
                _, tgt_features = model(t_pre, t_post, return_features=True)
                
                base_loss = criterion(logits, mask)
                c_loss = coral_loss(src_features, tgt_features)
                loss = base_loss + lambda_coral * c_loss
            else:
                logits = model(pre, post)
                loss = criterion(logits, mask)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * pre.size(0)
            probs = torch.sigmoid(logits)
            m = compute_metrics(probs, mask)
            for k in agg:
                agg[k] += m[k] * pre.size(0)

    n = len(loader.dataset)
    return total_loss / n, {k: v / n for k, v in agg.items()}


def main():
    import config
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["siamese", "transformer"], required=True)
    parser.add_argument("--data_root", required=True, help="Folder with train/ and validation/ subfolders")
    parser.add_argument("--target_data_root", default=None, help="Folder with target domain data for CORAL alignment")
    parser.add_argument("--in_channels", type=int, default=config.IN_CHANNELS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"{args.model}_{os.path.basename(args.data_root.rstrip('/'))}"
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "validation")
    
    train_dataset = NpyLandslideDataset(train_dir)
    val_dataset = NpyLandslideDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    target_loader = None
    if args.target_data_root and config.USE_CORAL:
        tgt_train_dir = os.path.join(args.target_data_root, "train")
        tgt_dataset = NpyLandslideDataset(tgt_train_dir)
        target_loader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = build_model(args.model, args.in_channels).to(device)
    
    import config
    from models.losses import get_loss_function

    pos_weight = torch.tensor([10.0]).to(device)
    criterion = get_loss_function(
        loss_type=config.LOSS_TYPE, 
        pos_weight=pos_weight, 
        lambda_boundary=config.LAMBDA_BOUNDARY
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_dice = 0.0
    patience_counter = 0
    early_stopping_patience = 12

    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, 
            target_loader=target_loader, 
            use_coral=config.USE_CORAL, 
            lambda_coral=config.LAMBDA_CORAL
        )
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        
        scheduler.step(val_metrics['dice'])

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"[{run_name}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} val_recall={val_metrics['recall']:.4f} "
            f"lr={current_lr:.2e}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            patience_counter = 0
            ckpt_path = os.path.join(args.checkpoint_dir, f"{run_name}_best.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "in_channels": args.in_channels,
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"  -> saved new best checkpoint to {ckpt_path} (dice={best_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break


if __name__ == "__main__":
    main()
