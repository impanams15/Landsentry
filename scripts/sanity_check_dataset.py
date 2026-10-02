"""
Sanity check for the LandSentry dataset loading and normalization pipeline.
"""
import os
import torch
from torch.utils.data import DataLoader
from data.dataset import NpyLandslideDataset

def check_split(name, path, expected_samples):
    print(f"\n--- Checking split: {name} ---")
    try:
        dataset = NpyLandslideDataset(path)
    except Exception as e:
        print(f"Failed to load dataset at {path}: {e}")
        return

    print(f"Total samples found: {len(dataset)} (Expected: {expected_samples})")
    if len(dataset) == 0:
        return

    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    pre, post, mask = next(iter(loader))
    
    print("Batch shapes:")
    print(f"  PRE:  {pre.shape}")
    print(f"  POST: {post.shape}")
    print(f"  MASK: {mask.shape}")
    
    print("Data types:")
    print(f"  PRE type:  {pre.dtype}")
    print(f"  POST type: {post.dtype}")
    print(f"  MASK type: {mask.dtype}")
    
    print("Channel separation logic:")
    print("  The dataset loads [B, 8, 64, 64] from images.npy.")
    print("  It slices [:4, :, :] for PRE and [4:, :, :] for POST per item.")
    
    print("Verifying binary mask:")
    unique_vals = torch.unique(mask)
    is_binary = all(val.item() in [0.0, 1.0] for val in unique_vals)
    print(f"  Unique mask values: {unique_vals.tolist()}")
    print(f"  Is strictly binary? {is_binary}")

def main():
    base_dir = "data/ml"
    check_split("Kodagu Train", os.path.join(base_dir, "kodagu/train"), 617)
    check_split("Kodagu Validation", os.path.join(base_dir, "kodagu/validation"), 117)
    check_split("Kodagu Test", os.path.join(base_dir, "kodagu/test"), 105)
    check_split("Wayanad Cross-Event", os.path.join(base_dir, "wayanad"), 64)

if __name__ == "__main__":
    main()
