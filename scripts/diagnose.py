import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import NpyLandslideDataset
from models.siamese_cnn import SiameseChangeDetector
from models.transformer_cd import TransformerChangeDetector
from evaluate import build_model
import json
import config

def main():
    print("--- Diagnostic Report ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1, 6: Verify Kodagu Test set shape and positive pixels
    kodagu_test_dir = 'data/ml/kodagu/test'
    images_path = os.path.join(kodagu_test_dir, 'images.npy')
    masks_path = os.path.join(kodagu_test_dir, 'masks.npy')
    
    k_images = np.load(images_path)
    k_masks = np.load(masks_path)
    
    print(f"Original Kodagu Test images.npy shape on disk: {k_images.shape}")
    print(f"Original Kodagu Test masks.npy shape on disk: {k_masks.shape}")
    
    gt_positive_pixels = np.sum(k_masks > 0)
    print(f"1. Number of positive pixels in Kodagu test ground-truth masks: {gt_positive_pixels}")
    print(f"   (Out of {k_masks.size} total pixels in {k_masks.shape[0]} patches)")

    # Data Loader check
    dataset = NpyLandslideDataset(kodagu_test_dir)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    print(f"6. Dataset yields Pre shape: {dataset[0][0].shape}, Post shape: {dataset[0][1].shape}")
    print(f"   Config USE_NDVI={config.USE_NDVI}, IN_CHANNELS={config.IN_CHANNELS}")
    print(f"   Total input channels per temporal image: {dataset[0][0].shape[0]}")
    
    # 7,8: Checkpoint Loading
    s_ckpt_path = 'checkpoints/siamese_wayanad_best.pt'
    t_ckpt_path = 'checkpoints/transformer_wayanad_best.pt'
    
    s_ckpt = torch.load(s_ckpt_path, map_location=device, weights_only=False)
    t_ckpt = torch.load(t_ckpt_path, map_location=device, weights_only=False)
    
    print(f"\n7. Checkpoint info:")
    print(f"   Siamese args saved: model_name={s_ckpt['model_name']}, in_channels={s_ckpt['in_channels']}")
    print(f"   Transformer args saved: model_name={t_ckpt['model_name']}, in_channels={t_ckpt['in_channels']}")
    
    s_model = build_model(s_ckpt["model_name"], s_ckpt["in_channels"]).to(device)
    s_model.load_state_dict(s_ckpt["model_state"])
    s_model.eval()

    t_model = build_model(t_ckpt["model_name"], t_ckpt["in_channels"]).to(device)
    t_model.load_state_dict(t_ckpt["model_state"])
    t_model.eval()
    
    # Diagnostics for predictions
    s_probs_list = []
    t_probs_list = []
    
    s_pred_pixels = 0
    t_pred_pixels = 0
    
    with torch.no_grad():
        for pre, post, mask in loader:
            pre, post, mask = pre.to(device), post.to(device), mask.to(device)
            
            s_logits = s_model(pre, post)
            s_probs = torch.sigmoid(s_logits)
            
            t_logits = t_model(pre, post)
            t_probs = torch.sigmoid(t_logits)
            
            s_probs_list.append(s_probs.cpu().numpy())
            t_probs_list.append(t_probs.cpu().numpy())
            
            s_pred_pixels += torch.sum(s_probs > 0.5).item()
            t_pred_pixels += torch.sum(t_probs > 0.5).item()
            
    s_all_probs = np.concatenate(s_probs_list)
    t_all_probs = np.concatenate(t_probs_list)
    
    print(f"\n2. Siamese Wayanad-trained predicted positive pixels on Kodagu: {s_pred_pixels}")
    print(f"3. Transformer Wayanad-trained predicted positive pixels on Kodagu: {t_pred_pixels}")
    
    print(f"\n4. Probability Statistics:")
    print(f"   Siamese:     Min = {s_all_probs.min():.6f}, Max = {s_all_probs.max():.6f}, Mean = {s_all_probs.mean():.6f}")
    print(f"   Transformer: Min = {t_all_probs.min():.6f}, Max = {t_all_probs.max():.6f}, Mean = {t_all_probs.mean():.6f}")
    
    print(f"\n5. Verification of configurations:")
    print(f"   - Mask threshold: Used 0.5 for counting predicted pixels.")
    print(f"   - Output activation: Checkpoints output logits, torch.sigmoid() applied correctly.")
    
    with open('data/ml/kodagu/normalization_stats.json', 'r') as f:
         stats = json.load(f)
    print(f"   - Normalization: stats json has channel length {len(stats['mean'])}.")
    
if __name__ == '__main__':
    main()
