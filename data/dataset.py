"""PyTorch Dataset for paired pre-/post-event landslide patches."""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class NpyLandslideDataset(Dataset):
    """
    Bitemporal landslide change-detection dataset.
    Loads .npy files (images.npy, masks.npy) directly.
    """

    def __init__(self, data_root, stats_path=None):
        """
        data_root: Directory containing images.npy and masks.npy
        stats_path: Path to normalization_stats.json. If None, expects it
                    at 'data/ml/kodagu/normalization_stats.json' relative to cwd.
        """
        self.data_root = data_root
        
        if stats_path is None:
            stats_path = os.path.join("data", "ml", "kodagu", "normalization_stats.json")
            
        images_path = os.path.join(data_root, "images.npy")
        masks_path = os.path.join(data_root, "masks.npy")
        
        if not os.path.exists(images_path) or not os.path.exists(masks_path):
            raise RuntimeError(f"Could not find images.npy or masks.npy in {data_root}")
            
        import config
        from utils.preprocessing import compute_ndvi
        
        # Load entire arrays into memory
        self.images = np.load(images_path)  # [N, 8, 64, 64] (raw before normalization)
        self.masks = np.load(masks_path)    # [N, 64, 64]
        
        # Optionally compute and append NDVI before z-score normalization
        if config.USE_NDVI:
            # We assume images are [N, 8, H, W] where ch 0-3 are Pre, 4-7 are Post
            # Red is index 0, NIR is index 3 in each set
            pre_raw = np.transpose(self.images[:, :4], (0, 2, 3, 1)) # [N, H, W, 4]
            post_raw = np.transpose(self.images[:, 4:], (0, 2, 3, 1))
            
            pre_ndvi = compute_ndvi(pre_raw, red_idx=0, nir_idx=3).transpose(0, 3, 1, 2) # [N, 1, H, W]
            post_ndvi = compute_ndvi(post_raw, red_idx=0, nir_idx=3).transpose(0, 3, 1, 2)
            
            # Stack so it becomes [N, 10, H, W] -> (Pre RGBN, Pre NDVI, Post RGBN, Post NDVI)
            self.images = np.concatenate([self.images[:, :4], pre_ndvi, self.images[:, 4:], post_ndvi], axis=1)
        
        # Load and apply normalization statistics
        with open(stats_path, "r") as f:
            stats = json.load(f)
            
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        
        # If NDVI was added but not in stats, we use dummy stats (mean=0, std=1) 
        # so NDVI stays in its native [-1, 1] range rather than crashing.
        if config.USE_NDVI and len(mean) == 8:
            mean = np.insert(mean, 4, 0.0) # After Pre, insert Pre_NDVI mean
            mean = np.append(mean, 0.0)    # After Post, append Post_NDVI mean
            std = np.insert(std, 4, 1.0)
            std = np.append(std, 1.0)
            
        mean = mean.reshape(1, -1, 1, 1)
        std = std.reshape(1, -1, 1, 1)
        
        # Z-score normalization
        self.images = (self.images.astype(np.float32) - mean) / std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        c = img.shape[0] // 2
        pre = torch.from_numpy(img[:c, :, :]).float()
        post = torch.from_numpy(img[c:, :, :]).float()
        
        mask = self.masks[idx] # [64, 64]
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) # [1, 64, 64]
        
        return pre, post, mask_t
