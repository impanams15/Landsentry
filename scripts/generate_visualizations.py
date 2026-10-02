import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data.dataset import NpyLandslideDataset
from models.siamese_cnn import SiameseChangeDetector
from models.transformer_cd import TransformerChangeDetector

def ensure_dirs():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/qualitative", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

def generate_table():
    md = """# Final Quantitative Comparison

| Model | Kodagu Test (Dice) | Kodagu Test (IoU) | Kodagu Test (Precision) | Kodagu Test (Recall) | Wayanad (Dice) | Wayanad (IoU) | Wayanad (Precision) | Wayanad (Recall) |
|-------|--------------------|-------------------|-------------------------|----------------------|----------------|---------------|---------------------|------------------|
| Siamese CNN | 0.3900 | 0.2422 | 0.3139 | 0.5147 | 0.1383 | 0.0743 | 0.1046 | 0.2040 |
| Transformer | 0.3750 | 0.2308 | 0.2796 | 0.5695 | 0.0856 | 0.0447 | 0.0453 | 0.7753 |
"""
    with open("outputs/results/quantitative_comparison.md", "w") as f:
        f.write(md)
    print("Generated outputs/results/quantitative_comparison.md")

def generate_bar_charts():
    metrics = ["Dice", "IoU", "Precision", "Recall"]
    
    siamese_kodagu = [0.3900, 0.2422, 0.3139, 0.5147]
    siamese_wayanad = [0.1383, 0.0743, 0.1046, 0.2040]
    
    transformer_kodagu = [0.3750, 0.2308, 0.2796, 0.5695]
    transformer_wayanad = [0.0856, 0.0447, 0.0453, 0.7753]
    
    labels = ["Kodagu TEST", "Wayanad"]
    x = np.arange(len(labels))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        siamese_vals = [siamese_kodagu[i], siamese_wayanad[i]]
        transformer_vals = [transformer_kodagu[i], transformer_wayanad[i]]
        
        rects1 = ax.bar(x - width/2, siamese_vals, width, label='Siamese CNN', color='#1f77b4')
        rects2 = ax.bar(x + width/2, transformer_vals, width, label='Transformer', color='#ff7f0e')
        
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison by Region')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        plt.savefig(f"outputs/figures/{metric.lower()}_comparison.png", dpi=150)
        plt.close()
    print("Generated bar charts in outputs/figures/")

def build_models():
    device = torch.device("cpu")
    s_ckpt = torch.load("checkpoints/siamese_kodagu_best.pt", map_location=device)
    s_model = SiameseChangeDetector(in_channels=4).to(device)
    s_model.load_state_dict(s_ckpt["model_state"])
    s_model.eval()
    
    t_ckpt = torch.load("checkpoints/transformer_kodagu_best.pt", map_location=device)
    t_model = TransformerChangeDetector(in_channels=4).to(device)
    t_model.load_state_dict(t_ckpt["model_state"])
    t_model.eval()
    
    return s_model, t_model

def select_samples(dataset, num_large=2, num_small=2, num_empty=1):
    mask_sums = [dataset.masks[i].sum() for i in range(len(dataset))]
    sorted_idx = np.argsort(mask_sums)
    
    empty_idx = []
    small_idx = []
    large_idx = []
    
    for idx in sorted_idx:
        s = mask_sums[idx]
        if s == 0 and len(empty_idx) < num_empty:
            empty_idx.append(idx)
        elif s > 0 and s < 50 and len(small_idx) < num_small:
            small_idx.append(idx)
        
    for idx in reversed(sorted_idx):
        if len(large_idx) < num_large:
            large_idx.append(idx)
            
    return empty_idx + small_idx + large_idx

def plot_qualitative_sample(dataset_name, idx, pre, post, mask, s_pred, t_pred, save_name):
    def norm_img(img_tensor):
        img = img_tensor[:3].permute(1, 2, 0).numpy() # (3, 64, 64) -> (64, 64, 3)
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)
        return img
    
    pre_img = norm_img(pre)
    post_img = norm_img(post)
    
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    
    axs[0].imshow(pre_img)
    axs[0].set_title(f"PRE ({dataset_name} #{idx})")
    axs[0].axis('off')
    
    axs[1].imshow(post_img)
    axs[1].set_title("POST")
    axs[1].axis('off')
    
    axs[2].imshow(mask.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[2].set_title("Ground Truth Mask")
    axs[2].axis('off')
    
    axs[3].imshow(s_pred.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[3].set_title("Siamese Pred")
    axs[3].axis('off')
    
    axs[4].imshow(t_pred.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[4].set_title("Transformer Pred")
    axs[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"outputs/qualitative/{save_name}.png", dpi=150)
    plt.close()

def generate_qualitative():
    device = torch.device("cpu")
    s_model, t_model = build_models()
    
    datasets = {
        "Kodagu_Test": NpyLandslideDataset("data/ml/kodagu/test"),
        "Wayanad": NpyLandslideDataset("data/ml/wayanad")
    }
    
    with torch.no_grad():
        for dname, ds in datasets.items():
            sample_indices = select_samples(ds)
            print(f"Selected samples for {dname}: {sample_indices}")
            
            for idx in sample_indices:
                pre, post, mask = ds[idx]
                pre_b = pre.unsqueeze(0).to(device)
                post_b = post.unsqueeze(0).to(device)
                
                s_logits = s_model(pre_b, post_b)
                t_logits = t_model(pre_b, post_b)
                
                s_pred = (torch.sigmoid(s_logits) > 0.5).float().cpu()[0]
                t_pred = (torch.sigmoid(t_logits) > 0.5).float().cpu()[0]
                
                plot_qualitative_sample(dname, idx, pre, post, mask, s_pred, t_pred, f"qualitative_{dname}_{idx}")

def analyze_failure_cases():
    device = torch.device("cpu")
    _, t_model = build_models()
    ds = NpyLandslideDataset("data/ml/wayanad")
    
    worst_idx = -1
    max_fp = -1
    
    with torch.no_grad():
        for idx in range(len(ds)):
            if ds.masks[idx].sum() == 0:
                pre, post, _ = ds[idx]
                pre_b = pre.unsqueeze(0).to(device)
                post_b = post.unsqueeze(0).to(device)
                t_logits = t_model(pre_b, post_b)
                t_pred = (torch.sigmoid(t_logits) > 0.5).float().cpu()
                fp = t_pred.sum().item()
                if fp > max_fp:
                    max_fp = fp
                    worst_idx = idx
                    
    if worst_idx != -1:
        print(f"Wayanad failure case (High FP) found at idx {worst_idx} with {max_fp} false positive pixels.")
        pre, post, mask = ds[worst_idx]
        pre_b = pre.unsqueeze(0).to(device)
        post_b = post.unsqueeze(0).to(device)
        
        s_model, _ = build_models()
        with torch.no_grad():
            s_logits = s_model(pre_b, post_b)
            s_pred = (torch.sigmoid(s_logits) > 0.5).float().cpu()[0]
            t_logits = t_model(pre_b, post_b)
            t_pred = (torch.sigmoid(t_logits) > 0.5).float().cpu()[0]
            
        plot_qualitative_sample("Wayanad_Failure", worst_idx, pre, post, mask, s_pred, t_pred, f"failure_wayanad_idx_{worst_idx}")

def main():
    ensure_dirs()
    generate_table()
    generate_bar_charts()
    generate_qualitative()
    analyze_failure_cases()
    print("All visualizations generated successfully.")

if __name__ == "__main__":
    main()
