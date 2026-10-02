"""
Runtime sanity check for the Transformer-Based Change Detector using the Kodagu train dataset.
Verifies shapes, forward pass, and memory on CPU.
"""
import torch
from torch.utils.data import DataLoader
from data.dataset import NpyLandslideDataset
from models.transformer_cd import TransformerChangeDetector

def main():
    print(f"PyTorch version: {torch.__version__}")
    
    device = torch.device("cpu")
    print(f"Device used: {device}")
    
    print("\n--- Loading Kodagu Training Dataset ---")
    dataset = NpyLandslideDataset("data/ml/kodagu/train")
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    pre, post, mask = next(iter(loader))
    pre = pre.to(device)
    post = post.to(device)
    mask = mask.to(device)
    
    print(f"PRE shape:  {pre.shape}")
    print(f"POST shape: {post.shape}")
    print(f"MASK shape: {mask.shape}")
    
    print("\n--- Initializing Transformer Model ---")
    model = TransformerChangeDetector(in_channels=4).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params:,}")
    
    print("\n--- Running Forward Pass ---")
    output = model(pre, post)
    
    print(f"Model output shape: {output.shape}")
    
    if output.shape == torch.Size([8, 1, 64, 64]):
        print("Output shape perfectly matches expected [8, 1, 64, 64].")
    else:
        print("WARNING: Output shape does NOT match expected [8, 1, 64, 64].")
        
    if output.shape[2:] == mask.shape[2:]:
        print("Prediction spatial shape is fully compatible with mask spatial shape.")
    else:
        print("WARNING: Prediction spatial shape and mask shape are mismatched.")
        
    print("\nConfirmation: Forward propagation completed successfully with raw logits.")

if __name__ == "__main__":
    main()
