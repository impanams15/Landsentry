import torch
from torch.utils.data import DataLoader
from data.dataset import NpyLandslideDataset
from models.siamese_cnn import SiameseChangeDetector
from models.transformer_cd import TransformerChangeDetector

def main():
    print("Initializing dataset...")
    # Load from kodagu validation to test
    dataset = NpyLandslideDataset("data/ml/kodagu/validation")
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    pre, post, mask = next(iter(loader))
    print(f"PRE shape: {pre.shape}")
    print(f"POST shape: {post.shape}")
    print(f"MASK shape: {mask.shape}")
    
    print("\nInitializing Siamese CNN...")
    siamese = SiameseChangeDetector(in_channels=4)
    print("Running Siamese CNN forward pass...")
    out_siamese = siamese(pre, post)
    print(f"Siamese CNN output shape: {out_siamese.shape}")
    
    print("\nInitializing Transformer Model...")
    transformer = TransformerChangeDetector(in_channels=4)
    print("Running Transformer forward pass...")
    out_transformer = transformer(pre, post)
    print(f"Transformer output shape: {out_transformer.shape}")
    
    print("\nVerification successful!")

if __name__ == "__main__":
    main()
