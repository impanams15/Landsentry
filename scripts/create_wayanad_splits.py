import os
import json
import numpy as np
import random

SEED = 42

def main():
    wayanad_dir = os.path.join('data', 'ml', 'wayanad')
    images_path = os.path.join(wayanad_dir, 'images.npy')
    masks_path = os.path.join(wayanad_dir, 'masks.npy')
    meta_path = os.path.join(wayanad_dir, 'metadata.json')

    # Load data
    images = np.load(images_path)
    masks = np.load(masks_path)
    
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # Note: no spatial metadata in metadata.json based on previous inspection
    has_spatial_metadata = False

    N = len(images)
    
    # Identify positive and background indices
    pos_indices = []
    bg_indices = []
    
    for i in range(N):
        if np.sum(masks[i]) > 0:
            pos_indices.append(i)
        else:
            bg_indices.append(i)

    assert len(pos_indices) == 17
    assert len(bg_indices) == 47

    # Shuffle with seed
    random.seed(SEED)
    random.shuffle(pos_indices)
    random.shuffle(bg_indices)

    # Split targets: 44 train, 10 val, 10 test
    # Positives: 17 -> 11 train, 3 val, 3 test
    # Backgrounds: 47 -> (44-11=33) train, (10-3=7) val, (10-3=7) test
    train_pos = pos_indices[:11]
    val_pos = pos_indices[11:14]
    test_pos = pos_indices[14:]

    train_bg = bg_indices[:33]
    val_bg = bg_indices[33:40]
    test_bg = bg_indices[40:]

    train_indices = train_pos + train_bg
    val_indices = val_pos + val_bg
    test_indices = test_pos + test_bg

    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)

    # Verification checks on indices
    all_indices = set(train_indices + val_indices + test_indices)
    assert len(all_indices) == N
    assert len(train_indices) == 44
    assert len(val_indices) == 10
    assert len(test_indices) == 10

    # Create directories and save shards
    splits = {
        'train': train_indices,
        'validation': val_indices,
        'test': test_indices
    }

    counts_report = {}

    for split_name, indices in splits.items():
        out_dir = os.path.join(wayanad_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)
        
        subset_images = images[indices]
        subset_masks = masks[indices]
        
        np.save(os.path.join(out_dir, 'images.npy'), subset_images)
        np.save(os.path.join(out_dir, 'masks.npy'), subset_masks)
        
        pos_count = sum(1 for idx in indices if idx in pos_indices)
        bg_count = len(indices) - pos_count
        
        counts_report[split_name] = {
            'total': len(indices),
            'positive': pos_count,
            'background': bg_count
        }

        # Write split-specific metadata
        split_meta = metadata.copy()
        split_meta['total_patches'] = len(indices)
        split_meta['positive_patches'] = pos_count
        split_meta['background_patches'] = bg_count
        split_meta['purpose'] = f"Wayanad {split_name} split"
        
        with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
            json.dump(split_meta, f, indent=4)

    # Save wayanad_splits.json
    splits_doc = {
        "random_seed": SEED,
        "total_samples": N,
        "split_counts": counts_report,
        "train_indices": train_indices,
        "validation_indices": val_indices,
        "test_indices": test_indices
    }
    splits_json_path = os.path.join(wayanad_dir, 'wayanad_splits.json')
    with open(splits_json_path, 'w') as f:
        json.dump(splits_doc, f, indent=4)

    # Verify saved shapes and channels
    print(f"--- Verification Report ---")
    print(f"Original images shape: {images.shape}")
    print(f"Original masks shape: {masks.shape}")
    print(f"All 64 original samples accounted for exactly once: {len(all_indices) == N and sum(len(x) for x in splits.values()) == N}")
    print(f"No duplicated indices across train/val/test: {len(set(train_indices).intersection(set(val_indices))) == 0 and len(set(train_indices).intersection(set(test_indices))) == 0 and len(set(val_indices).intersection(set(test_indices))) == 0}")
    print()
    for split_name in splits.keys():
        out_dir = os.path.join(wayanad_dir, split_name)
        img = np.load(os.path.join(out_dir, 'images.npy'))
        msk = np.load(os.path.join(out_dir, 'masks.npy'))
        c = counts_report[split_name]
        print(f"Split: {split_name}")
        print(f"  Images shape: {img.shape}")
        print(f"  Masks shape: {msk.shape}")
        print(f"  Binary masks check (only 0 and 1): {np.array_equal(msk, msk.astype(bool))}")
        print(f"  Counts -> Total: {c['total']}, Pos: {c['positive']}, Bg: {c['background']}")
        assert img.shape[1] == 8
        assert img.shape[0] == msk.shape[0]

    print("\nNormalization compatibility check (Kodagu stats existence):")
    kodagu_stats_path = os.path.join('data', 'ml', 'kodagu', 'normalization_stats.json')
    print(f"  Kodagu stats exist: {os.path.exists(kodagu_stats_path)}")
    print("\nSpatial metadata available: False")
    print("Spatial leakage could be assessed: No")
    print(f"Random seed used: {SEED}")
    
    print("\nFiles created:")
    print(" - " + splits_json_path)
    for s in splits.keys():
        print(f" - {os.path.join(wayanad_dir, s, 'images.npy')}")
        print(f" - {os.path.join(wayanad_dir, s, 'masks.npy')}")
        print(f" - {os.path.join(wayanad_dir, s, 'metadata.json')}")

if __name__ == '__main__':
    main()
