import os
import numpy as np


# ============================================================
# LAND SENTRY — PIXEL-LEVEL CLASS BALANCE ANALYSIS
# ============================================================

print("\n============================================================")
print("LAND SENTRY — PIXEL-LEVEL CLASS BALANCE ANALYSIS")
print("============================================================")


ML_DIR = "data/ml/kodagu"

SPLITS = [
    "train",
    "validation",
    "test"
]


for split in SPLITS:

    print("\n============================================================")
    print(f"{split.upper()} SPLIT")
    print("============================================================")

    masks_path = os.path.join(
        ML_DIR,
        split,
        "masks.npy"
    )

    masks = np.load(masks_path)

    total_pixels = masks.size

    landslide_pixels = np.sum(masks == 1)

    background_pixels = np.sum(masks == 0)

    landslide_percentage = (
        landslide_pixels / total_pixels
    ) * 100

    background_percentage = (
        background_pixels / total_pixels
    ) * 100

    imbalance_ratio = (
        background_pixels / landslide_pixels
        if landslide_pixels > 0
        else float("inf")
    )

    print(f"\nMask shape: {masks.shape}")

    print(f"\nTotal pixels      : {total_pixels:,}")
    print(f"Landslide pixels  : {landslide_pixels:,}")
    print(f"Background pixels : {background_pixels:,}")

    print(f"\nLandslide %       : {landslide_percentage:.4f}%")
    print(f"Background %      : {background_percentage:.4f}%")

    print(
        f"\nBackground : Landslide ratio = "
        f"{imbalance_ratio:.2f} : 1"
    )


print("\n============================================================")
print("ANALYSIS COMPLETE")
print("============================================================\n")