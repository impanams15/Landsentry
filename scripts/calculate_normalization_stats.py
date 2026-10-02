import os
import json
import numpy as np


# ============================================================
# LAND SENTRY — NORMALIZATION STATISTICS
# ============================================================

print("\n============================================================")
print("LAND SENTRY — CALCULATING NORMALIZATION STATISTICS")
print("============================================================")


# ============================================================
# PATHS
# ============================================================

TRAIN_IMAGES_PATH = "data/ml/kodagu/train/images.npy"

OUTPUT_DIR = "data/ml/kodagu"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "normalization_stats.json"
)


# ============================================================
# LOAD TRAINING DATA ONLY
# ============================================================

print("\nLoading Kodagu TRAIN dataset...")

images = np.load(TRAIN_IMAGES_PATH).astype(np.float32)

print(f"✓ Training images loaded")
print(f"✓ Shape: {images.shape}")

# Expected:
# (N, C, H, W)


# ============================================================
# CALCULATE CHANNEL-WISE MEAN AND STD
# ============================================================

print("\nCalculating channel-wise statistics...")

channel_means = []
channel_stds = []

num_channels = images.shape[1]

for channel in range(num_channels):

    channel_data = images[:, channel, :, :]

    mean = float(np.mean(channel_data))
    std = float(np.std(channel_data))

    channel_means.append(mean)
    channel_stds.append(std)

    print(
        f"Channel {channel + 1}: "
        f"Mean = {mean:.4f} | "
        f"Std = {std:.4f}"
    )


# ============================================================
# CHANNEL INFORMATION
# ============================================================

channel_names = [
    "PRE_B4",
    "PRE_B3",
    "PRE_B2",
    "PRE_B8",
    "POST_B4",
    "POST_B3",
    "POST_B2",
    "POST_B8"
]


# ============================================================
# SAVE STATISTICS
# ============================================================

stats = {
    "dataset": "Kodagu training split only",
    "purpose": (
        "Channel-wise normalization statistics for "
        "Land Sentry segmentation model"
    ),
    "normalization_method": "Z-score normalization",
    "formula": "(x - mean) / std",
    "channels": channel_names,
    "mean": channel_means,
    "std": channel_stds
}


with open(OUTPUT_FILE, "w") as f:
    json.dump(stats, f, indent=4)


print("\n============================================================")
print("NORMALIZATION STATISTICS SAVED")
print("============================================================")

print(f"\n✓ Saved to:")
print(f"  {OUTPUT_FILE}")

print("\nThese statistics will be used for:")
print("✓ Kodagu Training")
print("✓ Kodagu Validation")
print("✓ Kodagu Test")
print("✓ Wayanad Cross-Event Test")

print("\n============================================================")
print("🎉 NORMALIZATION SETUP COMPLETE!")
print("============================================================\n")