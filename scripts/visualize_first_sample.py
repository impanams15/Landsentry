from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

SAMPLE_DIR = Path("data/processed/kodagu/sample_001")

PRE_PATH = SAMPLE_DIR / "pre.tif"
POST_PATH = SAMPLE_DIR / "post.tif"
MASK_PATH = SAMPLE_DIR / "mask.tif"


print("\n========== LAND SENTRY SAMPLE QUALITY CHECK ==========\n")


# ============================================================
# READ FILES
# ============================================================

with rasterio.open(PRE_PATH) as src:
    pre = src.read()
    pre_profile = src.profile

with rasterio.open(POST_PATH) as src:
    post = src.read()
    post_profile = src.profile

with rasterio.open(MASK_PATH) as src:
    mask = src.read(1)
    mask_profile = src.profile


# ============================================================
# PRINT METADATA
# ============================================================

print("PRE IMAGE")
print("  Shape:", pre.shape)
print("  CRS:", pre_profile["crs"])
print("  Transform:", pre_profile["transform"])

print("\nPOST IMAGE")
print("  Shape:", post.shape)
print("  CRS:", post_profile["crs"])
print("  Transform:", post_profile["transform"])

print("\nMASK")
print("  Shape:", mask.shape)
print("  CRS:", mask_profile["crs"])
print("  Transform:", mask_profile["transform"])


# ============================================================
# ALIGNMENT CHECK
# ============================================================

print("\n========== ALIGNMENT CHECK ==========")

if (
    pre.shape[1:] == post.shape[1:] == mask.shape
    and pre_profile["crs"] == post_profile["crs"] == mask_profile["crs"]
    and pre_profile["transform"] == post_profile["transform"] == mask_profile["transform"]
):
    print("✓ PRE, POST and MASK are perfectly aligned!")
else:
    print("⚠️ Alignment mismatch detected!")


# ============================================================
# MASK CHECK
# ============================================================

print("\n========== MASK CHECK ==========")

unique_values = np.unique(mask)

print("Unique mask values:", unique_values)

landslide_pixels = np.sum(mask == 1)
background_pixels = np.sum(mask == 0)

print("Landslide pixels:", landslide_pixels)
print("Background pixels:", background_pixels)

if landslide_pixels > 0:
    print("✓ Mask contains landslide pixels!")
else:
    print("✗ WARNING: Mask is empty!")


# ============================================================
# CREATE RGB VISUALIZATION
# ============================================================

def create_rgb(image):

    # Sentinel-2 band order:
    # B4 = Red
    # B3 = Green
    # B2 = Blue

    rgb = np.transpose(image[:3], (1, 2, 0)).astype(float)

    # Robust normalization for visualization
    lower = np.percentile(rgb, 2)
    upper = np.percentile(rgb, 98)

    rgb = np.clip(
        (rgb - lower) / (upper - lower + 1e-8),
        0,
        1
    )

    return rgb


pre_rgb = create_rgb(pre)
post_rgb = create_rgb(post)


# ============================================================
# VISUALIZE
# ============================================================

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(pre_rgb)
plt.title("PRE-EVENT Sentinel-2")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(post_rgb)
plt.title("POST-EVENT Sentinel-2")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mask, cmap="gray")
plt.title("LANDSLIDE MASK")
plt.axis("off")

plt.tight_layout()

OUTPUT_PATH = SAMPLE_DIR / "sample_visualization.png"

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print(f"\n✓ Visualization saved to:\n{OUTPUT_PATH}")

print("\n========== QUALITY CHECK COMPLETE ==========\n")