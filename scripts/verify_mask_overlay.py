from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


SAMPLE_DIR = Path("data/processed/kodagu/sample_001")

PRE_PATH = SAMPLE_DIR / "pre.tif"
POST_PATH = SAMPLE_DIR / "post.tif"
MASK_PATH = SAMPLE_DIR / "mask.tif"


print("\n========== VERIFYING LANDSLIDE MASK OVERLAY ==========\n")


# --------------------------------------------------
# READ DATA
# --------------------------------------------------

with rasterio.open(PRE_PATH) as src:
    pre = src.read()

with rasterio.open(POST_PATH) as src:
    post = src.read()

with rasterio.open(MASK_PATH) as src:
    mask = src.read(1)


# --------------------------------------------------
# RGB FUNCTION
# --------------------------------------------------

def create_rgb(image):

    rgb = np.transpose(image[:3], (1, 2, 0)).astype(np.float32)

    # Normalize each visualization robustly
    for band in range(3):

        low = np.percentile(rgb[:, :, band], 2)
        high = np.percentile(rgb[:, :, band], 98)

        rgb[:, :, band] = np.clip(
            (rgb[:, :, band] - low) /
            (high - low + 1e-8),
            0,
            1
        )

    return rgb


pre_rgb = create_rgb(pre)
post_rgb = create_rgb(post)


# --------------------------------------------------
# FIND LANDSLIDE BOUNDING BOX
# --------------------------------------------------

ys, xs = np.where(mask == 1)

if len(xs) == 0:
    raise ValueError("Mask contains no landslide pixels!")


x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()


# Add padding around landslide
padding = 15

x_min = max(0, x_min - padding)
x_max = min(mask.shape[1], x_max + padding)

y_min = max(0, y_min - padding)
y_max = min(mask.shape[0], y_max + padding)


print("Landslide pixel bounding box:")
print(f"X: {xs.min()} → {xs.max()}")
print(f"Y: {ys.min()} → {ys.max()}")


# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 12))


# PRE
axes[0, 0].imshow(pre_rgb)
axes[0, 0].set_title("PRE-EVENT Sentinel-2")
axes[0, 0].axis("off")


# POST
axes[0, 1].imshow(post_rgb)
axes[0, 1].set_title("POST-EVENT Sentinel-2")
axes[0, 1].axis("off")


# POST + MASK
axes[1, 0].imshow(post_rgb)

masked = np.ma.masked_where(mask == 0, mask)

axes[1, 0].imshow(
    masked,
    cmap="Reds",
    alpha=0.65
)

axes[1, 0].set_title("POST-EVENT + LANDSLIDE MASK")
axes[1, 0].axis("off")


# ZOOMED AREA
axes[1, 1].imshow(
    post_rgb[y_min:y_max, x_min:x_max]
)

zoom_mask = mask[y_min:y_max, x_min:x_max]

masked_zoom = np.ma.masked_where(
    zoom_mask == 0,
    zoom_mask
)

axes[1, 1].imshow(
    masked_zoom,
    cmap="Reds",
    alpha=0.65
)

axes[1, 1].set_title("ZOOMED LANDSLIDE AREA")
axes[1, 1].axis("off")


plt.tight_layout()


OUTPUT = SAMPLE_DIR / "mask_overlay_verification.png"

plt.savefig(
    OUTPUT,
    dpi=150,
    bbox_inches="tight"
)

plt.show()


print(f"\n✓ Verification image saved:")
print(OUTPUT)

print("\n========== MASK VERIFICATION COMPLETE ==========\n")