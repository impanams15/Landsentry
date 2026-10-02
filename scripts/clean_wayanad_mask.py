import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage


print("\n============================================================")
print("LAND SENTRY — WAYANAD MASK CLEANING")
print("============================================================")


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_DIR = "data/processed/wayanad_event"

PRE_PATH = os.path.join(INPUT_DIR, "pre.tif")
POST_PATH = os.path.join(INPUT_DIR, "post.tif")

OUTPUT_MASK = os.path.join(INPUT_DIR, "mask.tif")
OUTPUT_VIS = os.path.join(INPUT_DIR, "wayanad_clean_mask.png")


# ------------------------------------------------------------
# LOAD IMAGES
# ------------------------------------------------------------

print("\nLoading PRE-event image...")

with rasterio.open(PRE_PATH) as src:
    pre = src.read()
    profile = src.profile.copy()

print("✓ PRE image loaded")


print("\nLoading POST-event image...")

with rasterio.open(POST_PATH) as src:
    post = src.read()

print("✓ POST image loaded")


# ------------------------------------------------------------
# CALCULATE NDVI
# ------------------------------------------------------------

print("\nCalculating NDVI...")

# Sentinel-2 band order:
# B4 = Red
# B3 = Green
# B2 = Blue
# B8 = NIR

RED_PRE = pre[0].astype(np.float32)
NIR_PRE = pre[3].astype(np.float32)

RED_POST = post[0].astype(np.float32)
NIR_POST = post[3].astype(np.float32)


EPSILON = 1e-6

ndvi_pre = (
    (NIR_PRE - RED_PRE) /
    (NIR_PRE + RED_PRE + EPSILON)
)

ndvi_post = (
    (NIR_POST - RED_POST) /
    (NIR_POST + RED_POST + EPSILON)
)


# ------------------------------------------------------------
# VEGETATION LOSS
# ------------------------------------------------------------

vegetation_loss = ndvi_pre - ndvi_post

print("✓ Vegetation-loss map created")


# ------------------------------------------------------------
# CREATE CANDIDATE D
# ------------------------------------------------------------

print("\nCreating Candidate D...")

candidate_mask = (
    (vegetation_loss > 0.15) &
    (ndvi_post < 0.70)
)

candidate_mask = candidate_mask.astype(np.uint8)

print(
    f"✓ Candidate pixels before cleaning: "
    f"{np.sum(candidate_mask)}"
)


# ------------------------------------------------------------
# CONNECTED COMPONENT ANALYSIS
# ------------------------------------------------------------

print("\nFinding connected components...")

structure = np.ones((3, 3), dtype=np.uint8)

labeled_mask, num_features = ndimage.label(
    candidate_mask,
    structure=structure
)

print(f"✓ Connected components found: {num_features}")


# ------------------------------------------------------------
# FIND LARGEST COMPONENT
# ------------------------------------------------------------

if num_features == 0:
    raise RuntimeError(
        "No landslide candidate pixels found. "
        "Try adjusting the thresholds."
    )


component_sizes = np.bincount(labeled_mask.ravel())

# Ignore background component (label 0)
component_sizes[0] = 0

largest_component_label = component_sizes.argmax()

largest_component_pixels = component_sizes[
    largest_component_label
]

print(
    f"✓ Largest component label: "
    f"{largest_component_label}"
)

print(
    f"✓ Largest component pixels: "
    f"{largest_component_pixels}"
)


# ------------------------------------------------------------
# CREATE CLEAN MASK
# ------------------------------------------------------------

clean_mask = (
    labeled_mask == largest_component_label
).astype(np.uint8)


# ------------------------------------------------------------
# OPTIONAL SMALL MORPHOLOGICAL CLEANUP
# ------------------------------------------------------------

print("\nApplying morphological cleanup...")

clean_mask = ndimage.binary_closing(
    clean_mask,
    structure=np.ones((3, 3))
)

clean_mask = ndimage.binary_opening(
    clean_mask,
    structure=np.ones((3, 3))
)

clean_mask = clean_mask.astype(np.uint8)

print(
    f"✓ Final landslide pixels: "
    f"{np.sum(clean_mask)}"
)


# ------------------------------------------------------------
# SAVE MASK AS GEOTIFF
# ------------------------------------------------------------

print("\nSaving cleaned Wayanad mask...")

profile.update(
    count=1,
    dtype=rasterio.uint8,
    nodata=0
)

with rasterio.open(
    OUTPUT_MASK,
    "w",
    **profile
) as dst:

    dst.write(clean_mask, 1)


print(f"✓ Mask saved: {OUTPUT_MASK}")


# ------------------------------------------------------------
# CREATE VISUALIZATION
# ------------------------------------------------------------

print("\nCreating visualization...")


# RGB visualization from POST image

rgb = np.stack(
    [
        post[0],
        post[1],
        post[2]
    ],
    axis=-1
).astype(np.float32)


# Normalize RGB for display

for i in range(3):

    band = rgb[:, :, i]

    low = np.percentile(band, 2)
    high = np.percentile(band, 98)

    rgb[:, :, i] = np.clip(
        (band - low) / (high - low + EPSILON),
        0,
        1
    )


plt.figure(figsize=(15, 6))


# POST IMAGE

plt.subplot(1, 3, 1)

plt.imshow(rgb)

plt.title(
    "WAYANAD POST-EVENT IMAGE\n"
    "December 2024"
)

plt.axis("off")


# ORIGINAL CANDIDATE

plt.subplot(1, 3, 2)

plt.imshow(rgb)

plt.imshow(
    candidate_mask,
    cmap="Reds",
    alpha=0.5
)

plt.title(
    "ORIGINAL CANDIDATE D\n"
    f"Pixels: {np.sum(candidate_mask)}"
)

plt.axis("off")


# CLEAN MASK

plt.subplot(1, 3, 3)

plt.imshow(rgb)

plt.imshow(
    clean_mask,
    cmap="Reds",
    alpha=0.6
)

plt.title(
    "CLEANED WAYANAD MASK\n"
    f"Pixels: {np.sum(clean_mask)}"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    OUTPUT_VIS,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


print(f"✓ Visualization saved: {OUTPUT_VIS}")


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n============================================================")
print("WAYANAD MASK CLEANING COMPLETE")
print("============================================================")

print(f"\nOriginal candidate pixels : {np.sum(candidate_mask)}")
print(f"Final mask pixels         : {np.sum(clean_mask)}")

print("\nFiles ready:")

print(f"✓ {PRE_PATH}")
print(f"✓ {POST_PATH}")
print(f"✓ {OUTPUT_MASK}")

print(
    "\n🎉 WAYANAD BITEMPORAL EVENT DATASET READY!"
)