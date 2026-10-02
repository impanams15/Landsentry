import os
import json
import numpy as np
import rasterio


# ============================================================
# LAND SENTRY — WAYANAD ML PATCH GENERATION
# ============================================================

print("\n============================================================")
print("LAND SENTRY — WAYANAD ML PATCH GENERATION")
print("============================================================")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "data/processed/wayanad_event"
OUTPUT_DIR = "data/ml/wayanad"

PATCH_SIZE = 64
STRIDE = 32   # Overlapping patches


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD PRE, POST AND MASK
# ============================================================

print("\nLoading Wayanad event data...")

pre_path = os.path.join(INPUT_DIR, "pre.tif")
post_path = os.path.join(INPUT_DIR, "post.tif")
mask_path = os.path.join(INPUT_DIR, "mask.tif")


with rasterio.open(pre_path) as src:
    pre = src.read().astype(np.float32)

with rasterio.open(post_path) as src:
    post = src.read().astype(np.float32)

with rasterio.open(mask_path) as src:
    mask = src.read(1).astype(np.uint8)


print("✓ PRE image loaded")
print(f"  Shape: {pre.shape}")

print("✓ POST image loaded")
print(f"  Shape: {post.shape}")

print("✓ MASK loaded")
print(f"  Shape: {mask.shape}")


# ============================================================
# VALIDATE ALIGNMENT
# ============================================================

if pre.shape[1:] != post.shape[1:]:
    raise ValueError("PRE and POST dimensions do not match!")

if pre.shape[1:] != mask.shape:
    raise ValueError("Images and mask dimensions do not match!")

print("\n✓ Spatial alignment verified")


# ============================================================
# COMBINE PRE + POST
# ============================================================

print("\nCreating 8-channel bitemporal input...")

image = np.concatenate(
    [pre, post],
    axis=0
)

print(f"✓ Combined image shape: {image.shape}")


# ============================================================
# PATCH GENERATION
# ============================================================

print("\n============================================================")
print("GENERATING WAYANAD PATCHES")
print("============================================================")


images = []
masks = []

height, width = mask.shape

for y in range(0, height - PATCH_SIZE + 1, STRIDE):

    for x in range(0, width - PATCH_SIZE + 1, STRIDE):

        image_patch = image[
            :,
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        mask_patch = mask[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        images.append(image_patch)
        masks.append(mask_patch)


images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.uint8)


print(f"\n✓ Total patches generated: {len(images)}")
print(f"✓ Images shape: {images.shape}")
print(f"✓ Masks shape : {masks.shape}")


# ============================================================
# PATCH STATISTICS
# ============================================================

positive_patches = np.sum(
    np.any(masks == 1, axis=(1, 2))
)

background_patches = len(masks) - positive_patches


print("\n============================================================")
print("PATCH STATISTICS")
print("============================================================")

print(f"Positive patches   : {positive_patches}")
print(f"Background patches : {background_patches}")
print(f"Total patches      : {len(masks)}")


# ============================================================
# SAVE DATASET
# ============================================================

print("\n============================================================")
print("SAVING WAYANAD ML DATASET")
print("============================================================")


images_path = os.path.join(
    OUTPUT_DIR,
    "images.npy"
)

masks_path = os.path.join(
    OUTPUT_DIR,
    "masks.npy"
)

metadata_path = os.path.join(
    OUTPUT_DIR,
    "metadata.json"
)


np.save(images_path, images)
np.save(masks_path, masks)


metadata = {
    "event": "Wayanad Mundakkai Landslide 2024",
    "input_channels": 8,
    "bands": [
        "PRE_B4",
        "PRE_B3",
        "PRE_B2",
        "PRE_B8",
        "POST_B4",
        "POST_B3",
        "POST_B2",
        "POST_B8"
    ],
    "patch_size": PATCH_SIZE,
    "stride": STRIDE,
    "total_patches": int(len(masks)),
    "positive_patches": int(positive_patches),
    "background_patches": int(background_patches),
    "purpose": "Independent cross-event evaluation dataset"
}


with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)


print(f"\n✓ Images saved: {images_path}")
print(f"✓ Masks saved : {masks_path}")
print(f"✓ Metadata saved: {metadata_path}")


print("\n============================================================")
print("🎉 WAYANAD ML PATCH DATASET READY!")
print("============================================================")

print("\nThis dataset will be used for:")
print("Kodagu → Wayanad cross-event generalization testing\n")