import os
import json
import random

import numpy as np
import rasterio


# ============================================================
# LAND SENTRY — ML DATASET PREPARATION
# ============================================================

print("\n============================================================")
print("LAND SENTRY — ML DATASET PREPARATION")
print("============================================================")


# ============================================================
# CONFIGURATION
# ============================================================

KODAGU_DIR = "data/processed/kodagu_batch"

OUTPUT_DIR = "data/ml"

PATCH_SIZE = 64

BACKGROUND_RATIO = 2

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

KODAGU_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "kodagu"
)

os.makedirs(KODAGU_OUTPUT, exist_ok=True)

print(f"\n✓ Output directory ready: {KODAGU_OUTPUT}")


# ============================================================
# LOAD RASTER SAMPLE
# ============================================================

def load_sample(sample_dir):

    pre_path = os.path.join(sample_dir, "pre.tif")
    post_path = os.path.join(sample_dir, "post.tif")
    mask_path = os.path.join(sample_dir, "mask.tif")

    with rasterio.open(pre_path) as src:
        pre = src.read().astype(np.float32)

    with rasterio.open(post_path) as src:
        post = src.read().astype(np.float32)

    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)

    return pre, post, mask


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(image):

    """
    Sentinel-2 reflectance values are generally scaled
    approximately between 0 and 10000.

    We clip extreme values and scale to 0–1.
    """

    image = np.clip(image, 0, 10000)

    return image / 10000.0


# ============================================================
# PAD IMAGE
# ============================================================

def pad_to_patch_size(image, mask, patch_size):

    channels, height, width = image.shape

    padded_height = (
        int(np.ceil(height / patch_size))
        * patch_size
    )

    padded_width = (
        int(np.ceil(width / patch_size))
        * patch_size
    )

    pad_h = padded_height - height
    pad_w = padded_width - width

    image_padded = np.pad(
        image,
        (
            (0, 0),
            (0, pad_h),
            (0, pad_w)
        ),
        mode="reflect"
    )

    mask_padded = np.pad(
        mask,
        (
            (0, pad_h),
            (0, pad_w)
        ),
        mode="constant",
        constant_values=0
    )

    return image_padded, mask_padded


# ============================================================
# EXTRACT PATCHES
# ============================================================

def extract_patches(image, mask, patch_size):

    patches = []

    _, height, width = image.shape

    for y in range(0, height, patch_size):

        for x in range(0, width, patch_size):

            image_patch = image[
                :,
                y:y + patch_size,
                x:x + patch_size
            ]

            mask_patch = mask[
                y:y + patch_size,
                x:x + patch_size
            ]

            patches.append(
                (
                    image_patch,
                    mask_patch
                )
            )

    return patches


# ============================================================
# PROCESS KODAGU DATASET
# ============================================================

print("\n============================================================")
print("PROCESSING KODAGU DATASET")
print("============================================================")


sample_dirs = []

for name in sorted(os.listdir(KODAGU_DIR)):

    sample_path = os.path.join(
        KODAGU_DIR,
        name
    )

    if (
        os.path.isdir(sample_path)
        and name.startswith("sample_")
    ):

        required_files = [
            "pre.tif",
            "post.tif",
            "mask.tif"
        ]

        files_ok = all(
            os.path.exists(
                os.path.join(sample_path, f)
            )
            for f in required_files
        )

        if files_ok:
            sample_dirs.append(sample_path)


print(
    f"\n✓ Valid Kodagu samples found: "
    f"{len(sample_dirs)}"
)


positive_patches = []
background_patches = []


for index, sample_dir in enumerate(
    sample_dirs,
    start=1
):

    sample_name = os.path.basename(
        sample_dir
    )

    print(
        f"\nProcessing {sample_name} "
        f"({index}/{len(sample_dirs)})"
    )

    # Load
    pre, post, mask = load_sample(
        sample_dir
    )

    # Normalize
    pre = normalize_image(pre)
    post = normalize_image(post)

    # Stack PRE + POST
    image = np.concatenate(
        [pre, post],
        axis=0
    )

    # Pad
    image, mask = pad_to_patch_size(
        image,
        mask,
        PATCH_SIZE
    )

    # Extract
    patches = extract_patches(
        image,
        mask,
        PATCH_SIZE
    )

    # Separate positive and background
    for image_patch, mask_patch in patches:

        landslide_pixels = np.sum(
            mask_patch > 0
        )

        if landslide_pixels > 0:

            positive_patches.append(
                (
                    image_patch,
                    mask_patch
                )
            )

        else:

            background_patches.append(
                (
                    image_patch,
                    mask_patch
                )
            )


# ============================================================
# BALANCE BACKGROUND PATCHES
# ============================================================

print("\n============================================================")
print("BALANCING PATCHES")
print("============================================================")

print(
    f"\nPositive patches found: "
    f"{len(positive_patches)}"
)

print(
    f"Background patches found: "
    f"{len(background_patches)}"
)


max_background = (
    len(positive_patches)
    * BACKGROUND_RATIO
)


if len(background_patches) > max_background:

    background_patches = random.sample(
        background_patches,
        max_background
    )


print(
    f"\nBackground patches retained: "
    f"{len(background_patches)}"
)


# ============================================================
# COMBINE DATA
# ============================================================

all_patches = (
    positive_patches
    + background_patches
)

random.shuffle(all_patches)


print(
    f"\nTotal ML patches: "
    f"{len(all_patches)}"
)


# ============================================================
# SAVE DATA
# ============================================================

print("\n============================================================")
print("SAVING ML DATASET")
print("============================================================")


images = np.stack(
    [patch[0] for patch in all_patches]
)

masks = np.stack(
    [patch[1] for patch in all_patches]
)


images_path = os.path.join(
    KODAGU_OUTPUT,
    "images.npy"
)

masks_path = os.path.join(
    KODAGU_OUTPUT,
    "masks.npy"
)


np.save(
    images_path,
    images
)

np.save(
    masks_path,
    masks
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "dataset": "Kodagu",

    "samples": len(sample_dirs),

    "patch_size": PATCH_SIZE,

    "input_channels": 8,

    "positive_patches": len(
        positive_patches
    ),

    "background_patches": len(
        background_patches
    ),

    "total_patches": len(
        all_patches
    ),

    "normalization": (
        "clip_0_10000_divide_10000"
    )
}


metadata_path = os.path.join(
    KODAGU_OUTPUT,
    "metadata.json"
)


with open(
    metadata_path,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("KODAGU ML DATASET READY")
print("============================================================")

print(f"\nSamples processed : {len(sample_dirs)}")

print(f"Positive patches  : {len(positive_patches)}")

print(
    f"Background patches: "
    f"{len(background_patches)}"
)

print(f"Total patches     : {len(all_patches)}")

print(f"\nImages shape: {images.shape}")

print(f"Masks shape : {masks.shape}")

print(
    f"\n✓ Images saved: "
    f"{images_path}"
)

print(
    f"✓ Masks saved: "
    f"{masks_path}"
)

print(
    f"✓ Metadata saved: "
    f"{metadata_path}"
)

print("\n🎉 KODAGU ML DATASET PREPARATION COMPLETE!")

print("\n============================================================\n")