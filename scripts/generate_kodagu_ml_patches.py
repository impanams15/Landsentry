import os
import json
import random

import numpy as np
import rasterio


# ============================================================
# LAND SENTRY — KODAGU SPLIT-WISE PATCH GENERATION
# ============================================================

print("\n============================================================")
print("LAND SENTRY — KODAGU PATCH GENERATION")
print("============================================================")


# ============================================================
# CONFIGURATION
# ============================================================

KODAGU_DIR = "data/processed/kodagu_batch"

ML_DIR = "data/ml/kodagu"

SPLITS_FILE = os.path.join(
    ML_DIR,
    "sample_splits.json"
)

PATCH_SIZE = 64

BACKGROUND_RATIO = 2

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(image):
    """
    Normalize Sentinel-2 reflectance values.

    Values are clipped to [0, 10000]
    and scaled to [0, 1].
    """

    image = np.clip(image, 0, 10000)

    return image / 10000.0


# ============================================================
# LOAD SAMPLE
# ============================================================

def load_sample(sample_name):

    sample_dir = os.path.join(
        KODAGU_DIR,
        sample_name
    )

    pre_path = os.path.join(
        sample_dir,
        "pre.tif"
    )

    post_path = os.path.join(
        sample_dir,
        "post.tif"
    )

    mask_path = os.path.join(
        sample_dir,
        "mask.tif"
    )

    with rasterio.open(pre_path) as src:
        pre = src.read().astype(np.float32)

    with rasterio.open(post_path) as src:
        post = src.read().astype(np.float32)

    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)

    return pre, post, mask


# ============================================================
# PAD SAMPLE
# ============================================================

def pad_sample(image, mask, patch_size):

    _, height, width = image.shape

    padded_height = (
        int(np.ceil(height / patch_size))
        * patch_size
    )

    padded_width = (
        int(np.ceil(width / patch_size))
        * patch_size
    )

    pad_height = padded_height - height

    pad_width = padded_width - width


    image = np.pad(
        image,
        (
            (0, 0),
            (0, pad_height),
            (0, pad_width)
        ),
        mode="reflect"
    )


    mask = np.pad(
        mask,
        (
            (0, pad_height),
            (0, pad_width)
        ),
        mode="constant",
        constant_values=0
    )

    return image, mask


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
                (image_patch, mask_patch)
            )

    return patches


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name, sample_names):

    print("\n============================================================")
    print(f"PROCESSING {split_name.upper()} SPLIT")
    print("============================================================")

    print(
        f"\nSamples in split: "
        f"{len(sample_names)}"
    )

    positive_patches = []

    background_patches = []


    # --------------------------------------------------------
    # PROCESS EACH ORIGINAL LANDSLIDE SAMPLE
    # --------------------------------------------------------

    for index, sample_name in enumerate(
        sample_names,
        start=1
    ):

        print(
            f"Processing {sample_name} "
            f"({index}/{len(sample_names)})"
        )


        pre, post, mask = load_sample(
            sample_name
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

        image, mask = pad_sample(
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


        # Separate positive/background patches

        for image_patch, mask_patch in patches:

            landslide_pixels = np.sum(
                mask_patch > 0
            )


            if landslide_pixels > 0:

                positive_patches.append(
                    (image_patch, mask_patch)
                )

            else:

                background_patches.append(
                    (image_patch, mask_patch)
                )


    # ========================================================
    # BALANCE BACKGROUND PATCHES
    # ========================================================

    max_background = (
        len(positive_patches)
        * BACKGROUND_RATIO
    )


    if len(background_patches) > max_background:

        background_patches = random.sample(
            background_patches,
            max_background
        )


    # ========================================================
    # COMBINE + SHUFFLE
    # ========================================================

    all_patches = (
        positive_patches
        + background_patches
    )


    random.shuffle(all_patches)


    # ========================================================
    # SAVE SPLIT
    # ========================================================

    split_dir = os.path.join(
        ML_DIR,
        split_name
    )

    os.makedirs(
        split_dir,
        exist_ok=True
    )


    images = np.stack(
        [patch[0] for patch in all_patches]
    )


    masks = np.stack(
        [patch[1] for patch in all_patches]
    )


    images_path = os.path.join(
        split_dir,
        "images.npy"
    )


    masks_path = os.path.join(
        split_dir,
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


    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "split": split_name,

        "samples": len(sample_names),

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

        "images_shape": list(
            images.shape
        ),

        "masks_shape": list(
            masks.shape
        )

    }


    metadata_path = os.path.join(
        split_dir,
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


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n✓ SPLIT COMPLETE")

    print(
        f"Positive patches   : "
        f"{len(positive_patches)}"
    )

    print(
        f"Background patches : "
        f"{len(background_patches)}"
    )

    print(
        f"Total patches      : "
        f"{len(all_patches)}"
    )

    print(
        f"Images shape       : "
        f"{images.shape}"
    )

    print(
        f"Masks shape        : "
        f"{masks.shape}"
    )


# ============================================================
# LOAD SAMPLE SPLITS
# ============================================================

print("\nLoading sample splits...")

with open(
    SPLITS_FILE,
    "r"
) as file:

    splits = json.load(file)


print("✓ Sample splits loaded")


# ============================================================
# GENERATE PATCHES FOR EACH SPLIT
# ============================================================

process_split(
    "train",
    splits["train"]
)


process_split(
    "validation",
    splits["validation"]
)


process_split(
    "test",
    splits["test"]
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("🎉 ALL KODAGU ML PATCHES GENERATED SUCCESSFULLY!")
print("============================================================")

print("\nDataset location:")

print(ML_DIR)

print("\n============================================================\n")