import os
import numpy as np
import rasterio


# ============================================================
# LAND SENTRY — WAYANAD CHANGE ANALYSIS
# ============================================================

DATA_DIR = "data/processed/wayanad_event"

PRE_PATH = os.path.join(DATA_DIR, "pre.tif")
POST_PATH = os.path.join(DATA_DIR, "post.tif")


def load_image(path):

    with rasterio.open(path) as src:

        image = src.read().astype(np.float32)

        print()
        print(f"File: {os.path.basename(path)}")
        print(f"Shape: {image.shape}")
        print(f"CRS: {src.crs}")
        print(f"Resolution: {src.res}")

    return image


def calculate_ndvi(image):

    # Band order exported:
    # B2 = Blue  -> index 0
    # B3 = Green -> index 1
    # B4 = Red   -> index 2
    # B8 = NIR   -> index 3

    red = image[2]
    nir = image[3]

    denominator = nir + red

    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(nir),
        where=denominator != 0
    )

    return ndvi


def print_statistics(name, array):

    valid = array[np.isfinite(array)]

    print()
    print(name)
    print("-" * 50)

    print(f"Min     : {np.min(valid):.4f}")
    print(f"Max     : {np.max(valid):.4f}")
    print(f"Mean    : {np.mean(valid):.4f}")
    print(f"Median  : {np.median(valid):.4f}")

    print(f"5th pct : {np.percentile(valid, 5):.4f}")
    print(f"25th pct: {np.percentile(valid, 25):.4f}")
    print(f"75th pct: {np.percentile(valid, 75):.4f}")
    print(f"95th pct: {np.percentile(valid, 95):.4f}")


def main():

    print()
    print("=" * 60)
    print("LAND SENTRY — WAYANAD CHANGE ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    print("\nLoading PRE-event image...")
    pre = load_image(PRE_PATH)

    print("\nLoading POST-event image...")
    post = load_image(POST_PATH)

    # --------------------------------------------------------
    # Alignment check
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ALIGNMENT CHECK")
    print("=" * 60)

    if pre.shape == post.shape:

        print("✓ PRE and POST images have identical dimensions")

    else:

        print("✗ PRE and POST dimensions DO NOT match")

        print(f"PRE : {pre.shape}")
        print(f"POST: {post.shape}")

        return

    # --------------------------------------------------------
    # Calculate NDVI
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CALCULATING NDVI")
    print("=" * 60)

    pre_ndvi = calculate_ndvi(pre)
    post_ndvi = calculate_ndvi(post)

    # Vegetation change:
    # Positive value = vegetation decreased after event

    vegetation_loss = pre_ndvi - post_ndvi

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_statistics(
        "PRE-EVENT NDVI",
        pre_ndvi
    )

    print_statistics(
        "POST-EVENT NDVI",
        post_ndvi
    )

    print_statistics(
        "VEGETATION LOSS (PRE NDVI - POST NDVI)",
        vegetation_loss
    )

    print()
    print("=" * 60)
    print("CHANGE ANALYSIS COMPLETE")
    print("=" * 60)

    print()
    print("✓ Next: Use these statistics to choose")
    print("  scientifically reasonable candidate-mask thresholds.")


if __name__ == "__main__":
    main()