import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from scipy import ndimage


# ============================================================
# LAND SENTRY — WAYANAD CANDIDATE MASK GENERATION
# ============================================================

DATA_DIR = "data/processed/wayanad_event"

PRE_PATH = os.path.join(DATA_DIR, "pre.tif")
POST_PATH = os.path.join(DATA_DIR, "post.tif")

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "wayanad_candidate_masks.png"
)


# ------------------------------------------------------------
# Load raster
# ------------------------------------------------------------

def load_raster(path):

    with rasterio.open(path) as src:

        image = src.read().astype(np.float32)
        profile = src.profile.copy()

    return image, profile


# ------------------------------------------------------------
# Calculate NDVI
# ------------------------------------------------------------

def calculate_ndvi(image):

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


# ------------------------------------------------------------
# Convert Sentinel-2 to RGB for visualization
# ------------------------------------------------------------

def create_rgb(image):

    # Band order:
    # B2 = Blue
    # B3 = Green
    # B4 = Red
    # B8 = NIR

    rgb = np.stack(
        [
            image[2],  # Red
            image[1],  # Green
            image[0]   # Blue
        ],
        axis=-1
    )

    # Percentile stretch
    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)

    rgb = np.clip(
        (rgb - low) / (high - low),
        0,
        1
    )

    return rgb


# ------------------------------------------------------------
# Remove tiny connected regions
# ------------------------------------------------------------

def remove_small_regions(mask, min_pixels=20):

    labeled, num_features = ndimage.label(mask)

    if num_features == 0:
        return mask.astype(np.uint8)

    region_sizes = np.bincount(
        labeled.ravel()
    )

    keep = region_sizes >= min_pixels

    keep[0] = False

    cleaned_mask = keep[labeled]

    return cleaned_mask.astype(np.uint8)


# ------------------------------------------------------------
# Create candidate mask
# ------------------------------------------------------------

def create_mask(
    vegetation_loss,
    post_ndvi,
    loss_threshold,
    post_ndvi_threshold
):

    # Main candidate condition
    mask = (
        (vegetation_loss > loss_threshold)
        &
        (post_ndvi < post_ndvi_threshold)
    )

    # Morphological cleanup
    mask = ndimage.binary_opening(
        mask,
        structure=np.ones((3, 3))
    )

    mask = ndimage.binary_closing(
        mask,
        structure=np.ones((3, 3))
    )

    # Remove tiny regions
    mask = remove_small_regions(
        mask,
        min_pixels=15
    )

    return mask


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("LAND SENTRY — WAYANAD CANDIDATE MASK GENERATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    print("\nLoading imagery...")

    pre, profile = load_raster(PRE_PATH)
    post, _ = load_raster(POST_PATH)

    print("✓ PRE loaded")
    print("✓ POST loaded")

    # --------------------------------------------------------
    # Calculate NDVI change
    # --------------------------------------------------------

    print("\nCalculating NDVI...")

    pre_ndvi = calculate_ndvi(pre)
    post_ndvi = calculate_ndvi(post)

    vegetation_loss = pre_ndvi - post_ndvi

    print("✓ NDVI calculated")
    print("✓ Vegetation-loss map calculated")

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    pre_rgb = create_rgb(pre)
    post_rgb = create_rgb(post)

    # --------------------------------------------------------
    # Candidate configurations
    # --------------------------------------------------------

    candidates = [

        {
            "name": "Candidate A\nLoss > 0.05\nPost NDVI < 0.60",
            "loss": 0.05,
            "ndvi": 0.60
        },

        {
            "name": "Candidate B\nLoss > 0.08\nPost NDVI < 0.65",
            "loss": 0.08,
            "ndvi": 0.65
        },

        {
            "name": "Candidate C\nLoss > 0.10\nPost NDVI < 0.70",
            "loss": 0.10,
            "ndvi": 0.70
        },

        {
            "name": "Candidate D\nLoss > 0.15\nPost NDVI < 0.70",
            "loss": 0.15,
            "ndvi": 0.70
        }

    ]

    masks = []

    print()
    print("=" * 60)
    print("GENERATING CANDIDATE MASKS")
    print("=" * 60)

    for candidate in candidates:

        mask = create_mask(
            vegetation_loss,
            post_ndvi,
            candidate["loss"],
            candidate["ndvi"]
        )

        masks.append(mask)

        print()
        print(candidate["name"].replace("\n", " | "))
        print(
            f"Landslide candidate pixels: {np.sum(mask)}"
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    print("\nCreating visualization...")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11)
    )

    # PRE
    axes[0, 0].imshow(pre_rgb)
    axes[0, 0].set_title(
        "PRE-EVENT\nJan–Apr 2024"
    )
    axes[0, 0].axis("off")

    # POST
    axes[0, 1].imshow(post_rgb)
    axes[0, 1].set_title(
        "POST-EVENT\nDecember 2024"
    )
    axes[0, 1].axis("off")

    # Vegetation loss
    change_plot = axes[0, 2].imshow(
        vegetation_loss,
        cmap="RdYlGn_r",
        vmin=-0.2,
        vmax=0.3
    )

    axes[0, 2].set_title(
        "VEGETATION LOSS\nPRE NDVI − POST NDVI"
    )

    axes[0, 2].axis("off")

    fig.colorbar(
        change_plot,
        ax=axes[0, 2],
        fraction=0.046
    )

    # Candidate masks
    for index, candidate in enumerate(candidates):

        row = 1
        col = index

        if index >= 3:
            # Put fourth candidate separately
            break

        axes[row, col].imshow(post_rgb)

        mask = masks[index]

        axes[row, col].imshow(
            mask,
            cmap="Reds",
            alpha=0.6
        )

        axes[row, col].set_title(
            candidate["name"]
            + f"\nPixels: {np.sum(mask)}"
        )

        axes[row, col].axis("off")

    # Replace last panel with Candidate D
    axes[1, 2].imshow(post_rgb)

    axes[1, 2].imshow(
        masks[3],
        cmap="Reds",
        alpha=0.6
    )

    axes[1, 2].set_title(
        candidates[3]["name"]
        + f"\nPixels: {np.sum(masks[3])}"
    )

    axes[1, 2].axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"✓ Visualization saved to:")
    print(f"  {OUTPUT_PATH}")

    plt.show()

    print()
    print("=" * 60)
    print("CANDIDATE MASK GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()