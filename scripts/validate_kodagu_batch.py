from pathlib import Path
import rasterio
import numpy as np
import pandas as pd


# ============================================================
# LAND SENTRY - KODAGU DATASET VALIDATOR
# ============================================================

DATASET_DIR = Path("data/processed/kodagu_batch")

results = []

print("\n========== LAND SENTRY DATASET VALIDATION ==========\n")


# Find sample directories
sample_dirs = sorted(
    [
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir() and folder.name.startswith("sample_")
    ]
)

print(f"Samples found: {len(sample_dirs)}\n")


for sample_dir in sample_dirs:

    sample_id = sample_dir.name

    pre_path = sample_dir / "pre.tif"
    post_path = sample_dir / "post.tif"
    mask_path = sample_dir / "mask.tif"

    print(f"Checking {sample_id}...")

    errors = []

    try:

        # ----------------------------------------------------
        # CHECK FILES
        # ----------------------------------------------------

        for file_path in [pre_path, post_path, mask_path]:

            if not file_path.exists():

                errors.append(
                    f"Missing file: {file_path.name}"
                )

        if errors:
            raise RuntimeError("; ".join(errors))


        # ----------------------------------------------------
        # READ RASTERS
        # ----------------------------------------------------

        with rasterio.open(pre_path) as src:
            pre = src.read()
            pre_shape = src.shape
            pre_crs = src.crs
            pre_transform = src.transform

        with rasterio.open(post_path) as src:
            post = src.read()
            post_shape = src.shape
            post_crs = src.crs
            post_transform = src.transform

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            mask_shape = src.shape
            mask_crs = src.crs
            mask_transform = src.transform


        # ----------------------------------------------------
        # CHECK BAND COUNTS
        # ----------------------------------------------------

        if pre.shape[0] != 4:

            errors.append(
                f"PRE has {pre.shape[0]} bands instead of 4"
            )

        if post.shape[0] != 4:

            errors.append(
                f"POST has {post.shape[0]} bands instead of 4"
            )


        # ----------------------------------------------------
        # CHECK DIMENSIONS
        # ----------------------------------------------------

        if pre_shape != post_shape:

            errors.append(
                "PRE and POST dimensions do not match"
            )

        if pre_shape != mask_shape:

            errors.append(
                "PRE and MASK dimensions do not match"
            )


        # ----------------------------------------------------
        # CHECK CRS
        # ----------------------------------------------------

        if pre_crs != post_crs:

            errors.append(
                "PRE and POST CRS do not match"
            )

        if pre_crs != mask_crs:

            errors.append(
                "PRE and MASK CRS do not match"
            )


        # ----------------------------------------------------
        # CHECK TRANSFORM / ALIGNMENT
        # ----------------------------------------------------

        if pre_transform != post_transform:

            errors.append(
                "PRE and POST transforms do not match"
            )

        if pre_transform != mask_transform:

            errors.append(
                "PRE and MASK transforms do not match"
            )


        # ----------------------------------------------------
        # CHECK MASK VALUES
        # ----------------------------------------------------

        unique_values = np.unique(mask)

        if not set(unique_values).issubset({0, 1}):

            errors.append(
                f"Invalid mask values: {unique_values}"
            )

        landslide_pixels = int(np.sum(mask == 1))

        if landslide_pixels == 0:

            errors.append(
                "Mask contains ZERO landslide pixels"
            )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if errors:

            status = "FAILED"

            print(f"  ❌ FAILED")

            for error in errors:

                print(f"     - {error}")

        else:

            status = "SUCCESS"

            print(
                f"  ✓ SUCCESS | "
                f"Shape: {pre_shape} | "
                f"Landslide pixels: {landslide_pixels}"
            )


        results.append({
            "sample_id": sample_id,
            "status": status,
            "shape": str(pre_shape),
            "landslide_pixels": landslide_pixels,
            "errors": "; ".join(errors)
        })


    except Exception as error:

        print(f"  ❌ ERROR: {error}")

        results.append({
            "sample_id": sample_id,
            "status": "ERROR",
            "shape": "",
            "landslide_pixels": 0,
            "errors": str(error)
        })


# ============================================================
# SAVE VALIDATION REPORT
# ============================================================

results_df = pd.DataFrame(results)

report_path = DATASET_DIR / "validation_report.csv"

results_df.to_csv(
    report_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========== VALIDATION SUMMARY ==========\n")

print(results_df["status"].value_counts())

print(f"\nValidation report saved to:")
print(report_path)

print("\n========================================\n")