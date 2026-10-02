import os
import numpy as np
import rasterio


print("\n============================================================")
print("LAND SENTRY — ML DATA INSPECTION")
print("============================================================")


def inspect_raster(name, path):
    print("\n------------------------------------------------------------")
    print(name)
    print("------------------------------------------------------------")

    with rasterio.open(path) as src:

        data = src.read()

        print(f"Path       : {path}")
        print(f"Shape      : {data.shape}")
        print(f"Bands      : {src.count}")
        print(f"Data type  : {data.dtype}")
        print(f"CRS        : {src.crs}")

        for i in range(data.shape[0]):

            band = data[i]

            print(f"\nBand {i + 1}")
            print(f"  Min  : {np.nanmin(band):.4f}")
            print(f"  Max  : {np.nanmax(band):.4f}")
            print(f"  Mean : {np.nanmean(band):.4f}")
            print(f"  Std  : {np.nanstd(band):.4f}")


# ============================================================
# KODAGU SAMPLE
# ============================================================

KODAGU_DIR = "data/processed/kodagu_batch/sample_001"

inspect_raster(
    "KODAGU — PRE EVENT",
    os.path.join(KODAGU_DIR, "pre.tif")
)

inspect_raster(
    "KODAGU — POST EVENT",
    os.path.join(KODAGU_DIR, "post.tif")
)


# ============================================================
# WAYANAD EVENT
# ============================================================

WAYANAD_DIR = "data/processed/wayanad_event"

inspect_raster(
    "WAYANAD — PRE EVENT",
    os.path.join(WAYANAD_DIR, "pre.tif")
)

inspect_raster(
    "WAYANAD — POST EVENT",
    os.path.join(WAYANAD_DIR, "post.tif")
)


print("\n============================================================")
print("INSPECTION COMPLETE")
print("============================================================\n")