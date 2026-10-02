import os
import rasterio
import numpy as np

print("\n============================================================")
print("LAND SENTRY — WAYANAD DATASET VALIDATION")
print("============================================================\n")

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

DATA_DIR = "data/processed/wayanad_event"

PRE_PATH = os.path.join(DATA_DIR, "pre.tif")
POST_PATH = os.path.join(DATA_DIR, "post.tif")
MASK_PATH = os.path.join(DATA_DIR, "mask.tif")


# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

print("Checking required files...\n")

required_files = {
    "PRE-event image": PRE_PATH,
    "POST-event image": POST_PATH,
    "Ground-truth mask": MASK_PATH
}

all_files_exist = True

for name, path in required_files.items():
    if os.path.exists(path):
        print(f"✓ Found {name}: {path}")
    else:
        print(f"✗ MISSING {name}: {path}")
        all_files_exist = False

if not all_files_exist:
    print("\n❌ Validation stopped because required files are missing.")
    raise SystemExit


# ------------------------------------------------------------
# LOAD DATASETS
# ------------------------------------------------------------

print("\n============================================================")
print("LOADING DATASETS")
print("============================================================\n")

with rasterio.open(PRE_PATH) as src:
    pre_shape = (src.count, src.height, src.width)
    pre_crs = src.crs
    pre_transform = src.transform
    pre_dtype = src.dtypes
    pre_bounds = src.bounds

with rasterio.open(POST_PATH) as src:
    post_shape = (src.count, src.height, src.width)
    post_crs = src.crs
    post_transform = src.transform
    post_dtype = src.dtypes
    post_bounds = src.bounds

with rasterio.open(MASK_PATH) as src:
    mask = src.read(1)
    mask_shape = mask.shape
    mask_crs = src.crs
    mask_transform = src.transform
    mask_dtype = src.dtypes
    mask_bounds = src.bounds


# ------------------------------------------------------------
# DISPLAY METADATA
# ------------------------------------------------------------

print("PRE-EVENT IMAGE")
print("-" * 50)
print(f"Shape     : {pre_shape}")
print(f"CRS       : {pre_crs}")
print(f"Data type : {pre_dtype}")
print(f"Bounds    : {pre_bounds}")

print("\nPOST-EVENT IMAGE")
print("-" * 50)
print(f"Shape     : {post_shape}")
print(f"CRS       : {post_crs}")
print(f"Data type : {post_dtype}")
print(f"Bounds    : {post_bounds}")

print("\nGROUND-TRUTH MASK")
print("-" * 50)
print(f"Shape     : {mask_shape}")
print(f"CRS       : {mask_crs}")
print(f"Data type : {mask_dtype}")
print(f"Bounds    : {mask_bounds}")


# ------------------------------------------------------------
# ALIGNMENT CHECKS
# ------------------------------------------------------------

print("\n============================================================")
print("SPATIAL ALIGNMENT CHECK")
print("============================================================\n")

pre_hw = pre_shape[1:]
post_hw = post_shape[1:]

shape_ok = (
    pre_hw == post_hw
    and pre_hw == mask_shape
)

crs_ok = (
    pre_crs == post_crs
    and pre_crs == mask_crs
)

transform_ok = (
    pre_transform == post_transform
    and pre_transform == mask_transform
)

if shape_ok:
    print("✓ Dimensions match")
else:
    print("✗ Dimension mismatch")
    print(f"  PRE : {pre_hw}")
    print(f"  POST: {post_hw}")
    print(f"  MASK: {mask_shape}")

if crs_ok:
    print("✓ CRS matches")
else:
    print("✗ CRS mismatch")

if transform_ok:
    print("✓ Spatial transforms match")
else:
    print("✗ Spatial transform mismatch")


# ------------------------------------------------------------
# MASK VALIDATION
# ------------------------------------------------------------

print("\n============================================================")
print("MASK VALIDATION")
print("============================================================\n")

unique_values = np.unique(mask)

print(f"Unique mask values: {unique_values}")

landslide_pixels = int(np.sum(mask > 0))
background_pixels = int(np.sum(mask == 0))
total_pixels = mask.size

landslide_percentage = (
    landslide_pixels / total_pixels
) * 100

print(f"\nTotal pixels      : {total_pixels}")
print(f"Background pixels : {background_pixels}")
print(f"Landslide pixels  : {landslide_pixels}")
print(f"Landslide area %  : {landslide_percentage:.2f}%")

# Binary check
if set(unique_values).issubset({0, 1}):
    print("\n✓ Mask is binary (0 = background, 1 = landslide)")
else:
    print("\n⚠ Mask is not strictly binary")
    print("  Converting interpretation: values > 0 represent landslide")


# ------------------------------------------------------------
# FINAL RESULT
# ------------------------------------------------------------

print("\n============================================================")
print("FINAL VALIDATION RESULT")
print("============================================================\n")

if shape_ok and crs_ok and transform_ok and landslide_pixels > 0:
    print("🎉 WAYANAD DATASET VALIDATION SUCCESSFUL!")
    print("\nDataset is ready for ML preprocessing and patch generation.")
else:
    print("⚠ VALIDATION FOUND ISSUES.")
    print("Review the checks above before proceeding.")

print("\n============================================================\n")