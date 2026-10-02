import ee
import geopandas as gpd
import pandas as pd

from pathlib import Path
import requests
import time
import json
import traceback


# ============================================================
# LAND SENTRY - KODAGU BITEMPORAL DATASET BUILDER
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "landsentry-507318"

INPUT_FILE = Path(
    "data/clean/kodagu/kodagu_landslides_clean.geojson"
)

OUTPUT_DIR = Path(
    "data/processed/kodagu_batch"
)

LOG_FILE = OUTPUT_DIR / "dataset_log.csv"


# ============================================================
# DATASET SETTINGS
# ============================================================

# Minimum estimated Sentinel-2 pixels
MIN_ESTIMATED_PIXELS = 25

# None = process ALL eligible samples
MAX_SAMPLES = None

# Context around each landslide
BUFFER_METERS = 300

# Sentinel-2 bands
BANDS = ["B4", "B3", "B2", "B8"]

# ------------------------------------------------------------
# PRE-EVENT PERIOD
# ------------------------------------------------------------

PRE_START = "2018-01-01"
PRE_END = "2018-04-30"


# ------------------------------------------------------------
# POST-EVENT PERIOD
# ------------------------------------------------------------

POST_START = "2018-09-01"
POST_END = "2018-12-31"


# ------------------------------------------------------------
# CLOUD FILTER
# ------------------------------------------------------------

MAX_CLOUD_PERCENTAGE = 30


# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

print("\n========== LAND SENTRY DATASET BUILDER ==========\n")

ee.Initialize(project=PROJECT_ID)

print("✓ Google Earth Engine initialized")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(f"✓ Output directory ready: {OUTPUT_DIR}")


# ============================================================
# LOAD KODAGU INVENTORY
# ============================================================

print("\nLoading Kodagu landslide inventory...")

gdf = gpd.read_file(INPUT_FILE)

print(f"✓ Total polygons loaded: {len(gdf)}")


# ============================================================
# CALCULATE POLYGON AREAS
# ============================================================

print("\nCalculating polygon areas...")

# Kodagu lies approximately in UTM Zone 43N
gdf_projected = gdf.to_crs("EPSG:32643")

# Calculate area in square metres
gdf["area_m2"] = gdf_projected.geometry.area

# Sentinel-2 resolution = approximately 10m × 10m
# Therefore one pixel ≈ 100 m²
gdf["estimated_pixels"] = (
    gdf["area_m2"] / 100
)


# ============================================================
# FILTER SMALL LANDSLIDES
# ============================================================

usable_gdf = gdf[
    gdf["estimated_pixels"] >= MIN_ESTIMATED_PIXELS
].copy()

print(
    f"✓ Samples with ≥ {MIN_ESTIMATED_PIXELS} "
    f"estimated pixels: {len(usable_gdf)}"
)


# ============================================================
# OPTIONAL SAMPLE LIMIT
# ============================================================

if MAX_SAMPLES is not None:

    usable_gdf = usable_gdf.head(MAX_SAMPLES)

    print(
        f"✓ TEST MODE: Processing "
        f"{len(usable_gdf)} samples"
    )

else:

    print(
        f"✓ FULL MODE: Processing "
        f"{len(usable_gdf)} samples"
    )


# ============================================================
# SENTINEL-2 CLOUD MASKING
# ============================================================

def mask_s2_clouds(image):

    scl = image.select("SCL")

    # Remove problematic pixels
    mask = (
        scl.neq(3)       # Cloud shadow
        .And(scl.neq(8)) # Medium probability cloud
        .And(scl.neq(9)) # High probability cloud
        .And(scl.neq(10))# Cirrus
    )

    return image.updateMask(mask)


# ============================================================
# DOWNLOAD IMAGE FUNCTION
# ============================================================

def download_image(image, region, output_path):

    """
    Downloads an Earth Engine image as GeoTIFF.
    """

    url = image.getDownloadURL({
        "region": region,
        "scale": 10,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF"
    })

    response = requests.get(
        url,
        timeout=180
    )

    response.raise_for_status()

    with open(output_path, "wb") as file:
        file.write(response.content)


# ============================================================
# CHECK IF SAMPLE IS COMPLETE
# ============================================================

def sample_is_complete(sample_dir):

    """
    A sample is considered complete only if all required
    files exist.
    """

    required_files = [

        sample_dir / "pre.tif",

        sample_dir / "post.tif",

        sample_dir / "mask.tif",

        sample_dir / "metadata.json"

    ]

    return all(
        file.exists()
        for file in required_files
    )


# ============================================================
# PROCESS ONE LANDSLIDE
# ============================================================

def process_sample(index, row):

    sample_id = f"sample_{index:03d}"

    sample_dir = OUTPUT_DIR / sample_id


    # ========================================================
    # RESUME SUPPORT
    # ========================================================

    if sample_dir.exists() and sample_is_complete(sample_dir):

        print(
            f"\n✓ {sample_id} already complete — skipping"
        )

        return {

            "sample_id": sample_id,

            "name": str(
                row.get("Name", "Unknown")
            ),

            "status": "SKIPPED",

            "pre_images": None,

            "post_images": None,

            "estimated_pixels": row[
                "estimated_pixels"
            ],

            "error": ""

        }


    # Create directory if needed
    sample_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    print("\n" + "=" * 60)

    print(f"PROCESSING {sample_id}")

    print(
        f"Name: "
        f"{row.get('Name', 'Unknown')}"
    )

    print(
        f"Estimated pixels: "
        f"{row['estimated_pixels']:.2f}"
    )

    print("=" * 60)


    try:

        # ====================================================
        # CONVERT GEOMETRY TO EARTH ENGINE
        # ====================================================

        geometry = row.geometry

        ee_geometry = ee.Geometry(
            geometry.__geo_interface__
        )


        # ====================================================
        # CREATE AOI
        # ====================================================

        aoi = (
            ee_geometry
            .buffer(BUFFER_METERS)
            .bounds()
        )


        # ====================================================
        # PRE-EVENT COLLECTION
        # ====================================================

        pre_collection = (

            ee.ImageCollection(
                "COPERNICUS/S2_SR_HARMONIZED"
            )

            .filterBounds(aoi)

            .filterDate(
                PRE_START,
                PRE_END
            )

            .filter(
                ee.Filter.lt(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    MAX_CLOUD_PERCENTAGE
                )
            )

            .map(mask_s2_clouds)

        )


        pre_count = (
            pre_collection
            .size()
            .getInfo()
        )


        # ====================================================
        # POST-EVENT COLLECTION
        # ====================================================

        post_collection = (

            ee.ImageCollection(
                "COPERNICUS/S2_SR_HARMONIZED"
            )

            .filterBounds(aoi)

            .filterDate(
                POST_START,
                POST_END
            )

            .filter(
                ee.Filter.lt(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    MAX_CLOUD_PERCENTAGE
                )
            )

            .map(mask_s2_clouds)

        )


        post_count = (
            post_collection
            .size()
            .getInfo()
        )


        print(f"PRE images: {pre_count}")

        print(f"POST images: {post_count}")


        # ====================================================
        # CHECK IMAGE AVAILABILITY
        # ====================================================

        if pre_count == 0:

            raise RuntimeError(
                "No suitable PRE-event images found"
            )


        if post_count == 0:

            raise RuntimeError(
                "No suitable POST-event images found"
            )


        # ====================================================
        # CREATE PRE-EVENT COMPOSITE
        # ====================================================

        pre_image = (

            pre_collection

            .median()

            .select(BANDS)

            .clip(aoi)

        )


        # ====================================================
        # CREATE POST-EVENT COMPOSITE
        # ====================================================

        post_image = (

            post_collection

            .median()

            .select(BANDS)

            .clip(aoi)

        )


        print("✓ PRE composite created")

        print("✓ POST composite created")


        # ====================================================
        # CREATE LANDSLIDE MASK
        # ====================================================

        mask_image = (

            ee.Image(0)

            .byte()

            .paint(

                featureCollection=
                ee.FeatureCollection([
                    ee.Feature(ee_geometry)
                ]),

                color=1

            )

            .rename("mask")

            .clip(aoi)

        )


        print("✓ Landslide mask created")


        # ====================================================
        # DOWNLOAD PRE IMAGE
        # ====================================================

        print("Downloading PRE image...")

        download_image(

            pre_image,

            aoi,

            sample_dir / "pre.tif"

        )

        print("✓ PRE saved")


        # ====================================================
        # DOWNLOAD POST IMAGE
        # ====================================================

        print("Downloading POST image...")

        download_image(

            post_image,

            aoi,

            sample_dir / "post.tif"

        )

        print("✓ POST saved")


        # ====================================================
        # DOWNLOAD MASK
        # ====================================================

        print("Downloading MASK...")

        download_image(

            mask_image,

            aoi,

            sample_dir / "mask.tif"

        )

        print("✓ MASK saved")


        # ====================================================
        # SAVE METADATA
        # ====================================================

        metadata = {

            "sample_id": sample_id,

            "landslide_name": str(
                row.get("Name", "Unknown")
            ),

            "area_m2": float(
                row["area_m2"]
            ),

            "estimated_pixels": float(
                row["estimated_pixels"]
            ),

            "pre_period":
                f"{PRE_START} to {PRE_END}",

            "post_period":
                f"{POST_START} to {POST_END}",

            "buffer_meters":
                BUFFER_METERS,

            "bands":
                BANDS

        }


        with open(

            sample_dir / "metadata.json",

            "w"

        ) as file:

            json.dump(

                metadata,

                file,

                indent=4

            )


        print(
            f"🎉 {sample_id} completed successfully!"
        )


        return {

            "sample_id": sample_id,

            "name": str(
                row.get("Name", "Unknown")
            ),

            "status": "SUCCESS",

            "pre_images": pre_count,

            "post_images": post_count,

            "estimated_pixels":
                row["estimated_pixels"],

            "error": ""

        }


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        print(f"❌ FAILED: {error}")

        traceback.print_exc()


        return {

            "sample_id": sample_id,

            "name": str(
                row.get("Name", "Unknown")
            ),

            "status": "FAILED",

            "pre_images": None,

            "post_images": None,

            "estimated_pixels":
                row["estimated_pixels"],

            "error": str(error)

        }


# ============================================================
# MAIN DATASET LOOP
# ============================================================

results = []

print(
    "\n========== STARTING DATASET GENERATION =========="
)


for i, (_, row) in enumerate(

    usable_gdf.iterrows(),

    start=1

):

    result = process_sample(i, row)

    results.append(result)


    # ========================================================
    # SAVE PROGRESS AFTER EVERY SAMPLE
    # ========================================================

    pd.DataFrame(results).to_csv(

        LOG_FILE,

        index=False

    )


    # Small delay between Earth Engine downloads
    time.sleep(2)


# ============================================================
# FINAL SUMMARY
# ============================================================

results_df = pd.DataFrame(results)


print(
    "\n========== DATASET GENERATION COMPLETE ==========\n"
)


print(
    results_df["status"].value_counts()
)


print("\nDataset location:")

print(OUTPUT_DIR)


print("\nLog file:")

print(LOG_FILE)


print(
    "\n🎉 LAND SENTRY KODAGU DATASET BUILD FINISHED!\n"
)