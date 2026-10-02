import os
import ee
import geemap

# ============================================================
# LAND SENTRY — EXPORT WAYANAD EVENT DATA
# ============================================================

PROJECT_ID = "landsentry"   # Change only if your GCP project ID is different

OUTPUT_DIR = "data/processed/wayanad_event"

LATITUDE = 11.46457777777778
LONGITUDE = 76.13478743504008

# AOI size around the event
BUFFER_METERS = 1500

# Sentinel-2 bands
BANDS = ["B2", "B3", "B4", "B8"]


def initialize_earth_engine():

    try:
        ee.Initialize(project="landsentry-507318")
        print("✓ Google Earth Engine initialized")

    except Exception as e:
        print("✗ Earth Engine initialization failed")
        print(e)
        raise


def mask_s2_clouds(image):
    """
    Basic Sentinel-2 cloud filtering using QA60.
    """

    qa = image.select("QA60")

    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(
            qa.bitwiseAnd(cirrus_bit_mask).eq(0)
        )
    )

    return (
        image
        .updateMask(mask)
        .select(BANDS)
        .copyProperties(image, ["system:time_start"])
    )


def create_composite(start_date, end_date, aoi, label):

    print()
    print(f"Creating {label} composite...")
    print(f"Period: {start_date} → {end_date}")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                40
            )
        )
    )

    count = collection.size().getInfo()

    print(f"Images found: {count}")

    if count == 0:
        raise RuntimeError(
            f"No Sentinel-2 images found for {label}"
        )

    clean_collection = collection.map(mask_s2_clouds)

    composite = clean_collection.median()

    return composite


def main():

    print()
    print("=" * 60)
    print("LAND SENTRY — WAYANAD EVENT EXPORT")
    print("=" * 60)

    # --------------------------------------------------------
    # Initialize GEE
    # --------------------------------------------------------

    initialize_earth_engine()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print(f"✓ Output directory ready: {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Create AOI
    # --------------------------------------------------------

    event_point = ee.Geometry.Point(
        [LONGITUDE, LATITUDE]
    )

    aoi = event_point.buffer(
        BUFFER_METERS
    ).bounds()

    print()
    print("✓ Wayanad event location loaded")

    print(f"Latitude : {LATITUDE}")
    print(f"Longitude: {LONGITUDE}")

    print(
        f"✓ AOI created: {BUFFER_METERS} m buffer"
    )

    # --------------------------------------------------------
    # PRE-EVENT
    # --------------------------------------------------------

    pre_image = create_composite(
        "2024-01-01",
        "2024-04-30",
        aoi,
        "PRE-EVENT"
    )

    # --------------------------------------------------------
    # POST-EVENT
    # --------------------------------------------------------

    post_image = create_composite(
        "2024-12-01",
        "2024-12-31",
        aoi,
        "POST-EVENT"
    )

    # --------------------------------------------------------
    # Export images
    # --------------------------------------------------------

    pre_path = os.path.join(
        OUTPUT_DIR,
        "pre.tif"
    )

    post_path = os.path.join(
        OUTPUT_DIR,
        "post.tif"
    )

    print()
    print("Downloading PRE-event image...")

    geemap.ee_export_image(
        pre_image,
        filename=pre_path,
        region=aoi,
        scale=10,
        crs="EPSG:4326"
    )

    print(f"✓ Saved: {pre_path}")

    print()
    print("Downloading POST-event image...")

    geemap.ee_export_image(
        post_image,
        filename=post_path,
        region=aoi,
        scale=10,
        crs="EPSG:4326"
    )

    print(f"✓ Saved: {post_path}")

    print()
    print("=" * 60)
    print("WAYANAD EVENT EXPORT COMPLETE")
    print("=" * 60)

    print()
    print("Files created:")

    print(f"✓ {pre_path}")
    print(f"✓ {post_path}")

    print()
    print("🎉 WAYANAD PRE + POST DATA READY!")


if __name__ == "__main__":
    main()