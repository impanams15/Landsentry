import ee
import geopandas as gpd
from pathlib import Path
import requests


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "landsentry-507318"

SHAPEFILE_PATH = Path(
    "data/clean/kodagu/kodagu_landslides_clean.geojson"
)

OUTPUT_DIR = Path(
    "data/processed/kodagu/sample_001"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

ee.Initialize(project=PROJECT_ID)

print("\n========== EXPORT FIRST KODAGU SAMPLE ==========\n")


# ============================================================
# LOAD LANDSLIDE
# ============================================================

gdf = gpd.read_file(SHAPEFILE_PATH)

landslide = gdf.iloc[0]

print("Selected landslide:")
print(landslide["Name"])


# ============================================================
# GEOMETRY + AOI
# ============================================================

geometry = landslide.geometry

ee_geometry = ee.Geometry(
    geometry.__geo_interface__
)

# 500 metres surrounding context
aoi = ee_geometry.buffer(500)


# ============================================================
# CLOUD MASK
# ============================================================

def mask_s2_clouds(image):

    scl = image.select("SCL")

    mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
    )

    return image.updateMask(mask)


# ============================================================
# PRE-EVENT IMAGE
# ============================================================

bands = ["B4", "B3", "B2", "B8"]

pre_collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2018-01-01", "2018-04-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .map(mask_s2_clouds)
)

pre_image = (
    pre_collection
    .median()
    .select(bands)
    .clip(aoi)
)


# ============================================================
# POST-EVENT IMAGE
# ============================================================

post_collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2018-10-01", "2019-01-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .map(mask_s2_clouds)
)

post_image = (
    post_collection
    .median()
    .select(bands)
    .clip(aoi)
)


# ============================================================
# LANDSLIDE MASK
# ============================================================

mask_image = (
    ee.Image()
    .byte()
    .paint(
        featureCollection=ee.FeatureCollection([
            ee.Feature(ee_geometry)
        ]),
        color=1
    )
    .rename("mask")
    .clip(aoi)
)


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_image(image, filename):

    print(f"\nDownloading {filename}...")

    url = image.getDownloadURL({
        "region": aoi,
        "scale": 10,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF"
    })

    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(
            f"Download failed: {response.status_code}\n"
            f"{response.text}"
        )

    output_path = OUTPUT_DIR / filename

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"✓ Saved: {output_path}")


# ============================================================
# DOWNLOAD ALL THREE
# ============================================================

download_image(pre_image, "pre.tif")

download_image(post_image, "post.tif")

download_image(mask_image, "mask.tif")


print("\n========== EXPORT COMPLETE ==========")

print("\nFiles created:")

for file in OUTPUT_DIR.iterdir():
    print("✓", file.name)

print("\n🎉 FIRST REAL KODAGU SAMPLE DOWNLOADED SUCCESSFULLY!\n")