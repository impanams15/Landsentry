import ee
import geopandas as gpd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "landsentry-507318"

SHAPEFILE_PATH = Path(
    "data/clean/kodagu/kodagu_landslides_clean.geojson"
)


# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

ee.Initialize(project=PROJECT_ID)

print("\n========== CREATE FIRST KODAGU SAMPLE ==========\n")


# ============================================================
# LOAD LANDSLIDE
# ============================================================

gdf = gpd.read_file(SHAPEFILE_PATH)

landslide = gdf.iloc[0]

print("Selected landslide:")
print("Name:", landslide["Name"])


# ============================================================
# CONVERT TO EARTH ENGINE GEOMETRY
# ============================================================

geometry = landslide.geometry

ee_geometry = ee.Geometry(
    geometry.__geo_interface__
)


# ============================================================
# CREATE AOI
# ============================================================

# 500 metre context around the landslide
aoi = ee_geometry.buffer(500)

print("✓ AOI created")


# ============================================================
# CLOUD MASK
# ============================================================

def mask_s2_clouds(image):

    scl = image.select("SCL")

    mask = (
        scl.neq(3)      # Cloud shadow
        .And(scl.neq(8))   # Medium probability cloud
        .And(scl.neq(9))   # High probability cloud
        .And(scl.neq(10))  # Cirrus
    )

    return image.updateMask(mask)


# ============================================================
# PRE-EVENT COLLECTION
# ============================================================

pre_collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2018-01-01", "2018-04-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .map(mask_s2_clouds)
)

pre_count = pre_collection.size().getInfo()

print(f"✓ PRE images available: {pre_count}")


# ============================================================
# POST-EVENT COLLECTION
# ============================================================

post_collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2018-10-01", "2019-01-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .map(mask_s2_clouds)
)

post_count = post_collection.size().getInfo()

print(f"✓ POST images available: {post_count}")


# ============================================================
# CREATE RGB + NIR COMPOSITES
# ============================================================

bands = ["B4", "B3", "B2", "B8"]

pre_image = (
    pre_collection
    .median()
    .select(bands)
    .clip(aoi)
)

post_image = (
    post_collection
    .median()
    .select(bands)
    .clip(aoi)
)

print("✓ PRE composite created")
print("✓ POST composite created")


# ============================================================
# CREATE LANDSLIDE MASK
# ============================================================

mask = (
    ee.Image()
    .byte()
    .paint(
        featureCollection=ee.FeatureCollection([
            ee.Feature(ee_geometry)
        ]),
        color=1
    )
    .clip(aoi)
)

print("✓ Landslide mask created")


# ============================================================
# CHECK IMAGE INFORMATION
# ============================================================

print("\nChecking bands...")

print("PRE bands:", pre_image.bandNames().getInfo())
print("POST bands:", post_image.bandNames().getInfo())

print("\nChecking mask bands:")
print(mask.bandNames().getInfo())


# ============================================================
# FINAL STATUS
# ============================================================

print("\n========== SUCCESS ==========")

print("🎉 First Kodagu bitemporal sample exists in Earth Engine!")
print("\nNext step: Export PRE + POST + MASK as GeoTIFF files.\n")