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
# INITIALIZE GOOGLE EARTH ENGINE
# ============================================================

ee.Initialize(project=PROJECT_ID)

print("\n========== LAND SENTRY: KODAGU GEE TEST ==========\n")


# ============================================================
# LOAD KODAGU INVENTORY
# ============================================================

gdf = gpd.read_file(SHAPEFILE_PATH)

print(f"Total Kodagu landslide features: {len(gdf)}")


# ============================================================
# SELECT FIRST LANDSLIDE
# ============================================================

landslide = gdf.iloc[0]

print("\nSelected landslide:")
print("Name:", landslide["Name"])
print("Area attribute:", landslide["area"])


# ============================================================
# CONVERT POLYGON TO EARTH ENGINE GEOMETRY
# ============================================================

geometry = landslide.geometry

# GeoJSON geometry dictionary
geojson_geometry = geometry.__geo_interface__

# Convert to Earth Engine geometry
ee_geometry = ee.Geometry(geojson_geometry)


# ============================================================
# CREATE AOI
# ============================================================

# Buffer around landslide polygon (500 metres)
aoi = ee_geometry.buffer(500)

print("\nAOI created successfully.")


# ============================================================
# SENTINEL-2 CLOUD MASK FUNCTION
# ============================================================

def mask_s2_clouds(image):

    scl = image.select("SCL")

    # Keep useful surface classes
    mask = (
        scl.neq(3)   # cloud shadow
        .And(scl.neq(8))   # cloud medium probability
        .And(scl.neq(9))   # cloud high probability
        .And(scl.neq(10))  # cirrus
    )

    return image.updateMask(mask)


# ============================================================
# PRE-EVENT SENTINEL-2
# ============================================================

print("\nSearching PRE-event Sentinel-2 images...")

pre_collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2018-05-01", "2018-07-15")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    .map(mask_s2_clouds)
)

pre_count = pre_collection.size().getInfo()

print(f"PRE-event images found: {pre_count}")


# ============================================================
# POST-EVENT SENTINEL-2
# ============================================================

print("\nSearching POST-event Sentinel-2 images...")

post_collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2018-09-01", "2018-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
    .map(mask_s2_clouds)
)

post_count = post_collection.size().getInfo()

print(f"POST-event images found: {post_count}")


# ============================================================
# CREATE COMPOSITES
# ============================================================

if pre_count > 0:

    pre_image = pre_collection.median()

    print("\n✓ PRE-event composite created.")

else:

    print("\n✗ No suitable PRE-event images found.")


if post_count > 0:

    post_image = post_collection.median()

    print("✓ POST-event composite created.")

else:

    print("✗ No suitable POST-event images found.")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n========== RESULTS ==========")

if pre_count > 0 and post_count > 0:

    print("🎉 SUCCESS!")
    print("We can create a genuine Kodagu bitemporal sample.")

else:

    print("⚠️ We need to adjust the date windows/cloud filtering.")


print("\n===============================================\n")