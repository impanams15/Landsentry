import ee
import geopandas as gpd
from pathlib import Path


PROJECT_ID = "landsentry-507318"

SHAPEFILE_PATH = Path(
    "data/clean/kodagu/kodagu_landslides_clean.geojson"
)


# ------------------------------------------------------------
# INITIALIZE EARTH ENGINE
# ------------------------------------------------------------

ee.Initialize(project=PROJECT_ID)

print("\n========== KODAGU SENTINEL-2 DATE CHECK ==========\n")


# ------------------------------------------------------------
# LOAD FIRST LANDSLIDE
# ------------------------------------------------------------

gdf = gpd.read_file(SHAPEFILE_PATH)

landslide = gdf.iloc[0]

print("Landslide:", landslide["Name"])

geometry = landslide.geometry
ee_geometry = ee.Geometry(geometry.__geo_interface__)

# Smaller/larger AOI doesn't matter much for this availability check
aoi = ee_geometry.buffer(500)


# ------------------------------------------------------------
# SENTINEL-2 COLLECTION
# ------------------------------------------------------------

collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
)


# ------------------------------------------------------------
# FUNCTION TO CHECK A PERIOD
# ------------------------------------------------------------

def check_period(name, start_date, end_date):

    images = (
        collection
        .filterDate(start_date, end_date)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    count = images.size().getInfo()

    print(f"\n{name}")
    print(f"Period: {start_date} → {end_date}")
    print(f"Images found: {count}")

    if count > 0:

        info = images.limit(5).getInfo()

        print("Best available images:")

        for feature in info["features"]:

            props = feature["properties"]

            date = props.get("system:time_start")
            cloud = props.get("CLOUDY_PIXEL_PERCENTAGE")

            import datetime

            readable_date = datetime.datetime.fromtimestamp(
                date / 1000
            ).strftime("%Y-%m-%d")

            print(
                f"  Date: {readable_date} | "
                f"Cloud: {cloud:.2f}%"
            )


# ------------------------------------------------------------
# CHECK MULTIPLE TIME PERIODS
# ------------------------------------------------------------

check_period(
    "EARLY 2018 PRE-EVENT",
    "2018-01-01",
    "2018-04-30"
)

check_period(
    "PRE-MONSOON 2018",
    "2018-05-01",
    "2018-06-30"
)

check_period(
    "MONSOON PERIOD",
    "2018-07-01",
    "2018-08-31"
)

check_period(
    "POST-EVENT 2018",
    "2018-09-01",
    "2018-12-31"
)


print("\n========== CHECK COMPLETE ==========\n")