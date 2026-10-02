import ee

print("\n========== LAND SENTRY: WAYANAD SENTINEL-2 DATE CHECK ==========\n")

# Initialize Earth Engine
ee.Initialize(project="landsentry-507318")

# --------------------------------------------------
# WAYANAD LANDSLIDE LOCATION
# --------------------------------------------------

longitude = 76.13478743504008
latitude = 11.46457777777778

point = ee.Geometry.Point([longitude, latitude])

# Create an AOI around the event location
# 2 km buffer for initial satellite-image inspection
aoi = point.buffer(2000).bounds()

print("✓ Google Earth Engine initialized")
print(f"✓ Event location: {latitude}, {longitude}")
print("✓ AOI created: 2 km buffer around event location\n")


# --------------------------------------------------
# SENTINEL-2 COLLECTION FUNCTION
# --------------------------------------------------

def check_period(name, start_date, end_date):

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    count = collection.size().getInfo()

    print("=" * 60)
    print(name)
    print(f"Period: {start_date} → {end_date}")
    print(f"Images found: {count}")

    if count > 0:

        images = collection.limit(10).getInfo()["features"]

        print("\nBest available images:")

        for image in images:

            properties = image["properties"]

            date = properties.get(
                "DATE_ACQUIRED",
                properties.get("system:time_start")
            )

            timestamp = properties.get("system:time_start")

            cloud = properties.get(
                "CLOUDY_PIXEL_PERCENTAGE",
                "Unknown"
            )

            if timestamp:
                import datetime

                date = datetime.datetime.fromtimestamp(
                    timestamp / 1000
                ).strftime("%Y-%m-%d")

            print(f"  Date: {date} | Cloud: {cloud:.2f}%")

    else:
        print("⚠️ No images found for this period.")

    print()


# --------------------------------------------------
# CHECK DIFFERENT PRE / POST WINDOWS
# --------------------------------------------------

check_period(
    "EARLY 2024 PRE-EVENT",
    "2024-01-01",
    "2024-04-30"
)

check_period(
    "PRE-MONSOON 2024",
    "2024-05-01",
    "2024-07-29"
)

check_period(
    "IMMEDIATE POST-EVENT",
    "2024-07-30",
    "2024-08-31"
)

check_period(
    "LATER POST-EVENT",
    "2024-09-01",
    "2024-10-31"
)

check_period(
    "LATE 2024 POST-EVENT",
    "2024-11-01",
    "2024-12-31"
)

print("========== CHECK COMPLETE ==========\n")