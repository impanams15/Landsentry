import ee
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# ============================================================
# LAND SENTRY: WAYANAD EVENT INSPECTION
# ============================================================

PROJECT_ID = "landsentry-507318"  # Keep your exact working project ID

ee.Initialize(project=PROJECT_ID)

print("\n========== LAND SENTRY: WAYANAD EVENT INSPECTION ==========\n")

# ------------------------------------------------------------
# EVENT LOCATION
# ------------------------------------------------------------

longitude = 76.13478743504008
latitude = 11.46457777777778

event_point = ee.Geometry.Point([longitude, latitude])

# Smaller AOI for detailed inspection
# 750 m around the known Mundakkai location
aoi = event_point.buffer(750).bounds()

print("✓ Earth Engine initialized")
print(f"✓ Event location: {latitude}, {longitude}")
print("✓ Detailed AOI created: 750 m buffer\n")


# ============================================================
# CLOUD MASK
# ============================================================

def mask_sentinel2(image):

    scl = image.select("SCL")

    mask = (
        scl.neq(3)   # cloud shadow
        .And(scl.neq(8))   # medium cloud
        .And(scl.neq(9))   # high cloud
        .And(scl.neq(10))  # cirrus
        .And(scl.neq(11))  # snow
    )

    return (
        image
        .updateMask(mask)
        .select(["B4", "B3", "B2", "B8"])
    )


# ============================================================
# CREATE COMPOSITES
# ============================================================

def create_composite(start_date, end_date):

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(mask_sentinel2)
    )

    count = collection.size().getInfo()

    print(
        f"{start_date} → {end_date}"
    )
    print(f"Images: {count}")

    return collection.median().clip(aoi)


print("Creating PRE-event composite...")

pre = create_composite(
    "2024-01-01",
    "2024-04-30"
)

print("\nCreating POST-event composite...")

post = create_composite(
    "2024-12-01",
    "2024-12-31"
)


# ============================================================
# DOWNLOAD THUMBNAIL
# ============================================================

def get_thumbnail(image):

    url = image.getThumbURL({
        "region": aoi,
        "dimensions": 1000,
        "bands": ["B4", "B3", "B2"],
        "min": 0,
        "max": 3000,
        "format": "png"
    })

    response = requests.get(url)
    response.raise_for_status()

    return plt.imread(BytesIO(response.content))


print("\nDownloading detailed images...")

pre_img = get_thumbnail(pre)
post_img = get_thumbnail(post)


# ============================================================
# VISUALIZE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 8)
)

axes[0].imshow(pre_img)
axes[0].set_title(
    "WAYANAD PRE-EVENT\nJan–Apr 2024",
    fontsize=16
)
axes[0].axis("off")


axes[1].imshow(post_img)
axes[1].set_title(
    "WAYANAD POST-EVENT\nDecember 2024",
    fontsize=16
)
axes[1].axis("off")


plt.tight_layout()

output_path = (
    "data/raw/wayanad/wayanad_event_inspection.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

print("\n✓ Inspection image saved:")
print(output_path)

plt.show()

print("\n========== INSPECTION COMPLETE ==========\n")