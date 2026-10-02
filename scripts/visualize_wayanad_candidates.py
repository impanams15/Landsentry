import ee
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

ee.Initialize(project="landsentry-507318")  # Keep your exact project ID here

print("\n========== WAYANAD IMAGE CANDIDATE VISUALIZATION ==========\n")

# ============================================================
# WAYANAD EVENT LOCATION
# ============================================================

longitude = 76.13478743504008
latitude = 11.46457777777778

point = ee.Geometry.Point([longitude, latitude])

# 2 km inspection area
aoi = point.buffer(2000).bounds()

print("✓ Earth Engine initialized")
print("✓ Wayanad AOI created\n")


# ============================================================
# CLOUD MASKING FUNCTION
# ============================================================

def mask_sentinel2(image):

    scl = image.select("SCL")

    # Remove:
    # 3 = cloud shadow
    # 8 = cloud medium probability
    # 9 = cloud high probability
    # 10 = cirrus
    # 11 = snow/ice

    mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    return (
        image
        .updateMask(mask)
        .select(["B4", "B3", "B2"])
    )


# ============================================================
# CREATE COMPOSITE
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
        f"Period {start_date} → {end_date}"
    )
    print(f"Images available: {count}")

    composite = collection.median().clip(aoi)

    return composite


# ============================================================
# PRE EVENT
# ============================================================

print("\nCreating PRE-event composite...")

pre = create_composite(
    "2024-01-01",
    "2024-04-30"
)


# ============================================================
# IMMEDIATE POST EVENT
# ============================================================

print("\nCreating IMMEDIATE POST-event composite...")

post_august = create_composite(
    "2024-07-30",
    "2024-08-31"
)


# ============================================================
# LATE POST EVENT
# ============================================================

print("\nCreating CLEAN LATE POST-event composite...")

post_december = create_composite(
    "2024-12-01",
    "2024-12-31"
)


# ============================================================
# GET IMAGE THUMBNAIL
# ============================================================

def get_thumbnail(image):

    url = image.getThumbURL({
        "region": aoi,
        "dimensions": 700,
        "bands": ["B4", "B3", "B2"],
        "min": 0,
        "max": 3000,
        "format": "png"
    })

    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download image: {response.status_code}"
        )

    return plt.imread(
        BytesIO(response.content),
        format="png"
    )


print("\nDownloading visualization images...")

pre_img = get_thumbnail(pre)
august_img = get_thumbnail(post_august)
december_img = get_thumbnail(post_december)


# ============================================================
# DISPLAY RESULTS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(pre_img)
axes[0].set_title(
    "PRE-EVENT\nJan–Apr 2024",
    fontsize=14
)
axes[0].axis("off")


axes[1].imshow(august_img)
axes[1].set_title(
    "IMMEDIATE POST-EVENT\nAug 2024",
    fontsize=14
)
axes[1].axis("off")


axes[2].imshow(december_img)
axes[2].set_title(
    "LATE POST-EVENT\nDec 2024",
    fontsize=14
)
axes[2].axis("off")


plt.tight_layout()

output_path = (
    "data/raw/wayanad/wayanad_date_comparison.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

print(f"\n✓ Visualization saved to:")
print(output_path)

print("\n🎉 WAYANAD DATE COMPARISON COMPLETE!\n")

plt.show()