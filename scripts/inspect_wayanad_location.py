import zipfile
from pathlib import Path
import re

print("\n========== LAND SENTRY: WAYANAD LOCATION INSPECTION ==========\n")

kmz_path = Path(
    "data/raw/wayanad/Mundakkai Landslide 2024 (1).kmz"
)

if not kmz_path.exists():
    print(f"✗ KMZ file not found: {kmz_path}")
    raise SystemExit

print(f"✓ KMZ found: {kmz_path.name}\n")

with zipfile.ZipFile(kmz_path, "r") as kmz:
    files = kmz.namelist()

    print("Files inside KMZ:")
    for file in files:
        print(f"  → {file}")

    kml_file = next(
        (f for f in files if f.lower().endswith(".kml")),
        None
    )

    if not kml_file:
        print("\n✗ No KML file found!")
        raise SystemExit

    kml_text = kmz.read(kml_file).decode("utf-8")

print("\nSearching for coordinates...\n")

match = re.search(
    r"<coordinates>\s*([^<]+)\s*</coordinates>",
    kml_text
)

if not match:
    print("✗ No coordinates found!")
    raise SystemExit

coordinates = match.group(1).strip()

print(f"Raw coordinates: {coordinates}")

lon, lat, *_ = coordinates.split(",")

lon = float(lon)
lat = float(lat)

print("\n========== WAYANAD EVENT LOCATION ==========")
print(f"Latitude : {lat}")
print(f"Longitude: {lon}")

print("\n✓ Location extracted successfully!")

print("\n================================================\n")