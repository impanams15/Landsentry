from pathlib import Path
import zipfile

print("\n========== LAND SENTRY: WAYANAD KMZ INSPECTION ==========\n")

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Wayanad raw data folder
WAYANAD_DIR = PROJECT_ROOT / "data" / "raw" / "wayanad"

print(f"Looking inside:\n{WAYANAD_DIR}\n")

# Find KMZ files automatically
kmz_files = list(WAYANAD_DIR.glob("*.kmz")) + list(WAYANAD_DIR.glob("*.KMZ"))

if not kmz_files:
    print("✗ No KMZ file found!")
    print("Make sure your Wayanad KMZ is inside:")
    print(WAYANAD_DIR)
    raise SystemExit

print(f"✓ Found {len(kmz_files)} KMZ file(s):\n")

for file in kmz_files:
    print(f"  → {file.name}")
    print(f"    Size: {file.stat().st_size / 1024:.2f} KB")

kmz_path = kmz_files[0]

print("\n========== INSPECTING KMZ CONTENTS ==========\n")

try:
    with zipfile.ZipFile(kmz_path, "r") as kmz:
        files = kmz.namelist()

        print("Files inside KMZ:\n")

        for file in files:
            print(f"  ✓ {file}")

        # Find KML file
        kml_files = [
            f for f in files
            if f.lower().endswith(".kml")
        ]

        if not kml_files:
            print("\n✗ No KML file found inside the KMZ!")
            raise SystemExit

        kml_file = kml_files[0]

        print(f"\n✓ Main KML file found: {kml_file}")

        # Read a preview
        kml_content = kmz.read(kml_file).decode(
            "utf-8",
            errors="ignore"
        )

        print("\n========== KML PREVIEW ==========\n")

        print(kml_content[:3000])

        print("\n========== INSPECTION COMPLETE ==========")

except zipfile.BadZipFile:
    print("✗ This file is not a valid KMZ/ZIP file.")