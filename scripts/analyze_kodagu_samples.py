import geopandas as gpd
import pandas as pd
from pathlib import Path


print("\n========== LAND SENTRY: KODAGU SAMPLE ANALYSIS ==========\n")


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/clean/kodagu/kodagu_landslides_clean.geojson"
)

OUTPUT_FILE = Path(
    "data/clean/kodagu/kodagu_sample_analysis.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Kodagu landslide inventory...")

gdf = gpd.read_file(INPUT_FILE)

print(f"✓ Total polygons loaded: {len(gdf)}")


# ============================================================
# PROJECT TO UTM FOR CORRECT AREA CALCULATION
# ============================================================

print("\nCalculating polygon areas in metres...")

# Kodagu lies approximately in UTM Zone 43N
gdf_projected = gdf.to_crs("EPSG:32643")

# Calculate area in square metres
gdf["area_m2"] = gdf_projected.geometry.area


# ============================================================
# ESTIMATE SENTINEL-2 PIXEL COVERAGE
# ============================================================

# Sentinel-2 RGB/NIR spatial resolution = 10 m
# One pixel covers approximately 10m × 10m = 100 m²

gdf["estimated_pixels"] = gdf["area_m2"] / 100


# ============================================================
# SORT BY SIZE
# ============================================================

analysis = gdf[
    ["Name", "area_m2", "estimated_pixels"]
].copy()

analysis = analysis.sort_values(
    by="estimated_pixels",
    ascending=False
)


# ============================================================
# SAVE RESULTS
# ============================================================

analysis.to_csv(OUTPUT_FILE, index=False)

print(f"\n✓ Analysis saved to:")
print(OUTPUT_FILE)


# ============================================================
# PRINT STATISTICS
# ============================================================

print("\n========== PIXEL COVERAGE STATISTICS ==========\n")

print(analysis["estimated_pixels"].describe())


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = [10, 25, 50, 100, 250, 500]

print("\n========== USABLE SAMPLE COUNTS ==========\n")

for threshold in thresholds:

    count = (
        analysis["estimated_pixels"] >= threshold
    ).sum()

    print(
        f"Polygons with ≥ {threshold} estimated pixels: "
        f"{count}"
    )


# ============================================================
# TOP 10 LARGEST LANDSLIDES
# ============================================================

print("\n========== TOP 10 LARGEST LANDSLIDES ==========\n")

print(
    analysis.head(10).to_string(index=False)
)


print("\n========== ANALYSIS COMPLETE ==========\n")