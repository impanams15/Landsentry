import geopandas as gpd
from pathlib import Path

INPUT_PATH = Path("data/raw/kodagu/Landslide_polygons.shp")
OUTPUT_DIR = Path("data/clean/kodagu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "kodagu_landslides_clean.geojson"


print("Loading Kodagu landslide inventory...")

gdf = gpd.read_file(INPUT_PATH)

print(f"Total features: {len(gdf)}")
print(f"Invalid geometries before cleaning: {(~gdf.geometry.is_valid).sum()}")


# --------------------------------------------------
# STEP 1: Remove Z dimension
# --------------------------------------------------

# The shapefile contains POLYGON Z geometries.
# We only need 2D longitude/latitude coordinates.

gdf["geometry"] = gdf.geometry.force_2d()


# --------------------------------------------------
# STEP 2: Fix invalid geometries
# --------------------------------------------------

gdf["geometry"] = gdf.geometry.make_valid()


# --------------------------------------------------
# STEP 3: Remove empty geometries if any
# --------------------------------------------------

gdf = gdf[
    gdf.geometry.notnull() &
    ~gdf.geometry.is_empty
].copy()


print(f"Invalid geometries after cleaning: {(~gdf.geometry.is_valid).sum()}")
print(f"Remaining features: {len(gdf)}")


# --------------------------------------------------
# STEP 4: Save clean dataset
# --------------------------------------------------

gdf.to_file(
    OUTPUT_PATH,
    driver="GeoJSON"
)

print("\nClean inventory saved successfully:")
print(OUTPUT_PATH)

print("\nCRS:", gdf.crs)
print("\nGeometry types:")
print(gdf.geom_type.value_counts())