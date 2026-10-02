import geopandas as gpd

# Path to the Kodagu landslide polygon shapefile
SHAPEFILE_PATH = "data/raw/kodagu/Landslide_polygons.shp"

# Load the shapefile
gdf = gpd.read_file(SHAPEFILE_PATH)

print("\n========== KODAGU DATASET INSPECTION ==========\n")

# Number of polygons
print("Number of landslide polygons:", len(gdf))

# Coordinate Reference System
print("\nCRS:")
print(gdf.crs)

# Attribute columns
print("\nColumns:")
print(gdf.columns.tolist())

# First few records
print("\nFirst 5 records:")
print(gdf.head())

# Geometry types
print("\nGeometry types:")
print(gdf.geom_type.value_counts())

# Bounding box
print("\nBounding Box:")
print(gdf.total_bounds)

# Check for missing geometries
print("\nMissing geometries:", gdf.geometry.isnull().sum())

# Check geometry validity
print("Invalid geometries:", (~gdf.geometry.is_valid).sum())

# Calculate polygon areas
print("\nArea information:")
print(gdf.geometry.area.describe())

print("\n========== INSPECTION COMPLETE ==========\n")