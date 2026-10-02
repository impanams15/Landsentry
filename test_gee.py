import ee

# Your Google Cloud Project ID
PROJECT_ID = "landsentry-507318"

# Initialize Google Earth Engine
ee.Initialize(project=PROJECT_ID)

print("✓ Google Earth Engine connected successfully!")

# Test Sentinel-2 access
collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

print("✓ Sentinel-2 collection accessed successfully!")

# Simple test: count images
count = collection.limit(10).size().getInfo()

print(f"✓ Earth Engine query successful!")
print(f"  Test collection size returned: {count}")

print("\n🎉 GOOGLE EARTH ENGINE IS READY FOR LANDSENTRY!")