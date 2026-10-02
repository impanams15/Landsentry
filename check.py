import ee
import sys

print("Checking Earth Engine Credentials...")
try:
    ee.Initialize(project="landsentry-508714")
    print("SUCCESS: You are authenticated and initialized securely!")
except Exception as e:
    print("FAILED:", str(e))
    sys.exit(1)
