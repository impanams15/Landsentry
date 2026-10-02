print("\n========== LAND SENTRY ENVIRONMENT TEST ==========\n")

# Core packages
import numpy as np
print("✓ NumPy:", np.__version__)

import pandas as pd
print("✓ Pandas:", pd.__version__)

import matplotlib
print("✓ Matplotlib:", matplotlib.__version__)

import sklearn
print("✓ Scikit-learn:", sklearn.__version__)

from tqdm import tqdm
print("✓ tqdm")

# Geospatial packages
import geopandas as gpd
print("✓ GeoPandas:", gpd.__version__)

import rasterio
print("✓ Rasterio:", rasterio.__version__)

import shapely
print("✓ Shapely:", shapely.__version__)

import pyproj
print("✓ PyProj:", pyproj.__version__)

# Image processing
import cv2
print("✓ OpenCV:", cv2.__version__)

from PIL import Image
print("✓ Pillow:", Image.__version__)

# Google Earth Engine
import ee
print("✓ Earth Engine API")

import geemap
print("✓ Geemap:", geemap.__version__)

# Dashboard
import streamlit
print("✓ Streamlit:", streamlit.__version__)

import folium
print("✓ Folium:", folium.__version__)

import streamlit_folium
print("✓ streamlit-folium")

# Deep Learning
import torch
print("✓ PyTorch:", torch.__version__)

import torchvision
print("✓ Torchvision:", torchvision.__version__)

print("\nChecking PyTorch hardware...")

if torch.cuda.is_available():
    print("✓ CUDA GPU available!")
    print("  GPU:", torch.cuda.get_device_name(0))
else:
    print("ℹ CUDA GPU not available — CPU mode is currently active.")
    print("  This is completely fine for development.")

print("\n========== ALL ENVIRONMENT IMPORTS SUCCESSFUL ==========")
print("🎉 LandSentry environment is ready!\n")