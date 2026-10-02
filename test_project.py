print("\n========== LAND SENTRY PROJECT TEST ==========\n")

# Configuration
import config
print("✓ config.py")

# Dataset
from data.dataset import NpyLandslideDataset
print("✓ data/dataset.py")

# Models
from models.siamese_cnn import SiameseChangeDetector
print("✓ models/siamese_cnn.py")

from models.transformer_cd import TransformerChangeDetector
print("✓ models/transformer_cd.py")

# Utilities
from utils.preprocessing import *
print("✓ utils/preprocessing.py")

from utils.metrics import *
print("✓ utils/metrics.py")

# Google Earth Engine utilities
from utils import gee_utils
print("✓ utils/gee_utils.py")

print("\n========== PROJECT IMPORT TEST SUCCESSFUL ==========")
print("🎉 LandSentry project structure is working!\n")