"""Central configuration constants for LandSentry."""

# Sentinel-2 bands used throughout the project: Red, Green, Blue, Near-Infrared.
# NIR is included because vegetation/soil contrast on the NIR band is a strong
# landslide-scar cue (bare, saturated soil vs. healthy vegetation).
BANDS = ("B4", "B3", "B2", "B8")
USE_NDVI = True
USE_DEM = False

def get_in_channels():
    c = len(BANDS)
    if USE_NDVI:
        c += 1
    if USE_DEM:
        c += 3  # elevation, slope, aspect
    return c

IN_CHANNELS = get_in_channels()

# LOSS CONFIGURATION
# Options: "bce_dice", "focal_tversky", "boundary_aware"
LOSS_TYPE = "bce_dice"
LAMBDA_BOUNDARY = 0.5
USE_CORAL = False
LAMBDA_CORAL = 0.1

# Training patch size. Must be divisible by 4 so the transformer model's
# stride-2/stride-2 encoder and its matching decoder reconstruct the exact
# input resolution.
PATCH_SIZE = 64

# Sentinel-2 ground sample distance in metres, used to convert a pixel count
# into an estimated affected area in square kilometres.
PIXEL_SIZE_M = 10

DEFAULT_CHECKPOINT_DIR = "checkpoints"
