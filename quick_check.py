import requests, os, sys

results = []

# 1. Backend
try:
    r = requests.get("http://localhost:8000/", timeout=4)
    results.append(("PASS", "Backend: " + r.json().get("message","")))
except Exception as e:
    results.append(("FAIL", "Backend: " + str(e)))

# 2. DB
try:
    r = requests.get("http://localhost:8000/api/incidents", timeout=4)
    js = r.json()
    results.append(("PASS", "DB: Reports={}, Incidents={}".format(len(js["reports"]), len(js["incidents"]))))
except Exception as e:
    results.append(("FAIL", "DB: " + str(e)))

# 3. Checkpoints
pts = [f for f in os.listdir("checkpoints") if f.endswith(".pt")] if os.path.isdir("checkpoints") else []
for p in pts:
    results.append(("PASS", "Checkpoint: " + p))
if not pts:
    results.append(("WARN", "No .pt checkpoints found"))

# 4. EXIF
try:
    from utils.exif_utils import extract_gps_from_image
    results.append(("PASS", "EXIF GPS extractor OK"))
except Exception as e:
    results.append(("FAIL", "EXIF: " + str(e)))

# 5. Models
try:
    import config
    from models.siamese_cnn import SiameseChangeDetector
    from models.transformer_cd import TransformerChangeDetector
    SiameseChangeDetector(in_channels=config.IN_CHANNELS)
    TransformerChangeDetector(in_channels=config.IN_CHANNELS)
    results.append(("PASS", "Models OK  in_channels={}  loss={}  ndvi={}  coral={}".format(
        config.IN_CHANNELS, config.LOSS_TYPE, config.USE_NDVI, config.USE_CORAL)))
except Exception as e:
    results.append(("FAIL", "Models: " + str(e)))

# 6. Loss functions
try:
    from models.losses import get_loss_function, coral_loss
    get_loss_function("bce_dice")
    get_loss_function("focal_tversky")
    get_loss_function("boundary_aware")
    results.append(("PASS", "All 3 loss functions OK (BCEDice, FocalTversky, BoundaryAware)"))
except Exception as e:
    results.append(("FAIL", "Losses: " + str(e)))

# 7. Frontend
try:
    r = requests.get("http://localhost:8501", timeout=4)
    results.append(("PASS", "Streamlit frontend live at http://localhost:8501"))
except Exception as e:
    results.append(("WARN", "Frontend unreachable: " + str(e)))

# 8. GEE credentials
cred = os.path.expanduser("~/.config/earthengine/credentials")
if os.path.exists(cred):
    results.append(("PASS", "GEE credentials file present"))
else:
    results.append(("WARN", "GEE: No credentials -> run: earthengine authenticate"))

# Print
print("")
print("=" * 58)
print("  LandSentry 2.0  --  Stack Health Report")
print("=" * 58)
for tag, msg in results:
    print("[{}]  {}".format(tag, msg))

failures = [m for t,m in results if t == "FAIL"]
warns = [m for t,m in results if t == "WARN"]
print("")
print("=" * 58)
if not failures:
    print("  PLATFORM READY  |  {} warning(s)".format(len(warns)))
    print("  Dashboard  -->  http://localhost:8501")
    print("  API docs   -->  http://localhost:8000/docs")
else:
    print("  {} FAILURE(S)".format(len(failures)))
    for f in failures:
        print("    -", f)
print("=" * 58)
