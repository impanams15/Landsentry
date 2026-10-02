"""
LandSentry 2.0 - Quick Health Check (no GEE auth attempt)
"""
import requests, os, sys, subprocess, importlib

BACKEND = "http://localhost:8000"
FRONTEND = "http://localhost:8501"
PASS, FAIL, WARN = "[PASS]", "[FAIL]", "[WARN]"
issues = []

def check(label, fn):
    try:
        fn()
    except Exception as e:
        print(FAIL, label, "->", e)
        issues.append(label)

print("=" * 60)
print("  LandSentry 2.0  --  Pre-flight Health Check")
print("=" * 60)

# 1. Backend
print("\n--- 1. Backend API ---")
try:
    r = requests.get(BACKEND + "/", timeout=4)
    js = r.json()
    print(PASS if js.get("status") == "ok" else WARN, "Backend:", js.get("message"))
except Exception as e:
    print(FAIL, "Backend unreachable:", e)
    issues.append("Backend unreachable")

# 2. Database
print("\n--- 2. Database ---")
try:
    r = requests.get(BACKEND + "/api/incidents", timeout=4)
    js = r.json()
    print(PASS, f"DB OK  |  Reports: {len(js.get('reports',[]))}  |  Incidents: {len(js.get('incidents',[]))}")
except Exception as e:
    print(FAIL, "DB unreachable:", e)
    issues.append("DB unreachable")

# 3. Checkpoints
print("\n--- 3. Model Checkpoints ---")
ckpts = []
if os.path.isdir("checkpoints"):
    ckpts = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
if ckpts:
    for c in ckpts:
        print(PASS, c)
else:
    print(WARN, "No .pt checkpoints found. Train models first.")
    issues.append("No checkpoints")

# 4. Dependencies
print("\n--- 4. Python Packages ---")
for pkg in ["streamlit","torch","PIL","sqlalchemy","fastapi","folium","streamlit_folium","requests","numpy","matplotlib","ee"]:
    try:
        importlib.import_module(pkg)
        print(PASS, pkg)
    except:
        print(FAIL, pkg, "MISSING -- run: pip install", pkg)
        issues.append(f"{pkg} missing")

# 5. GEE (non-blocking check)
print("\n--- 5. Google Earth Engine ---")
try:
    import ee
    # Just check if credentials file exists, don't auth
    cred_paths = [
        os.path.expanduser("~/.config/earthengine/credentials"),
        os.path.expanduser("~\\.config\\earthengine\\credentials")
    ]
    found = any(os.path.exists(p) for p in cred_paths)
    if found:
        print(PASS, "GEE credentials file found (not initialized yet)")
    else:
        print(WARN, "GEE credentials missing. Run: earthengine authenticate")
        issues.append("GEE not authenticated")
except ImportError:
    print(FAIL, "earthengine-api not installed")
    issues.append("earthengine-api missing")

# 6. EXIF
print("\n--- 6. EXIF Utils ---")
try:
    from utils.exif_utils import extract_gps_from_image
    print(PASS, "EXIF GPS extractor ready")
except Exception as e:
    print(FAIL, e)
    issues.append("EXIF broken")

# 7. ML Models
print("\n--- 7. ML Models ---")
try:
    import config
    from models.siamese_cnn import SiameseChangeDetector
    from models.transformer_cd import TransformerChangeDetector
    s = SiameseChangeDetector(in_channels=config.IN_CHANNELS)
    t = TransformerChangeDetector(in_channels=config.IN_CHANNELS)
    print(PASS, f"SiameseChangeDetector  in_channels={config.IN_CHANNELS}")
    print(PASS, f"TransformerChangeDetector  in_channels={config.IN_CHANNELS}")
    print(PASS, f"Config: LOSS={config.LOSS_TYPE} | NDVI={config.USE_NDVI} | DEM={config.USE_DEM} | CORAL={config.USE_CORAL}")
except Exception as e:
    print(FAIL, "Model error:", e)
    issues.append("Model broken")

# 8. Loss Functions
print("\n--- 8. Loss Functions ---")
try:
    from models.losses import get_loss_function, coral_loss
    get_loss_function("bce_dice")
    get_loss_function("focal_tversky")
    get_loss_function("boundary_aware")
    print(PASS, "BCEDice / FocalTversky / BoundaryAware all instantiated")
except Exception as e:
    print(FAIL, "Loss error:", e)
    issues.append("Losses broken")

# 9. Frontend
print("\n--- 9. Streamlit Frontend ---")
try:
    r = requests.get(FRONTEND, timeout=4)
    print(PASS if r.status_code == 200 else WARN,
          f"Streamlit HTTP {r.status_code} at {FRONTEND}")
except Exception as e:
    print(WARN, f"Frontend not responding: {e}")
    issues.append("Streamlit not running")

# Summary
print()
print("=" * 60)
if not issues:
    print("  ALL CHECKS PASSED. LandSentry 2.0 is fully operational.")
    print(f"  Dashboard: {FRONTEND}")
    print(f"  API docs:  {BACKEND}/docs")
else:
    print(f"  {len(issues)} issue(s) detected:")
    for i, iss in enumerate(issues, 1):
        print(f"    {i}. {iss}")
print("=" * 60)
