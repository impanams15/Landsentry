import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.engine import Base, engine
import os
import shutil
from PIL import Image

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_exif_fallback_manual():
    # If no EXIF and no manual, it should fail
    # We will upload a dummy text file masking as image
    os.makedirs("uploads", exist_ok=True)
    with open("dummy.jpg", "w") as f:
        f.write("dummy")
        
    with open("dummy.jpg", "rb") as f:
        response = client.post("/api/reports", files={"image": ("dummy.jpg", f, "image/jpeg")})
        
    assert response.status_code == 200
    js = response.json()
    assert js["status"] == "error"
    assert js["needs_manual"] == True
    
    # Clean up
    os.remove("dummy.jpg")

def test_exif_manual_success():
    with open("dummy.jpg", "w") as f:
        f.write("dummy")
        
    with open("dummy.jpg", "rb") as f:
        response = client.post(
            "/api/reports", 
            files={"image": ("dummy.jpg", f, "image/jpeg")},
            data={"manual_lat": 12.0, "manual_lon": 75.0, "description": "Test"}
        )
        
    assert response.status_code == 200
    js = response.json()
    assert js["status"] == "success"
    assert js["source"] == "USER_SELECTED"
    
    os.remove("dummy.jpg")
