from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from backend.database.engine import engine, Base
import backend.database.models as models

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="LandSentry 2.0 API", description="Bitemporal Landslide Change Detection Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Form, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
import shutil
import uuid
import datetime

from backend.database.engine import get_db
from utils.exif_utils import extract_gps_from_image

@app.get("/")
def read_root():
    return {"status": "ok", "message": "LandSentry 2.0 API is running"}


@app.post("/api/reports")
async def create_report(
    image: UploadFile = File(...),
    manual_lat: Optional[float] = Form(None),
    manual_lon: Optional[float] = Form(None),
    description: Optional[str] = Form(""),
    db: Session = Depends(get_db)
):
    """
    Ingest a citizen report, enforce EXIF GPS checking, and persist to DB.
    """
    os.makedirs("uploads", exist_ok=True)
    temp_path = f"uploads/{uuid.uuid4()}_{image.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    lat, lon = extract_gps_from_image(temp_path)
    source = "EXIF"
    
    if lat is None or lon is None:
        if manual_lat is not None and manual_lon is not None:
            lat = manual_lat
            lon = manual_lon
            source = "USER_SELECTED"
        else:
            # Need coordinates from user if EXIF fails
            return {"status": "error", "message": "GPS metadata unavailable. Please provide manual coordinates.", "needs_manual": True}

    # Save to the actual SQLite database
    new_report = models.Report(
        reported_at=datetime.datetime.utcnow(),
        latitude=lat,
        longitude=lon,
        location_source=source,
        description=description,
        image_path=temp_path
    )
    
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    
    return {
        "status": "success", 
        "report_id": new_report.id, 
        "latitude": lat, 
        "longitude": lon, 
        "source": source
    }


@app.get("/api/incidents")
def list_incidents(db: Session = Depends(get_db)):
    incidents = db.query(models.Incident).all()
    reports = db.query(models.Report).all()
    return {"incidents": incidents, "reports": reports}


from pydantic import BaseModel

class VerificationPayload(BaseModel):
    confidence: float
    affected_area: float
    satellite_before: str
    satellite_after: str

@app.post("/api/reports/{report_id}/verify")
def verify_report_to_incident(report_id: int, payload: VerificationPayload, db: Session = Depends(get_db)):
    """Automated AI verification: Promotes an unverified report into a tracked incident based on AI confidence."""
    report = db.query(models.Report).filter(models.Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    incident = models.Incident(
        latitude=report.latitude,
        longitude=report.longitude,
        ai_confidence=payload.confidence,
        affected_area=payload.affected_area,
        satellite_before_date=payload.satellite_before,
        satellite_after_date=payload.satellite_after,
        satellite_verification_status="AI_CONFIRMED" if payload.confidence >= 0.50 else "AI_REJECTED",
        verification_status="LIKELY" if payload.confidence >= 0.50 else "REJECTED",
        authority_status="PENDING_REVIEW"
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    
    report.incident_id = incident.id
    db.commit()
    
    return {"status": "success", "incident_id": incident.id, "verification_status": incident.verification_status}


