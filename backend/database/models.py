from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .engine import Base

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=True)
    
    image_path = Column(String, nullable=False)
    description = Column(String, nullable=True)
    
    # Location details
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    location_source = Column(String, default="USER_SELECTED") # EXIF, USER_SELECTED, MANUAL
    
    # Reporting info
    reported_at = Column(DateTime(timezone=True), server_default=func.now())
    image_timestamp = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    incident = relationship("Incident", back_populates="reports")


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    
    # Aggregated location from reports
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # AI Screening results
    ai_landslide_probability = Column(Float, nullable=True)
    ai_confidence = Column(Float, nullable=True)
    
    # Satellite verification
    satellite_verification_status = Column(String, default="PENDING")
    satellite_change_score = Column(Float, nullable=True)
    satellite_before_date = Column(String, nullable=True)
    satellite_after_date = Column(String, nullable=True)
    
    # Terrain details
    terrain_risk_score = Column(Float, nullable=True)
    
    # Verdict Statuses
    overall_confidence = Column(Float, nullable=True)
    verification_status = Column(String, default="UNVERIFIED") # UNVERIFIED, LIKELY, VERIFIED, REJECTED
    severity = Column(String, default="LOW") # LOW, MODERATE, HIGH, CRITICAL
    affected_area = Column(Float, nullable=True)
    
    authority_status = Column(String, default="PENDING")
    public_status = Column(String, default="NOT_ANNOUNCED")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    reports = relationship("Report", back_populates="incident")
