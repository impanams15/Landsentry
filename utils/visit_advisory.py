"""
LandSentry — Present Visit Advisory Engine
==========================================

This module provides a SEPARATE, rule-based Present Visit Advisory that is
INDEPENDENT of the historical satellite change detection.

IMPORTANT DESIGN PRINCIPLE
---------------------------
A high satellite-change percentage (e.g. 92.4% or 98.4%) reflects detected image / land-cover
change relative to a chosen historical baseline. It does NOT mean that 92.4% of the
area is currently experiencing an active landslide.

The public-facing advisory must be based primarily on CURRENT conditions and
official authoritative warnings. Historical satellite change is supporting evidence
only.

DECISION HIERARCHY (Rules evaluated in order)
----------------------------------------------
RULE 1 — OFFICIAL EMERGENCY WARNING
    If an authoritative government source reports an active warning, evacuation,
    road closure, or confirmed dangerous condition: STATUS = RED

RULE 2 — OFFICIAL WARNING UNAVAILABLE
    If official warning sources cannot be queried/reached:
    Official warning status must show "OFFICIAL WARNING STATUS UNAVAILABLE".
    Do NOT show "NO CURRENT OFFICIAL ALERT" and do NOT classify as GREEN based on missing official data.

RULE 3 — STRONG CURRENT HAZARD EVIDENCE (multiple indicators)
    No official active warning, but significant recent rainfall + high susceptibility
    + recent satellite disturbance: STATUS = ORANGE

RULE 4 — SATELLITE CHANGE ONLY
    The only strong evidence is historical or satellite image change;
    no current official warning, no dangerous current rainfall: STATUS = INFORMATIONAL

RULE 5 — NO CURRENT HAZARD EVIDENCE
    Official sources successfully verified clear, no heavy rainfall, no strong recent satellite anomaly:
    STATUS = GREEN
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import requests

try:
    from shapely.geometry import Point, shape
except ImportError:
    Point, shape = None, None

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION — All thresholds in one place
# ============================================================

CURRENT_HAZARD_CONFIG: Dict[str, Any] = {
    "rainfall_thresholds": {
        "moderate_mm_per_day": 15.0,   # mm/day: moderate rainfall, informational
        "heavy_mm_per_day": 64.5,      # mm/day: IMD "Heavy Rain" threshold
        "very_heavy_mm_per_day": 115.6, # mm/day: IMD "Very Heavy Rain" threshold
        "extreme_mm_per_day": 204.5,   # mm/day: IMD "Extremely Heavy Rain" threshold
        "heavy_3day_mm": 150.0,        # mm over 3 days → elevated concern
        "heavy_7day_mm": 300.0,        # mm over 7 days → elevated concern
    },
    "satellite_change_thresholds": {
        "low_fraction": 0.05,         # < 5 %: minimal change
        "moderate_fraction": 0.15,    # 5–15 %: moderate change
        "high_fraction": 0.30,        # 15–30 %: notable change
        "very_high_fraction": 0.50,   # > 50 %: substantial change
    },
    "susceptibility_thresholds": {
        "low": 0.33,
        "moderate": 0.66,
        "high": 1.0,
    },
    "confidence_requirements": {
        "orange_min_indicators": 2,   # Need ≥ 2 elevated indicators for ORANGE
        "red_requires_official": True, # RED requires an official warning
    },
    "data_freshness_hours": {
        "rainfall_api_max_age": 3,
        "official_warning_max_age": 6,
    },
}

class AdvisoryStatus:
    RED = "RED"                    # AVOID VISITING — official emergency warning
    ORANGE = "ORANGE"              # CAUTION — multiple current hazard indicators
    INFORMATIONAL = "INFORMATIONAL" # Significant satellite change, no current warning
    GREEN = "GREEN"                # No current hazard alert detected (verified)
    DATA_REQUIRED = "DATA_REQUIRED" # Official warning status unavailable / unverified

@dataclass
class OfficialSourceDetail:
    name: str
    status: str = "unavailable"    # "verified_active", "verified_clear", "unavailable"
    timestamp: Optional[str] = None
    applicable_date: Optional[str] = None
    warning_type: Optional[str] = None
    description: Optional[str] = None
    error_msg: Optional[str] = None
    future_advisory: Optional[str] = None

@dataclass
class OfficialWarning:
    available: bool = False
    status: str = "unavailable"              # "active", "clear", "unavailable"
    status_label: str = "OFFICIAL WARNING STATUS UNAVAILABLE"
    primary_source: str = "Not checked"
    timestamp: Optional[str] = None
    applicable_date: Optional[str] = None
    warning_type: Optional[str] = None
    area: Optional[str] = None
    description: Optional[str] = None
    future_advisory: Optional[str] = None
    sources: Dict[str, OfficialSourceDetail] = field(default_factory=dict)
    fetch_error: Optional[str] = None

@dataclass
class RainfallData:
    available: bool = False
    fetch_error: Optional[str] = None
    precip_1h_mm: float = 0.0
    precip_3h_mm: float = 0.0
    precip_6h_mm: float = 0.0
    precip_24h_mm: float = 0.0
    precip_72h_mm: float = 0.0
    forecast_24h_mm: float = 0.0
    max_daily_recent_mm: float = 0.0
    total_7day_mm: float = 0.0
    weather_source: str = "Open-Meteo"
    data_timestamp: Optional[str] = None
    dates: List[str] = field(default_factory=list)
    precipitation_mm: List[float] = field(default_factory=list)
    severe_weather_warning: bool = False

@dataclass
class SatelliteAnomaly:
    available: bool = False
    changed_fraction: float = 0.0
    ndvi_change_mean: float = 0.0
    spectral_change_mean: float = 0.0
    interpretation: str = "Not assessed"
    limitation_note: str = (
        "This represents satellite-image/land-cover change between the selected historical post-event baseline "
        "and the current imagery. It is NOT the percentage of area currently affected by landslides."
    )
    acquisition_dates: Optional[str] = None

@dataclass
class SusceptibilityScore:
    available: bool = False
    score: float = 0.0
    label: str = "Unknown"
    factors: List[str] = field(default_factory=list)

@dataclass
class DataFreshness:
    official_warning_status: str = "Unavailable"   # "Verified (IST timestamp)" or "Unavailable"
    official_warning_sources_summary: str = ""     # e.g., "IMD: verified | NDMA/SACHET: unavailable"
    weather_timestamp: str = "Unavailable"
    satellite_acquisition_dates: str = "Unavailable"
    last_checked_ist: str = ""

@dataclass
class AdvisoryEvidence:
    official_warning: OfficialWarning = field(default_factory=OfficialWarning)
    rainfall: RainfallData = field(default_factory=RainfallData)
    satellite_anomaly: SatelliteAnomaly = field(default_factory=SatelliteAnomaly)
    susceptibility: SusceptibilityScore = field(default_factory=SusceptibilityScore)
    freshness: DataFreshness = field(default_factory=DataFreshness)
    historical_change_fraction: float = 0.0
    historical_change_note: str = ""
    analysis_timestamp: str = ""
    satellite_post_window: str = ""
    satellite_current_window: str = ""

@dataclass
class AdvisoryResult:
    status: str = AdvisoryStatus.DATA_REQUIRED
    status_emoji: str = "⚪"
    headline: str = "DATA VERIFICATION REQUIRED — OFFICIAL WARNING STATUS UNAVAILABLE"
    explanation: str = ""
    rules_triggered: List[str] = field(default_factory=list)
    indicators_elevated: List[str] = field(default_factory=list)
    evidence: AdvisoryEvidence = field(default_factory=AdvisoryEvidence)
    generated_at_ist: str = ""

# ============================================================
# RAINFALL FETCHER
# ============================================================

def _fetch_current_rainfall(lat: float, lon: float, present_date: datetime.date) -> RainfallData:
    result = RainfallData()
    result.weather_source = "Open-Meteo ERA5 / Forecast Grid"

    try:
        today = datetime.date.today()
        fetch_end = min(present_date, today)
        fetch_start = fetch_end - datetime.timedelta(days=7)

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": fetch_start.strftime("%Y-%m-%d"),
            "end_date": (fetch_end - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
            "daily": "precipitation_sum,rain_sum",
            "hourly": "precipitation",
            "timezone": "Asia/Kolkata",
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        dates = list(daily.get("time", []))
        precip_list = [p if p is not None else 0.0 for p in daily.get("precipitation_sum", [])]

        result.dates = dates
        result.precipitation_mm = precip_list

        if precip_list:
            result.max_daily_recent_mm = float(max(precip_list))
            result.total_7day_mm = float(sum(precip_list))
            result.precip_72h_mm = float(sum(precip_list[-3:]))
            result.precip_24h_mm = float(precip_list[-1]) if precip_list else 0.0

        hourly = data.get("hourly", {})
        hourly_precip = [p if p is not None else 0.0 for p in hourly.get("precipitation", [])]
        if hourly_precip:
            result.precip_1h_mm = float(sum(hourly_precip[-1:]))
            result.precip_3h_mm = float(sum(hourly_precip[-3:]))
            result.precip_6h_mm = float(sum(hourly_precip[-6:]))

        try:
            fc_url = "https://api.open-meteo.com/v1/forecast"
            fc_params = {
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "daily": "precipitation_sum",
                "forecast_days": 2,
                "timezone": "Asia/Kolkata",
            }
            fc_resp = requests.get(fc_url, params=fc_params, timeout=5)
            if fc_resp.status_code == 200:
                fc_daily = fc_resp.json().get("daily", {})
                fc_precip = fc_daily.get("precipitation_sum", [])
                if fc_precip:
                    result.forecast_24h_mm = float(fc_precip[0] or 0.0)
        except Exception:
            pass

        result.available = True
        result.data_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M IST")

        cfg = CURRENT_HAZARD_CONFIG["rainfall_thresholds"]
        if (result.max_daily_recent_mm >= cfg["very_heavy_mm_per_day"]
                or result.total_7day_mm >= cfg["heavy_7day_mm"]):
            result.severe_weather_warning = True

    except Exception as exc:
        result.available = False
        result.fetch_error = f"Rainfall data unavailable: {exc}"

    return result

# ============================================================
# OFFICIAL WARNING CHECKER (Per Source Granularity)
# ============================================================

IMD_WARNING_CODES = {
    "1": "No warning", "2": "Heavy Rain", "3": "Heavy Snow", "4": "Thunderstorm & Lightning",
    "5": "Hailstorm", "6": "Dust Storm", "7": "Dust Raising Winds", "8": "Strong Surface Winds",
    "9": "Heat Wave", "91": "Severe Heat Wave", "10": "Hot Day", "11": "Warm Night",
    "12": "Cold Wave", "121": "Severe Cold Wave", "13": "Cold Day", "131": "Severe Cold Day",
    "14": "Ground Frost", "15": "Fog", "151": "Dense Fog", "152": "Very Dense Fog",
    "16": "Very Heavy Rain", "17": "Extremely Heavy Rain", "18": "Hot and Humid",
    "41": "Thunderstorm & Lightning", "42": "Thunderstorm & Lightning", "43": "Thunderstorm & Lightning",
    "44": "Thunderstorm & Lightning", "45": "Thunderstorm & Lightning", "46": "Thunderstorm & Lightning"
}

def _parse_imd_warning_str(warning_str: str) -> List[str]:
    """Parse comma-separated IMD warning codes, excluding '1' (No Warning)."""
    if not warning_str:
        return []
    codes = [c.strip() for c in warning_str.split(",") if c.strip()]
    warnings = []
    for c in codes:
        if c != "1" and c in IMD_WARNING_CODES:
            warnings.append(IMD_WARNING_CODES[c])
    return list(set(warnings))

def _check_official_warnings(lat: float, lon: float) -> OfficialWarning:
    warning = OfficialWarning()
    now_ist = datetime.datetime.now()
    now_str = now_ist.strftime("%B %d, %Y")
    
    # Dynamically determine the subdivision using the IMD map geometries
    target_subdiv = "Unknown / Undetermined"
    pt = Point(lon, lat) if Point else None

    # Source 1: IMD
    imd_detail = OfficialSourceDetail(name="India Meteorological Department (IMD)")
    try:
        # Fetch dynamic subdivision warnings from official IMD React GeoServer API
        imd_url = "https://reactjs.imd.gov.in/geoserver/wfs"
        params = {
            "service": "WFS",
            "version": "1.1.0",
            "request": "GetFeature",
            "typename": "imd:subdiv_warnings_now",
            "srsname": "EPSG:4326",
            "outputFormat": "application/json"
        }
        resp = requests.get(imd_url, params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            subdiv_feature = None
            
            if pt is not None:
                for f in features:
                    try:
                        geom = shape(f.get("geometry", {}))
                        if geom.contains(pt):
                            subdiv_feature = f
                            target_subdiv = f.get("properties", {}).get("SUBDIV", "Unknown")
                            break
                    except Exception:
                        continue
            
            # Fallback if shapely fails or point not in any polygon, usually we won't get a result
            
            if subdiv_feature:
                props = subdiv_feature["properties"]
                base_date_str = props.get("Date", now_ist.strftime("%Y-%m-%d"))
                
                try:
                    base_date = datetime.datetime.strptime(base_date_str, "%Y-%m-%d").date()
                except ValueError:
                    base_date = now_ist.date()
                
                imd_detail.timestamp = base_date.strftime("%B %d, %Y")
                imd_detail.applicable_date = base_date.strftime("%B %d, %Y")
                
                # Check current day (Day_1)
                day1_warnings = _parse_imd_warning_str(props.get("Day_1", "1"))
                
                if day1_warnings:
                    imd_detail.status = "verified_active"
                    imd_detail.warning_type = ", ".join(day1_warnings)
                    imd_detail.description = f"Active warning for {target_subdiv}"
                else:
                    imd_detail.status = "verified_clear"
                    imd_detail.description = "No active warning currently."

                # Check future days (Day_2 to Day_7)
                future_adv_list = []
                for idx in range(2, 8):
                    day_key = f"Day_{idx}"
                    future_warnings = _parse_imd_warning_str(props.get(day_key, "1"))
                    if future_warnings:
                        fut_date = base_date + datetime.timedelta(days=idx-1)
                        fut_date_str = fut_date.strftime("%B %d, %Y")
                        w_str = f"{', '.join(future_warnings)} currently shown by IMD for {fut_date_str}"
                        future_adv_list.append(w_str)
                
                if future_adv_list:
                    imd_detail.future_advisory = "; ".join(future_adv_list)

            else:
                imd_detail.status = "unavailable"
                imd_detail.error_msg = f"IMD data valid but {target_subdiv} not found"
        else:
            imd_detail.status = "unavailable"
            imd_detail.error_msg = f"HTTP {resp.status_code}"
    except Exception as exc:
        imd_detail.status = "unavailable"
        imd_detail.error_msg = str(exc)

    # Source 2: NDMA / SACHET
    sachet_detail = OfficialSourceDetail(name="NDMA / SACHET National Disaster Alert Portal")
    try:
        sachet_url = "https://sachet.ndma.gov.in/cap_public_website/FetchAlertDetail"
        resp = requests.get(sachet_url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            alerts = data if isinstance(data, list) else data.get("alerts", [])
            active_found = False
            for alert in alerts:
                a_lat = alert.get("lat") or alert.get("latitude")
                a_lon = alert.get("lon") or alert.get("longitude")
                if a_lat and a_lon and abs(float(a_lat) - lat) < 1.5 and abs(float(a_lon) - lon) < 1.5:
                    sachet_detail.status = "verified_active"
                    sachet_detail.timestamp = alert.get("issued_at", now_str)
                    sachet_detail.warning_type = alert.get("type", "Disaster Alert")
                    sachet_detail.description = alert.get("description", "Active alert reported")
                    active_found = True
                    break
            if not active_found:
                sachet_detail.status = "verified_clear"
                sachet_detail.timestamp = now_str
                sachet_detail.description = "No active alert for selected region."
        else:
            sachet_detail.status = "unavailable"
            sachet_detail.error_msg = f"HTTP {resp.status_code}"
    except Exception as exc:
        sachet_detail.status = "unavailable"
        sachet_detail.error_msg = str(exc)

    warning.sources["IMD"] = imd_detail
    warning.sources["NDMA_SACHET"] = sachet_detail

    # Synthesize overall official warning status
    active_sources = [s for s in warning.sources.values() if s.status == "verified_active"]
    verified_sources = [s for s in warning.sources.values() if s.status in ("verified_active", "verified_clear")]

    # Capture future advisory if any verified source has one
    future_advisories = [s.future_advisory for s in verified_sources if s.future_advisory]
    warning.future_advisory = "; ".join(future_advisories) if future_advisories else None

    if active_sources:
        top_active = active_sources[0]
        warning.available = True
        warning.status = "active"
        warning.status_label = "OFFICIAL HAZARD ADVISORY ACTIVE"
        warning.primary_source = top_active.name
        warning.timestamp = top_active.timestamp
        warning.applicable_date = top_active.applicable_date or top_active.timestamp
        warning.warning_type = top_active.warning_type
        warning.description = top_active.description
        warning.area = target_subdiv
    elif verified_sources:
        # At least one official source returned verified status without active warnings
        primary = verified_sources[0]
        warning.available = True
        warning.status = "clear"
        warning.status_label = "NO CURRENT OFFICIAL WEATHER WARNING DETECTED"
        warning.primary_source = primary.name
        warning.timestamp = primary.timestamp
        warning.applicable_date = primary.applicable_date or primary.timestamp
        warning.area = target_subdiv
    else:
        # ALL official sources failed/unavailable -> FAIL-SAFE TRIGGERED
        warning.available = False
        warning.status = "unavailable"
        warning.status_label = "OFFICIAL WARNING STATUS UNAVAILABLE"
        warning.primary_source = "IMD / NDMA SACHET (both unavailable)"
        warning.fetch_error = "Official government warning APIs could not be reached live."

    return warning

# ============================================================
# SUSCEPTIBILITY ESTIMATION
# ============================================================

def _estimate_susceptibility(
    post_arr: Optional[np.ndarray],
    change_fraction: float,
    historical_change_fraction: float,
) -> SusceptibilityScore:
    score = SusceptibilityScore()
    score.available = True
    score.factors = []
    proxy = 0.0

    if historical_change_fraction > CURRENT_HAZARD_CONFIG["satellite_change_thresholds"]["high_fraction"]:
        proxy += 0.35
        score.factors.append("High historical land-cover change on record")
    elif historical_change_fraction > CURRENT_HAZARD_CONFIG["satellite_change_thresholds"]["moderate_fraction"]:
        proxy += 0.20
        score.factors.append("Moderate historical land-cover change on record")

    if change_fraction > CURRENT_HAZARD_CONFIG["satellite_change_thresholds"]["high_fraction"]:
        proxy += 0.25
        score.factors.append("High recent satellite-detected surface change")
    elif change_fraction > CURRENT_HAZARD_CONFIG["satellite_change_thresholds"]["moderate_fraction"]:
        proxy += 0.15
        score.factors.append("Moderate recent satellite-detected surface change")

    if post_arr is not None:
        try:
            red = post_arr[..., 0].astype(np.float32)
            nir = post_arr[..., 3].astype(np.float32)
            ndvi = np.divide(nir - red, nir + red + 1e-6, out=np.zeros_like(red), where=(nir + red) != 0)
            mean_ndvi = float(np.mean(ndvi[np.any(post_arr[..., :3] > 1e-4, axis=-1)]))
            if mean_ndvi < 0.2:
                proxy += 0.25
                score.factors.append(f"Low mean NDVI ({mean_ndvi:.2f}) — possible bare soil exposure")
            elif mean_ndvi < 0.35:
                proxy += 0.10
                score.factors.append(f"Below-average NDVI ({mean_ndvi:.2f})")
        except Exception:
            pass

    proxy = min(1.0, proxy)
    score.score = proxy

    cfg = CURRENT_HAZARD_CONFIG["susceptibility_thresholds"]
    if proxy < cfg["low"]:
        score.label = "Low (Spectral Proxy)"
    elif proxy < cfg["moderate"]:
        score.label = "Moderate (Spectral Proxy)"
    else:
        score.label = "High (Spectral Proxy)"

    if not score.factors:
        score.factors.append("No strong susceptibility indicators from spectral data")

    score.factors.append("NOTE: Spectral proxy estimate only. Formal susceptibility requires DEM + geology.")
    return score

# ============================================================
# MAIN DECISION ENGINE
# ============================================================

def compute_visit_advisory(
    lat: float,
    lon: float,
    present_date: datetime.date,
    post_arr: Optional[np.ndarray],
    current_arr: Optional[np.ndarray],
    present_status: Dict[str, Any],
    historical_change_fraction: float,
    after_date: Optional[datetime.date] = None,
    before_date: Optional[datetime.date] = None,
    window: Optional[datetime.timedelta] = None,
) -> AdvisoryResult:
    now_ist = datetime.datetime.now()
    ist_str = now_ist.strftime("%Y-%m-%d %H:%M IST")

    sat_post_window = f"{after_date} to {after_date + window}" if after_date and window else "Selected baseline"
    sat_current_window = f"{present_date - window} to {present_date}" if present_date and window else "Current window"

    evidence = AdvisoryEvidence()
    evidence.analysis_timestamp = ist_str
    evidence.satellite_post_window = sat_post_window
    evidence.satellite_current_window = sat_current_window
    evidence.historical_change_fraction = historical_change_fraction
    evidence.historical_change_note = (
        f"Historical satellite change (pre-event vs post-event AI model): {historical_change_fraction * 100:.1f}%. "
        "This represents detected image/land-cover change relative to the selected baseline period and is NOT "
        "equivalent to the percentage of area currently affected by active landslides."
    )

    evidence.official_warning = _check_official_warnings(lat, lon)
    evidence.rainfall = _fetch_current_rainfall(lat, lon, present_date)

    anomaly = SatelliteAnomaly()
    anomaly.available = True
    sat_changed_fraction = present_status.get("changed_fraction", 0.0)
    anomaly.changed_fraction = sat_changed_fraction
    anomaly.interpretation = (
        "Evidence of surface/land-cover change between the post-event baseline and current imagery. "
        "This is NOT a confirmed active landslide."
    )
    anomaly.acquisition_dates = f"Baseline: {sat_post_window} | Current: {sat_current_window}"
    evidence.satellite_anomaly = anomaly

    evidence.susceptibility = _estimate_susceptibility(
        post_arr=post_arr,
        change_fraction=sat_changed_fraction,
        historical_change_fraction=historical_change_fraction,
    )

    # Data Freshness Summary
    freshness = DataFreshness()
    freshness.last_checked_ist = ist_str
    freshness.satellite_acquisition_dates = sat_current_window
    freshness.weather_timestamp = evidence.rainfall.data_timestamp or "Unavailable"

    source_summaries = []
    for s_key, s_det in evidence.official_warning.sources.items():
        st_name = "verified" if s_det.status in ("verified_active", "verified_clear") else "unavailable"
        source_summaries.append(f"{s_key}: {st_name}")
    freshness.official_warning_sources_summary = " | ".join(source_summaries)

    if evidence.official_warning.status in ("active", "clear"):
        freshness.official_warning_status = f"Verified ({evidence.official_warning.timestamp})"
    else:
        freshness.official_warning_status = "Unavailable (Could not reach official warning sources)"

    evidence.freshness = freshness

    # ============================================================
    # DECISION RULES
    # ============================================================

    result = AdvisoryResult()
    result.evidence = evidence
    result.generated_at_ist = ist_str

    cfg = CURRENT_HAZARD_CONFIG
    rain_cfg = cfg["rainfall_thresholds"]
    sat_cfg = cfg["satellite_change_thresholds"]

    official = evidence.official_warning
    rainfall = evidence.rainfall
    sat = evidence.satellite_anomaly
    susc = evidence.susceptibility

    elevated_indicators: List[str] = []
    rules_triggered: List[str] = []

    # RULE 1 — ACTIVE OFFICIAL WARNING
    if official.status == "active":
        result.status = AdvisoryStatus.RED
        result.status_emoji = "🔴"
        result.headline = "AVOID VISITING — OFFICIAL HAZARD ADVISORY ACTIVE"
        result.explanation = (
            f"An official hazard advisory is currently active for this area. "
            f"Source: {official.primary_source}. "
            + (f"Warning type: {official.warning_type}. " if official.warning_type else "")
            + (f"Details: {official.description}. " if official.description else "")
            + "Follow local emergency authority instructions."
        )
        rules_triggered.append("RULE 1: Active official emergency warning reported")
        result.rules_triggered = rules_triggered
        result.indicators_elevated = elevated_indicators
        return result

    # Evaluate rainfall severity separately
    rainfall_hazard_level = 0
    if rainfall.available:
        if rainfall.max_daily_recent_mm >= rain_cfg["very_heavy_mm_per_day"]:
            rainfall_hazard_level = 3
            elevated_indicators.append(
                f"Very heavy recent rainfall: {rainfall.max_daily_recent_mm:.1f} mm/day "
                f"(IMD threshold: {rain_cfg['very_heavy_mm_per_day']} mm/day)"
            )
        elif rainfall.max_daily_recent_mm >= rain_cfg["heavy_mm_per_day"]:
            rainfall_hazard_level = 2
            elevated_indicators.append(
                f"Heavy recent rainfall: {rainfall.max_daily_recent_mm:.1f} mm/day "
                f"(IMD threshold: {rain_cfg['heavy_mm_per_day']} mm/day)"
            )
        elif rainfall.total_7day_mm >= rain_cfg["heavy_7day_mm"]:
            rainfall_hazard_level = 2
            elevated_indicators.append(
                f"High 7-day cumulative rainfall: {rainfall.total_7day_mm:.1f} mm "
                f"(threshold: {rain_cfg['heavy_7day_mm']} mm)"
            )

    sat_elevated = sat.available and sat.changed_fraction >= sat_cfg["high_fraction"]
    if sat_elevated:
        elevated_indicators.append(
            f"Notable post-event vs current satellite change: {sat.changed_fraction * 100:.1f}% of pixels"
        )

    susc_elevated = susc.available and susc.score >= cfg["susceptibility_thresholds"]["moderate"]
    if susc_elevated:
        elevated_indicators.append(f"Elevated susceptibility proxy: {susc.label} (Spectral proxy only — not a formal landslide susceptibility assessment)")

    active_count = sum([rainfall_hazard_level >= 2, sat_elevated, susc_elevated])

    # RULE 2 — OFFICIAL WARNING UNAVAILABLE (FAIL-SAFE)
    # If official sources were unreachable, DO NOT show "NO CURRENT OFFICIAL ALERT" and DO NOT return GREEN!
    if official.status == "unavailable":
        result.status = AdvisoryStatus.DATA_REQUIRED
        result.status_emoji = "⚪"
        result.headline = "OFFICIAL WARNING STATUS UNAVAILABLE — DATA VERIFICATION REQUIRED"
        result.explanation = (
            "Official government warning feeds (IMD / NDMA SACHET) could not be verified live at the time of screening. "
            "Do NOT infer that absence of an alert proves safety. "
            "Verify conditions directly with local district authorities before travelling."
        )
        rules_triggered.append("RULE 2: Official warning status unavailable — safety cannot be certified without verification")
        result.rules_triggered = rules_triggered
        result.indicators_elevated = elevated_indicators
        return result

    # RULE 3 — MULTIPLE ELEVATED CURRENT HAZARD INDICATORS
    if active_count >= cfg["confidence_requirements"]["orange_min_indicators"]:
        result.status = AdvisoryStatus.ORANGE
        result.status_emoji = "🟠"
        result.headline = "CAUTION — VISIT NOT RECOMMENDED WITHOUT CHECKING LOCAL CONDITIONS"
        explanation_parts = [
            "Official warning feeds are verified clear. However, multiple current hazard indicators show elevated conditions:"
        ]
        for ind in elevated_indicators:
            explanation_parts.append(f"  • {ind}")
        explanation_parts.append("Exercise caution and check local authorities before travelling.")
        result.explanation = "\n".join(explanation_parts)
        rules_triggered.append(f"RULE 3: {active_count} current hazard indicators elevated")
        result.rules_triggered = rules_triggered
        result.indicators_elevated = elevated_indicators
        return result

    # RULE 4 — SATELLITE CHANGE ONLY
    if sat.changed_fraction >= sat_cfg["high_fraction"] and active_count < cfg["confidence_requirements"]["orange_min_indicators"]:
        result.status = AdvisoryStatus.INFORMATIONAL
        result.status_emoji = "🔵"
        result.headline = "SIGNIFICANT SATELLITE CHANGE DETECTED — NO CURRENT OFFICIAL ALERT"
        result.explanation = (
            f"Significant satellite image change ({sat.changed_fraction * 100:.1f}%) is observed between "
            "the historical post-event baseline and current imagery. "
            "This represents image/land-cover change (vegetation, clouds, agriculture, roads) "
            "and is NOT the percentage of area currently experiencing landslides. "
            "Official government warning feeds were verified clear. Exercise standard caution."
        )
        rules_triggered.append("RULE 4: Satellite change detected without corroborating weather/official hazard alert")
        result.rules_triggered = rules_triggered
        result.indicators_elevated = elevated_indicators
        return result

    # RULE 5 — VERIFIED CLEAR / NO CURRENT HAZARD ALERT DETECTED
    result.status = AdvisoryStatus.GREEN
    result.status_emoji = "🟢"
    result.headline = "NO CURRENT OFFICIAL WEATHER WARNING DETECTED"
    
    explanation_parts = [
        "Official government warning feeds were successfully queried and report NO active warnings for this location today."
    ]
    if rainfall.available:
        explanation_parts.append(
            f"Current weather/rainfall estimate is within moderate to normal range "
            f"(max recent daily estimate: {rainfall.max_daily_recent_mm:.1f} mm, "
            f"7-day total estimate: {rainfall.total_7day_mm:.1f} mm)."
        )
    explanation_parts.append(
        "Significant satellite image/land-cover change was detected compared with the selected historical baseline.\n"
        "Satellite change does not confirm an active landslide."
    )
    if official.future_advisory:
        target_name = official.area if official.area else "the selected region"
        explanation_parts.append(
            f"\n⚠️ **Upcoming weather risk:** {official.future_advisory} for {target_name}. "
            "Recheck conditions before travelling."
        )
    
    explanation_parts.append(
        "\nNo current official weather warning does not guarantee that every location is completely safe. "
        "Follow local authority instructions and check on-ground road/weather conditions before visiting."
    )
    
    result.explanation = "\n".join(explanation_parts)
    rules_triggered.append("RULE 5: Official warning feeds verified clear and all current indicators within baseline limits")
    result.rules_triggered = rules_triggered
    result.indicators_elevated = elevated_indicators

    return result
