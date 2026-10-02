"""
Utility modules for Features 1, 2, and 3:
1. Automated PDF & GeoJSON Incident Report Generator
2. Real-Time Rainfall & Weather Integration (Open-Meteo API)
3. Multi-Temporal Slide Progression & Recovery Tracking (NDVI Rebound / Scar Analysis)
"""
from __future__ import annotations

import io
import json
import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import requests
import PIL.Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# ============================================================
# FEATURE 1: GEOJSON & PDF REPORT GENERATION
# ============================================================

def generate_geojson_report(result: Dict[str, Any], transform, crs) -> str:
    """
    Generates a standard GeoJSON FeatureCollection string representing detected landslide polygons
    and key metadata for direct import into QGIS or ArcGIS.
    """
    import cv2
    from rasterio.warp import transform as warp_transform

    mask = result.get("change_mask")
    features = []

    if mask is not None and mask.sum() > 0:
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for idx, cnt in enumerate(contours):
            if len(cnt) < 3:
                continue
            
            # Convert contour pixel coordinates to native CRS -> WGS84
            px_cols = cnt[:, 0, 0]
            px_rows = cnt[:, 0, 1]
            
            xs, ys = [], []
            for r, c in zip(px_rows, px_cols):
                x, y = transform * (c + 0.5, r + 0.5)
                xs.append(x)
                ys.append(y)
                
            lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
            coords = [[float(ln), float(lt)] for ln, lt in zip(lons, lats)]
            # Close polygon if not closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            poly_area_km2 = float(cv2.contourArea(cnt) * (abs(transform.a * transform.e) / 1e6 if hasattr(transform, 'a') else 0.0001))
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "id": idx + 1,
                    "hazard_type": "Landslide Scar",
                    "detection_date": str(result.get("after_date", "N/A")),
                    "model_confidence": round(float(result.get("confidence", 0.0)), 4),
                    "estimated_area_km2": round(poly_area_km2, 4)
                }
            })

    geojson_doc = {
        "type": "FeatureCollection",
        "metadata": {
            "title": "LandSentry Incident Assessment Report",
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "detection_mode": result.get("detection_mode", "ai"),
            "before_date": str(result.get("before_date", "")),
            "after_date": str(result.get("after_date", "")),
            "present_date": str(result.get("present_date", "")),
            "affected_area_km2": round(float(result.get("affected_area_km2", 0.0)), 4)
        },
        "features": features
    }
    return json.dumps(geojson_doc, indent=2)


def generate_pdf_report(result: Dict[str, Any]) -> bytes:
    """
    Generates a publication-ready PDF Incident Assessment Report in memory using ReportLab.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom color palette
    NAVY = colors.HexColor("#1A2B4C")
    TEAL = colors.HexColor("#0D9488")
    TEXT_DARK = colors.HexColor("#1E293B")
    BG_LIGHT = colors.HexColor("#F8FAFC")
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=NAVY,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=TEAL,
        spaceAfter=12
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK
    )
    bold_body_style = ParagraphStyle(
        "BoldBodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=NAVY
    )

    elements = []

    # Header
    elements.append(Paragraph("🚨 LANDSENTRY OFFICIAL INCIDENT ASSESSMENT REPORT", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | Automated Disaster Intelligence", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceBefore=0, spaceAfter=10))

    # Executive Summary Table
    elements.append(Paragraph("Executive Summary", section_heading))

    top_region = result["change_regions"][0] if result.get("change_regions") else None
    loc_name = top_region["place_name"] if top_region else "Unknown Location / Clear Region"
    lat_str = f"{top_region['centroid_lat']:.6f}° N" if top_region else "N/A"
    lon_str = f"{top_region['centroid_lon']:.6f}° E" if top_region else "N/A"

    summary_data = [
        [Paragraph("Primary Location:", bold_body_style), Paragraph(f"<b>{loc_name}</b>", body_style)],
        [Paragraph("Centroid Coordinates:", bold_body_style), Paragraph(f"{lat_str}, {lon_str}", body_style)],
        [Paragraph("Estimated Affected Area:", bold_body_style), Paragraph(f"<b>{result.get('affected_area_km2', 0.0):.4f} km²</b> ({result.get('changed_pixels', 0)} pixels)", body_style)],
        [Paragraph("AI Confidence Rating:", bold_body_style), Paragraph(f"{result.get('confidence', 0.0)*100:.1f}%", body_style)],
        [Paragraph("Analysis Date Windows:", bold_body_style), Paragraph(f"Before: {result.get('before_date')} | After: {result.get('after_date')} | Present: {result.get('present_date')}", body_style)],
        [Paragraph("Present Safety Screening:", bold_body_style), Paragraph(f"<b>{result.get('present_status', {}).get('visitor_alert', 'N/A')}</b> — {result.get('present_status', {}).get('label', '')}", body_style)],
    ]

    summary_table = Table(summary_data, colWidths=[150, 390])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 10))

    # Helper to convert numpy array RGB image to ReportLab RLImage
    def numpy_to_rl_image(img_arr: np.ndarray, width_pt=104, height_pt=104) -> RLImage:
        img_uint8 = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
        pil_img = PIL.Image.fromarray(img_uint8)
        img_buf = io.BytesIO()
        pil_img.save(img_buf, format="PNG")
        img_buf.seek(0)
        return RLImage(img_buf, width=width_pt, height=height_pt)

    elements.append(Paragraph("High-Resolution Multi-Temporal Snapshots", section_heading))

    # Build side-by-side images table
    rgb_pre = result.get("rgb_pre")
    rgb_post = result.get("rgb_post")
    rgb_current = result.get("rgb_current")
    change_mask = result.get("change_mask")
    display_valid = result.get("display_valid")

    # Composite detection overlay image
    if rgb_post is not None and change_mask is not None:
        post_with_mask = rgb_post.copy()
        detected = change_mask > 0
        post_with_mask[detected] = (
            0.55 * rgb_post[detected] + 0.45 * np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        if display_valid is not None:
            post_with_mask[~display_valid] = 0.55
    else:
        post_with_mask = rgb_post

    img_pre_rl = numpy_to_rl_image(rgb_pre, 125, 125)
    img_post_rl = numpy_to_rl_image(rgb_post, 125, 125)
    img_overlay_rl = numpy_to_rl_image(post_with_mask, 125, 125)
    img_current_rl = numpy_to_rl_image(rgb_current, 125, 125)

    img_table_data = [
        [img_pre_rl, img_post_rl, img_overlay_rl, img_current_rl],
        [
            Paragraph("<b>Pre-Event</b>", body_style),
            Paragraph("<b>Post-Event</b>", body_style),
            Paragraph("<b>Detected Mask</b>", body_style),
            Paragraph("<b>Present Screening</b>", body_style)
        ]
    ]

    img_table = Table(img_table_data, colWidths=[135, 135, 135, 135])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 10))

    # Notes & Disclaimer Section
    elements.append(Paragraph("System Recommendation & Disclaimer", section_heading))
    disclaimer_text = (
        "This report was generated automatically by LandSentry using bitemporal Sentinel-2 satellite imagery "
        "and deep learning change detection algorithms. The present condition screening indicates: "
        f"<i>'{result.get('present_status', {}).get('visitor_message', 'No details available')}'</i>. "
        "This document is intended for emergency planning and rapid response. Field ground-truthing remains recommended for civil engineering actions."
    )
    elements.append(Paragraph(disclaimer_text, body_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# FEATURE 2: REAL-TIME RAINFALL & WEATHER INTEGRATION (OPEN-METEO)
# ============================================================

def fetch_historical_rainfall(
    lat: float,
    lon: float,
    start_date: datetime.date,
    end_date: datetime.date
) -> Dict[str, Any]:
    """
    Fetches daily precipitation totals (in mm) for the given AOI centroid from Open-Meteo API
    across the exact user-selected timeframe (Before Date -> After Date -> Present Date).
    """
    try:
        today = datetime.date.today()
        
        # If user picked a future end date, archive API stops at yesterday
        fetch_end = min(end_date, today - datetime.timedelta(days=1))
        
        # Open-Meteo Archive API endpoint
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": fetch_end.strftime("%Y-%m-%d"),
            "daily": "precipitation_sum,rain_sum",
            "timezone": "auto"
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        daily = data.get("daily", {})
        dates = list(daily.get("time", []))
        precip = list(daily.get("precipitation_sum", []))

        # If user selected a date range that includes today/future, append live forecast data
        if end_date >= today:
            try:
                forecast_url = "https://api.open-meteo.com/v1/forecast"
                fc_params = {
                    "latitude": round(lat, 4),
                    "longitude": round(lon, 4),
                    "start_date": today.strftime("%Y-%m-%d"),
                    "end_date": min(end_date, today + datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
                    "daily": "precipitation_sum",
                    "timezone": "auto"
                }
                fc_resp = requests.get(forecast_url, params=fc_params, timeout=5)
                if fc_resp.status_code == 200:
                    fc_daily = fc_resp.json().get("daily", {})
                    fc_dates = fc_daily.get("time", [])
                    fc_precip = fc_daily.get("precipitation_sum", [])
                    for d, p in zip(fc_dates, fc_precip):
                        if d not in dates:
                            dates.append(d)
                            precip.append(p)
            except Exception:
                pass  # Fallback gracefully
        
        total_precip = sum(p for p in precip if p is not None)
        max_daily_precip = max((p for p in precip if p is not None), default=0.0)

        return {
            "dates": dates,
            "precipitation_mm": [p if p is not None else 0.0 for p in precip],
            "total_precip_mm": float(total_precip),
            "max_daily_precip_mm": float(max_daily_precip),
            "error": None
        }
    except Exception as e:
        return {
            "dates": [],
            "precipitation_mm": [],
            "total_precip_mm": 0.0,
            "max_daily_precip_mm": 0.0,
            "error": f"Unable to fetch weather data: {str(e)}"
        }


# ============================================================
# FEATURE 3: MULTI-TEMPORAL SLIDE PROGRESSION & RECOVERY TRACKING
# ============================================================

def compute_progression_metrics(
    pre_arr: np.ndarray,
    post_arr: np.ndarray,
    current_arr: np.ndarray,
    change_mask: np.ndarray
) -> Dict[str, Any]:
    """
    Calculates NDVI stats and vegetation rebound / scar expansion dynamics across Pre, Post, and Present timestamps.
    """
    # Channel 0: Red, Channel 3: NIR
    red_pre, nir_pre = pre_arr[..., 0].astype(np.float32), pre_arr[..., 3].astype(np.float32)
    red_post, nir_post = post_arr[..., 0].astype(np.float32), post_arr[..., 3].astype(np.float32)
    red_curr, nir_curr = current_arr[..., 0].astype(np.float32), current_arr[..., 3].astype(np.float32)

    ndvi_pre = np.divide(nir_pre - red_pre, nir_pre + red_pre + 1e-6, out=np.zeros_like(red_pre), where=(nir_pre+red_pre)!=0)
    ndvi_post = np.divide(nir_post - red_post, nir_post + red_post + 1e-6, out=np.zeros_like(red_post), where=(nir_post+red_post)!=0)
    ndvi_curr = np.divide(nir_curr - red_curr, nir_curr + red_curr + 1e-6, out=np.zeros_like(red_curr), where=(nir_curr+red_curr)!=0)

    # Focus on detected landslide pixels or whole AOI if no mask
    if change_mask is not None and change_mask.sum() > 0:
        eval_mask = change_mask > 0
    else:
        eval_mask = np.ones(ndvi_pre.shape, dtype=bool)

    mean_ndvi_pre = float(np.mean(ndvi_pre[eval_mask]))
    mean_ndvi_post = float(np.mean(ndvi_post[eval_mask]))
    mean_ndvi_curr = float(np.mean(ndvi_curr[eval_mask]))

    initial_drop = mean_ndvi_pre - mean_ndvi_post
    recovery = mean_ndvi_curr - mean_ndvi_post

    if initial_drop > 0.05 and recovery > 0.03:
        status_text = "🌿 Vegetation Recovery / NDVI Rebound"
        status_color = "green"
    elif mean_ndvi_curr < mean_ndvi_post - 0.03:
        status_text = "⚠️ Secondary Slope Collapse / Scar Expansion"
        status_color = "red"
    else:
        status_text = "⚖️ Stabilized Landslide Scar"
        status_color = "blue"

    return {
        "ndvi_pre": mean_ndvi_pre,
        "ndvi_post": mean_ndvi_post,
        "ndvi_curr": mean_ndvi_curr,
        "ndvi_pre_map": ndvi_pre,
        "ndvi_post_map": ndvi_post,
        "ndvi_curr_map": ndvi_curr,
        "status_text": status_text,
        "status_color": status_color,
        "rebound_val": float(recovery)
    }
