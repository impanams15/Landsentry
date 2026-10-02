"""Present-day satellite change status derived from post-event and current imagery.

IMPORTANT TERMINOLOGY
---------------------
The values returned here represent SATELLITE IMAGE CHANGE — specifically, the
spectral/NDVI difference between the post-event composite and the current
composite.

They do NOT represent:
  • Percentage of the area actively landsliding
  • Confirmation that a landslide is currently occurring
  • A public emergency alert level

For the public-facing "Present Visit Advisory" (with official warnings,
rainfall, and susceptibility), see utils/visit_advisory.py.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def compute_present_status(post_arr: np.ndarray, current_arr: np.ndarray) -> Dict[str, object]:
    """Compare post-event and current RGB+NIR imagery on their shared grid.

    Returns a screening signal with clearly labelled satellite-change metrics.
    This is supporting evidence for the Visit Advisory engine, NOT the advisory
    decision itself.
    """
    post = post_arr[..., :4].astype(np.float32)
    current = current_arr[..., :4].astype(np.float32)
    post_valid = np.any(post[..., :3] > 1e-4, axis=-1)
    current_valid = np.any(current[..., :3] > 1e-4, axis=-1)
    valid = post_valid & current_valid
    valid_count = int(valid.sum())
    total_count = int(valid.size)

    if valid_count == 0:
        return {
            "status": "insufficient_evidence",
            "label": "Insufficient imagery",
            "message": (
                "The post-event and present composites do not share enough valid pixels. "
                "This is a data-quality limitation, not an active hazard finding."
            ),
            # Legacy alert fields — kept for compatibility but renamed to be accurate
            "visitor_alert": "SATELLITE DATA INSUFFICIENT",
            "visitor_message": (
                "Not enough usable satellite imagery is available to assess current "
                "surface conditions. Check local authority advisories directly."
            ),
            "valid_fraction": 0.0,
            "changed_fraction": 0.0,
            "mask": np.zeros(valid.shape, dtype=np.uint8),
            # New explicit satellite-change fields
            "satellite_change_label": "Data Insufficient",
            "satellite_change_note": (
                "Imagery coverage below threshold for reliable change detection."
            ),
        }

    red_post, nir_post = post[..., 0], post[..., 3]
    red_current, nir_current = current[..., 0], current[..., 3]
    ndvi_post = (nir_post - red_post) / (nir_post + red_post + 1e-6)
    ndvi_current = (nir_current - red_current) / (nir_current + red_current + 1e-6)
    ndvi_change = np.abs(ndvi_post - ndvi_current)
    spectral_change = np.mean(np.abs(post - current), axis=-1)
    change_mask = ((ndvi_change > 0.08) | (spectral_change > 0.12)) & valid
    changed_fraction = float(change_mask.sum() / valid_count)

    # ----------------------------------------------------------------
    # Satellite-change classification
    # These labels describe WHAT THE SATELLITE SEES, not whether a
    # landslide is currently happening.
    # ----------------------------------------------------------------
    if changed_fraction > 0.15:
        status = "significant_satellite_change"
        label = "Significant Satellite Change Detected"
        message = (
            f"{changed_fraction * 100:.1f}% of valid pixels show spectral/NDVI change "
            "between the post-event baseline and current imagery. "
            "This indicates observed surface/land-cover change. "
            "Clouds, shadows, seasonal vegetation, agriculture, construction, and "
            "illumination differences can all cause satellite image change. "
            "This does NOT confirm an active landslide."
        )
        satellite_change_label = f"{changed_fraction * 100:.1f}% Satellite Change Detected"
        satellite_change_note = (
            "This percentage represents detected satellite-image / land-cover change "
            "relative to the post-event baseline. It is NOT equivalent to the percentage "
            "of area currently affected by landslides."
        )
        # Legacy fields — no longer used for direct alert decisions
        visitor_alert = "SIGNIFICANT SATELLITE CHANGE DETECTED"
        visitor_message = (
            "A high satellite image-change percentage has been detected relative to "
            "the post-event baseline. This is a remote-sensing observation and does "
            "not by itself confirm an active landslide. "
            "Refer to the Present Visit Advisory for a complete current-conditions assessment."
        )
    elif changed_fraction > 0.05:
        status = "moderate_satellite_change"
        label = "Moderate Satellite Change Detected"
        message = (
            f"{changed_fraction * 100:.1f}% of valid pixels show spectral/NDVI change "
            "between the post-event baseline and current imagery. "
            "Some surface change is detected. Local verification is recommended."
        )
        satellite_change_label = f"{changed_fraction * 100:.1f}% Satellite Change Detected"
        satellite_change_note = (
            "Moderate satellite image change relative to the post-event baseline. "
            "This does not confirm an active landslide condition."
        )
        visitor_alert = "MODERATE SATELLITE CHANGE DETECTED"
        visitor_message = (
            "Moderate satellite image change is detected. "
            "Refer to the Present Visit Advisory for a complete current-conditions assessment."
        )
    else:
        status = "minimal_satellite_change"
        label = "Minimal Satellite Change Detected"
        message = (
            f"{changed_fraction * 100:.1f}% of valid pixels changed between the "
            "post-event baseline and current imagery — broadly consistent with the baseline. "
            "This is not a guarantee of physical safety."
        )
        satellite_change_label = f"{changed_fraction * 100:.1f}% Satellite Change Detected"
        satellite_change_note = (
            "Low satellite image change relative to the post-event baseline. "
            "The absence of satellite change does not certify that a location is safe."
        )
        visitor_alert = "MINIMAL SATELLITE CHANGE"
        visitor_message = (
            "Current satellite comparison shows minimal change relative to the post-event "
            "baseline. This is not an official clearance. "
            "Refer to the Present Visit Advisory for a complete current-conditions assessment."
        )

    return {
        "status": status,
        "label": label,
        "message": message,
        "visitor_alert": visitor_alert,
        "visitor_message": visitor_message,
        "valid_fraction": float(valid_count / max(total_count, 1)),
        "changed_fraction": changed_fraction,
        "mask": change_mask.astype(np.uint8),
        # New explicit satellite-change fields
        "satellite_change_label": satellite_change_label,
        "satellite_change_note": satellite_change_note,
    }