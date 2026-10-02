"""
Georeferencing utilities for converting a detected landslide/change mask
into real-world WGS84 (EPSG:4326) coordinates.

Everything here is computed dynamically from:
  1. The actual affine transform + CRS that GDAL/rasterio read from the
     downloaded Sentinel-2/GEE GeoTIFF (see utils.gee_utils.download_as_array),
     and
  2. The pixel locations of the model's own detected change mask.

No latitude/longitude is ever hardcoded. If a different AOI is drawn, or a
different date range is used, the transform/CRS captured at download time
changes accordingly, and every coordinate produced here changes with it.
"""
from __future__ import annotations

from typing import List, Optional, TypedDict

import numpy as np
import cv2
from rasterio.warp import transform as warp_transform


class BoundingBox(TypedDict):
    north: float
    south: float
    east: float
    west: float


class ChangeRegion(TypedDict):
    pixel_count: int
    area_km2: float
    centroid_lat: float
    centroid_lon: float
    bbox: BoundingBox


def reverse_geocode_region(latitude: float, longitude: float) -> str:
    """Return the most specific readable hyper-local address (locality, street, hamlet, spot) for a WGS84 point."""
    import requests

    # 1. Try OpenStreetMap Nominatim at zoom=18 (building/street precision)
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": latitude,
                "lon": longitude,
                "format": "jsonv2",
                "zoom": 18,
                "addressdetails": 1,
            },
            headers={"User-Agent": "LandSentry-Dashboard/1.0"},
            timeout=5,
        )
        if response.status_code == 200:
            address = response.json().get("address", {})
            specific_parts = [
                address.get("amenity") or address.get("building") or address.get("landmark"),
                address.get("hamlet") or address.get("isolated_dwelling") or address.get("locality"),
                address.get("road") or address.get("pedestrian") or address.get("footway"),
                address.get("neighbourhood") or address.get("quarter") or address.get("suburb"),
                address.get("village") or address.get("town") or address.get("city"),
                address.get("county") or address.get("state_district"),
                address.get("state"),
            ]
            readable_parts = list(dict.fromkeys(part for part in specific_parts if part))
            # If OSM gives a detailed multi-part name (more than just village + state), use it
            if len(readable_parts) >= 2:
                return ", ".join(readable_parts)
    except Exception:
        pass

    # 2. BigDataCloud Fallback — extracts the most specific named administrative subdivision
    try:
        bdc_url = "https://api.bigdatacloud.net/data/reverse-geocode-client"
        resp = requests.get(bdc_url, params={"latitude": latitude, "longitude": longitude, "localityLanguage": "en"}, timeout=5)
        if resp.status_code == 200:
            bdc_data = resp.json()
            # Extract named entries from localityInfo.administrative, sorted most-specific first (highest admin level)
            admin_entries = bdc_data.get("localityInfo", {}).get("administrative", [])
            # Sort by adminLevel descending — higher level = more specific (e.g. village > taluk > state)
            admin_entries_sorted = sorted(
                [e for e in admin_entries if e.get("name") and e.get("adminLevel")],
                key=lambda x: x["adminLevel"],
                reverse=True
            )
            # Pick top-3 most specific names (e.g. "Made Revenue Village, Madikeri Taluk, Kodagu")
            best_parts = [e["name"] for e in admin_entries_sorted[:3]]
            if best_parts:
                return ", ".join(dict.fromkeys(best_parts))
    except Exception:
        pass

    return f"Spot ({latitude:.4f}° N, {longitude:.4f}° E)"


def pixel_area_km2(transform, crs=None) -> float:
    """
    Real ground-sample area of a single pixel, in km^2.

    If the raster CRS is geographic (EPSG:4326), transform.a/e are in degrees,
    so we must convert the 1-pixel footprint into a metric CRS before computing
    the area. This keeps the estimate correct even when GEE returns a raster in
    latitude/longitude instead of UTM meters.
    """
    if crs is None:
        px_w = abs(transform.a)
        px_h = abs(transform.e)
        return (px_w * px_h) / 1_000_000.0

    # Build the corners of a single pixel in the source raster CRS and project
    # them to Web Mercator (EPSG:3857), which gives true metric distances.
    corners_x = []
    corners_y = []
    for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        x, y = transform * (col, row)
        corners_x.append(x)
        corners_y.append(y)

    projected_x, projected_y = warp_transform(crs, "EPSG:3857", corners_x, corners_y)
    width_m = abs(max(projected_x) - min(projected_x))
    height_m = abs(max(projected_y) - min(projected_y))
    return (width_m * height_m) / 1_000_000.0


def bounds_to_geojson_polygon(bounds):
    """Convert folium map bounds to a live GeoJSON polygon for the active AOI."""
    south = bounds["_southWest"]["lat"]
    west = bounds["_southWest"]["lng"]
    north = bounds["_northEast"]["lat"]
    east = bounds["_northEast"]["lng"]

    return {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]],
    }


def _rowcol_to_xy(transform, row, col):
    """Pixel-center (row, col) -> (x, y) in the raster's native CRS."""
    x, y = transform * (col + 0.5, row + 0.5)
    return x, y


def compute_raster_bounds_wgs84(transform, crs, height: int, width: int) -> BoundingBox:
    """
    Bounding box (N/S/E/W, WGS84) of the FULL downloaded raster/AOI extent,
    computed from its real transform+CRS. Used to correctly geolocate the
    mask overlay on the map (never a fixed/example bounding box).
    """
    corner_rc = [(0, 0), (0, width), (height, 0), (height, width)]
    xs, ys = [], []
    for row, col in corner_rc:
        x, y = transform * (col, row)
        xs.append(x)
        ys.append(y)

    lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
    return {
        "north": max(lats),
        "south": min(lats),
        "east": max(lons),
        "west": min(lons),
    }


def extract_change_regions(
    mask: np.ndarray,
    transform,
    crs,
    min_region_pixels: int = 3,
) -> List[ChangeRegion]:
    """
    Finds each significant connected component in a detected change mask and
    converts its centroid + bounding box into real WGS84 coordinates using
    the actual georeferencing of the raster the mask was produced from.

    mask: 2D binary array (1 = detected change/landslide pixel), the SAME
          final mask the model produced and the SAME one affected-area is
          computed from.
    transform, crs: the real affine transform/CRS captured at download time
          for the image this mask corresponds to (see gee_utils.download_as_array).
    min_region_pixels: tiny specks (a handful of noisy pixels) below this
          size are not reported as a "significant" region. This is a noise
          filter on pixel COUNT, not a hardcoded location.

    Returns a list of regions, largest first. Empty list if no region meets
    min_region_pixels (caller should treat this as "nothing significant
    detected" and must NOT fabricate/substitute a location).
    """
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return []

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if num_labels <= 1:
        return []

    px_km2 = pixel_area_km2(transform, crs)

    # Batch every point (centroid + 4 bbox corners per region) through a
    # single reprojection call for efficiency and consistency.
    candidate_ids = [
        lbl for lbl in range(1, num_labels)
        if int(stats[lbl, cv2.CC_STAT_AREA]) >= min_region_pixels
    ]
    if not candidate_ids:
        return []

    xs, ys, point_meta = [], [], []
    for lbl in candidate_ids:
        cx, cy = centroids[lbl]  # cv2 gives (x=col, y=row)
        x, y = _rowcol_to_xy(transform, row=cy, col=cx)
        xs.append(x)
        ys.append(y)
        point_meta.append(("centroid", lbl))

        x0 = int(stats[lbl, cv2.CC_STAT_LEFT])
        y0 = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        for row, col in [(y0, x0), (y0, x0 + w), (y0 + h, x0), (y0 + h, x0 + w)]:
            bx, by = transform * (col, row)
            xs.append(bx)
            ys.append(by)
            point_meta.append(("bbox", lbl))

    lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)

    per_region_bbox_lons = {lbl: [] for lbl in candidate_ids}
    per_region_bbox_lats = {lbl: [] for lbl in candidate_ids}
    centroid_lonlat = {}
    for (kind, lbl), lon, lat in zip(point_meta, lons, lats):
        if kind == "centroid":
            centroid_lonlat[lbl] = (lon, lat)
        else:
            per_region_bbox_lons[lbl].append(lon)
            per_region_bbox_lats[lbl].append(lat)

    regions: List[ChangeRegion] = []
    for lbl in candidate_ids:
        pixel_count = int(stats[lbl, cv2.CC_STAT_AREA])
        lon, lat = centroid_lonlat[lbl]
        regions.append({
            "pixel_count": pixel_count,
            "area_km2": pixel_count * px_km2,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "bbox": {
                "north": max(per_region_bbox_lats[lbl]),
                "south": min(per_region_bbox_lats[lbl]),
                "east": max(per_region_bbox_lons[lbl]),
                "west": min(per_region_bbox_lons[lbl]),
            },
        })

    regions.sort(key=lambda r: r["pixel_count"], reverse=True)
    return regions


def largest_region(regions: List[ChangeRegion]) -> Optional[ChangeRegion]:
    """Convenience accessor: the single largest detected region, or None."""
    return regions[0] if regions else None
