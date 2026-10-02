from utils import geo_utils
from utils.present_status import compute_present_status


def test_bounds_to_geojson_polygon_uses_live_map_bounds():
    bounds = {
        "_southWest": {"lat": 11.52, "lng": 75.97},
        "_northEast": {"lat": 11.70, "lng": 76.11},
    }

    polygon = geo_utils.bounds_to_geojson_polygon(bounds)

    assert polygon["type"] == "Polygon"
    assert polygon["coordinates"][0][0] == [75.97, 11.52]
    assert polygon["coordinates"][0][1] == [76.11, 11.52]
    assert polygon["coordinates"][0][2] == [76.11, 11.70]
    assert polygon["coordinates"][0][3] == [75.97, 11.70]


def test_present_status_marks_matching_imagery_as_screening_stable():
    image = __import__("numpy").full((4, 4, 4), 0.3, dtype=float)
    result = compute_present_status(image, image.copy())

    assert result["status"] == "stable_screening"
    assert result["changed_fraction"] == 0.0
    assert result["valid_fraction"] == 1.0
    assert result["visitor_alert"] == "NO SATELLITE HAZARD SIGNAL"


def test_present_status_blocks_safety_decision_without_valid_pixels():
    import numpy as np

    image = np.zeros((3, 3, 4), dtype=float)
    result = compute_present_status(image, image.copy())

    assert result["status"] == "insufficient_evidence"
    assert result["visitor_alert"] == "NO SAFETY DECISION"
