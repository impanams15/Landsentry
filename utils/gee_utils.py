"""
Google Earth Engine (GEE) utilities for on-demand Sentinel-2 imagery retrieval.

Before first use, authenticate once from a terminal:
    earthengine authenticate

This module is only imported by the dashboard (dashboard/app.py) at runtime,
so training/evaluation do not require GEE credentials at all.
"""
import numpy as np
import ee
import os
import math


def initialize_gee(project_id=None):
    """Authenticate & initialize the Earth Engine API session."""
    if project_id is None:
        project_id = os.getenv("GEE_PROJECT_ID")
        
    if not project_id:
        raise ValueError(
            "Google Earth Engine project ID is missing. "
            "Please configure it by setting the GEE_PROJECT_ID environment variable. "
            "Example in PowerShell: $env:GEE_PROJECT_ID=\"your-actual-project-id\""
        )

    try:
        ee.Initialize(project=project_id)
    except Exception as exc:
        if "permission" in str(exc).lower() or "serviceusage.services.use" in str(exc):
            raise RuntimeError(
                f"Earth Engine project '{project_id}' rejected this account. "
                "Grant the authenticated account the Service Usage Consumer role "
                "on that Google Cloud project, then restart the dashboard."
            ) from exc
        ee.Authenticate()
        ee.Initialize(project=project_id)


def _mask_s2_clouds(image):
    """Cloud-masks a Sentinel-2 SR image using its QA60 band (standard GEE recipe)."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)


def _attach_cloud_probability(collection, aoi, start_date, end_date):
    """Attach the official Sentinel-2 cloud-probability band to each scene."""
    cloud_collection = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    joined = ee.Join.saveFirst("cloud_probability").apply(
        collection,
        cloud_collection,
        ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def add_probability(image):
        image = ee.Image(image)
        cloud_probability = ee.Image(image.get("cloud_probability"))
        return image.addBands(cloud_probability.select("probability"))

    return ee.ImageCollection(joined).map(add_probability)


def fetch_sentinel2_composite(aoi_geojson, start_date, end_date, bands=("B4", "B3", "B2", "B8")):
    """
    Builds a cloud-masked, median Sentinel-2 SR composite for an AOI and date range.

    aoi_geojson: GeoJSON geometry dict (e.g. drawn by the user on the dashboard map)
    start_date, end_date: 'YYYY-MM-DD' strings
    bands: Sentinel-2 band names, default = Red, Green, Blue, NIR

    Returns an ee.Image (still server-side; call download_as_array() to pull pixels).
    """
    aoi = ee.Geometry(aoi_geojson)
    collection = None
    source_collection = None
    scene_count = 0

    # SR is preferred. TOA covers older Sentinel-2 acquisitions where SR has
    # no catalog entries, including valid 2018 dates.
    for collection_name in (
        "COPERNICUS/S2_SR_HARMONIZED",
        "COPERNICUS/S2_HARMONIZED",
    ):
        base_collection = (
            ee.ImageCollection(collection_name)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
        )
        for cloud_limit in (30, 60, 100):
            candidate = base_collection.filter(
                ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_limit)
            )
            scene_count = candidate.size().getInfo()
            if scene_count > 0:
                collection = candidate
                source_collection = collection_name
                break
        if collection is not None:
            break

    if collection is None or scene_count == 0:
        raise RuntimeError(
            f"No Sentinel-2 scenes found for {start_date} to {end_date} "
            f"over the selected AOI. Choose dates with Sentinel-2 coverage "
            "or widen the date range."
        )

    coverage = aoi.intersection(collection.geometry(), ee.ErrorMargin(1))
    # Optimize composite generation: QA60 cloud masking is 3-5x faster than S2_CLOUD_PROBABILITY join
    composite = collection.map(_mask_s2_clouds).select(list(bands)).median().divide(10000)
    composite = composite.clip(coverage)
    return composite


def fetch_dem_composite(aoi_geojson):
    """
    Fetches terrain features (elevation, slope, aspect) using Copernicus DEM GLO-30.
    Returns an ee.Image with bands ['elevation', 'slope', 'aspect'].
    """
    aoi = ee.Geometry(aoi_geojson)
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select('DEM').map(lambda image: image.setDefaultProjection('EPSG:4326', None, 30)).mosaic()
    
    # ee.Terrain.products adds 'elevation', 'slope', 'aspect' (and others)
    terrain = ee.Terrain.products(dem).select(['elevation', 'slope', 'aspect']).clip(aoi)
    
    return terrain


def download_as_array(image, aoi_geojson, scale=10, return_geo=False, region=None):
    """
    Downloads an ee.Image over an AOI as a numpy array of shape [H, W, bands],
    using Earth Engine's getDownloadURL and requests for robust error handling.

    If return_geo=True, also returns the real affine transform and CRS that
    GDAL/rasterio read from the downloaded GeoTIFF, i.e. the actual
    georeferencing of this specific image. This is what lets downstream code
    (see utils/geo_utils.py) convert detected-mask pixel coordinates back into
    real WGS84 latitude/longitude for the exact AOI/date range that was
    requested, instead of relying on any hardcoded location.

    Returns:
        array only (H, W, bands)                         if return_geo=False
        (array, transform, crs)                           if return_geo=True
    """
    import tempfile
    import os
    import rasterio
    import requests
    import zipfile

    aoi = region if region is not None else ee.Geometry(aoi_geojson)

    # Earth Engine limits direct downloads to 50 MB. Increase the pixel size
    # only for large user-drawn AOIs so the full live AOI remains downloadable.
    coordinates = aoi_geojson.get("coordinates", [])
    points = []

    def collect_points(value):
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(
            isinstance(item, (int, float)) for item in value
        ):
            points.append(value)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect_points(child)

    collect_points(coordinates)
    if points:
        min_lon = min(point[0] for point in points)
        max_lon = max(point[0] for point in points)
        min_lat = min(point[1] for point in points)
        max_lat = max(point[1] for point in points)
        width_m = abs(max_lon - min_lon) * 111_320 * max(math.cos(math.radians((min_lat + max_lat) / 2)), 0.1)
        height_m = abs(max_lat - min_lat) * 110_540
        estimated_pixels = (width_m / scale) * (height_m / scale)
        max_pixels = 2_000_000
        if estimated_pixels > max_pixels:
            scale = math.ceil(math.sqrt((width_m * height_m) / max_pixels))
    
    try:
        url = image.getDownloadURL({
            'scale': scale,
            'region': aoi,
            'format': 'GEO_TIFF'
        })
    except Exception as e:
        raise RuntimeError(f"Earth Engine download failure: {e}")

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP failure: status {response.status_code}")

    with tempfile.TemporaryDirectory() as tmp:
        dl_path = os.path.join(tmp, "downloaded_data")
        
        with open(dl_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                f.write(chunk)
                
        if zipfile.is_zipfile(dl_path):
            with zipfile.ZipFile(dl_path, 'r') as zip_ref:
                zip_ref.extractall(tmp)
            tifs = [f for f in os.listdir(tmp) if f.endswith('.tif')]
            if not tifs:
                raise FileNotFoundError("No TIFF in extracted ZIP")
            out_path = os.path.join(tmp, tifs[0])
        else:
            out_path = dl_path + ".tif"
            os.rename(dl_path, out_path)

        with rasterio.open(out_path) as src:
            arr = src.read()  # [bands, H, W]
            transform = src.transform  # real affine geotransform of THIS download
            crs = src.crs              # real CRS of THIS download (e.g. UTM zone)

    result = np.transpose(arr, (1, 2, 0))  # -> [H, W, bands]

    if return_geo:
        return result, transform, crs
    return result
