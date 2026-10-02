def render_page():
    """
    LandSentry Interactive Dashboard

    Draw an Area of Interest (AOI), pick Before/After dates,
    fetch Sentinel-2 imagery using Google Earth Engine,
    run a trained LandSentry model, and display detected
    landslide changes.
    """

    import sys
    import os
    import datetime
    from affine import Affine

    # Add project root to Python path
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    import streamlit as st
    import numpy as np
    import torch
    import folium
    import ee

    from streamlit_folium import st_folium
    from folium.plugins import Draw

    from models.siamese_cnn import SiameseChangeDetector
    from models.transformer_cd import TransformerChangeDetector
    from utils.preprocessing import normalize_percentile
    from utils import gee_utils
    from utils import geo_utils
    from utils.present_status import compute_present_status
    from utils.visit_advisory import compute_visit_advisory, AdvisoryStatus

    # ============================================================
    # CONFIGURATION
    # ============================================================

    LOOKBACK_DAYS = 30

    # Your Google Earth Engine / Google Cloud Project ID
    GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "landsentry-508714")

    TODAY = datetime.date.today()
    DEFAULT_BEFORE_DATE = datetime.date(2024, 7, 1)
    DEFAULT_AFTER_DATE = datetime.date(2024, 8, 1)
    DEFAULT_PRESENT_DATE = TODAY

    # ============================================================
    # STREAMLIT PAGE CONFIGURATION
    # ============================================================


    st.title("LandSentry — Bitemporal Landslide Change Detection")
    st.caption("Western Ghats | Siamese CNN & Transformer-based bitemporal change detection")

    # ============================================================
    # MODEL LOADING
    # ============================================================

    @st.cache_resource
    def load_model(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model_cls = SiameseChangeDetector if ckpt["model_name"] == "siamese" else TransformerChangeDetector
        
        in_channels = ckpt.get("in_channels", 4)
        model = model_cls(in_channels=in_channels)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, in_channels


    def prepare_display_pair(pre_arr, post_arr):
        """Enhance real RGB reflectance for display and return the shared valid mask."""
        pre_rgb = pre_arr[..., :3].astype(np.float32)
        post_rgb = post_arr[..., :3].astype(np.float32)
        pre_valid = np.any(pre_rgb > 1e-4, axis=-1)
        post_valid = np.any(post_rgb > 1e-4, axis=-1)
        valid = pre_valid & post_valid
        if not np.any(valid):
            raise RuntimeError("GEE returned no valid RGB pixels for the comparison.")

        pre_display = np.empty_like(pre_rgb, dtype=np.float32)
        post_display = np.empty_like(post_rgb, dtype=np.float32)

        for image, image_valid, output in (
            (pre_rgb, pre_valid, pre_display),
            (post_rgb, post_valid, post_display),
        ):
            for band in range(3):
                values = image[..., band][image_valid]
                lower, upper = np.percentile(values, [1, 99])
                span = max(float(upper - lower), 1e-6)
                output[..., band] = np.clip((image[..., band] - lower) / span, 0, 1)
            output[:] = np.power(output, 0.85)

        # Neutral gray marks only true missing pixels; it is never interpreted as
        # a land-cover change and prevents nodata from appearing as black terrain.
        pre_display[~pre_valid] = 0.55
        post_display[~post_valid] = 0.55
        return pre_display, post_display, valid

    # ============================================================
    # SIDEBAR SETTINGS
    # ============================================================

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    with st.sidebar:
        st.header("Settings")
        with st.form("detection_form", clear_on_submit=False):
            ckpt_options = {
                "Siamese CNN (Kodagu)": "checkpoints/siamese_kodagu_best.pt",
                "Transformer (Kodagu)": "checkpoints/transformer_kodagu_best.pt",
                "Siamese CNN (Wayanad)": "checkpoints/siamese_wayanad_best.pt",
                "Transformer (Wayanad)": "checkpoints/transformer_wayanad_best.pt",
            }
            selected_label = st.selectbox("Model architecture & training domain", list(ckpt_options.keys()))
            checkpoint_path = ckpt_options[selected_label]
            before_date = st.date_input("Before date", DEFAULT_BEFORE_DATE)
            after_date = st.date_input("After date", DEFAULT_AFTER_DATE)
            present_date = st.date_input("Present date", DEFAULT_PRESENT_DATE)
            threshold = st.slider("Detection threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
            submitted = st.form_submit_button("Fetch imagery & detect changes")

    # ============================================================
    # LOCATION SEARCH & MAP
    # ============================================================
    import requests


    @st.cache_data(ttl=86400, show_spinner=False)
    def lookup_region_name(latitude, longitude):
        """Cache readable place names while retaining exact pixel-derived coordinates."""
        return geo_utils.reverse_geocode_region(latitude, longitude)


    def tag_regions(regions):
        tagged_regions = []
        for region in regions:
            tagged_region = dict(region)
            tagged_region["place_name"] = lookup_region_name(
                region["centroid_lat"], region["centroid_lon"]
            )
            tagged_regions.append(tagged_region)
        return tagged_regions

    st.subheader("1. Locate & Draw Area of Interest (AOI)")

    col_search, col_btn = st.columns([3, 1])
    with col_search:
        search_query = st.text_input("Search for a place to zoom in:", placeholder="e.g., Wayanad, Kerala")
    with col_btn:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Search Location"):
            if search_query:
                try:
                    url = f"https://nominatim.openstreetmap.org/search?q={search_query}&format=json&limit=1"
                    resp = requests.get(url, headers={'User-Agent': 'LandSentry-Dashboard'})
                    data = resp.json()
                    if len(data) > 0:
                        st.session_state.map_center = [float(data[0]['lat']), float(data[0]['lon'])]
                        st.session_state.map_zoom = 12
                    else:
                        st.error("Location not found. Please try a different name.")
                except Exception as e:
                    st.warning("Cannot reach the OpenStreetMap search service (Network/DNS block). Please pan/zoom on the map directly to locate your region.", icon="⚠️")

    if "map_center" in st.session_state:
        base_map = folium.Map(
            location=st.session_state.map_center,
            zoom_start=st.session_state.get("map_zoom", 12),
        )
    else:
        base_map = folium.Map(zoom_start=2)

    Draw(
        export=False,
        draw_options={
            "rectangle": True,
            "polygon": True,
            "circle": False,
            "marker": False,
            "polyline": False
        }
    ).add_to(base_map)

    map_state = st_folium(base_map, height=500, width=900)

    aoi_geojson = None
    if map_state and map_state.get("last_active_drawing"):
        aoi_geojson = map_state["last_active_drawing"]["geometry"]

    # ============================================================
    # RUN LANDSLIDE DETECTION
    # ============================================================

    if submitted:
        # Do not leave an older comparison visible while a new request is being validated.
        st.session_state.last_result = None
        if before_date >= after_date:
            st.session_state.last_result = {
                "error": "Before date must be earlier than After date for a valid bitemporal comparison."
            }
        elif after_date >= present_date:
            st.session_state.last_result = {
                "error": "Present date must be later than After date so current condition is compared with the post-event baseline."
            }
        elif aoi_geojson is None:
            st.session_state.last_result = {"error": "Please draw an Area of Interest (AOI) on the map first."}
        elif not os.path.exists(checkpoint_path):
            st.session_state.last_result = {"error": f"Checkpoint not found: {checkpoint_path}"}
        else:
            window = datetime.timedelta(days=LOOKBACK_DAYS)

            try:
                with st.spinner("Initializing Google Earth Engine..."):
                    gee_utils.initialize_gee(project_id=GEE_PROJECT_ID)

                with st.spinner("Fetching Sentinel-2 imagery..."):
                    pre_ee = gee_utils.fetch_sentinel2_composite(
                        aoi_geojson,
                        str(before_date - window),
                        str(before_date)
                    )
                    post_ee = gee_utils.fetch_sentinel2_composite(
                        aoi_geojson,
                        str(after_date),
                        str(after_date + window)
                    )
                    current_ee = gee_utils.fetch_sentinel2_composite(
                        aoi_geojson,
                        str(present_date - window),
                        str(present_date + datetime.timedelta(days=1))
                    )

                    common_region = pre_ee.geometry().intersection(
                        post_ee.geometry(), ee.ErrorMargin(1)
                    ).intersection(current_ee.geometry(), ee.ErrorMargin(1))

                    pre_arr, pre_transform, pre_crs = gee_utils.download_as_array(
                        pre_ee, aoi_geojson, return_geo=True, region=common_region
                    )
                    post_arr, post_transform, post_crs = gee_utils.download_as_array(
                        post_ee, aoi_geojson, return_geo=True, region=common_region
                    )
                    current_arr, current_transform, current_crs = gee_utils.download_as_array(
                        current_ee, aoi_geojson, return_geo=True, region=common_region
                    )

                    pre_valid_fraction = float(np.mean(np.any(pre_arr[..., :3] > 1e-4, axis=-1)))
                    post_valid_fraction = float(np.mean(np.any(post_arr[..., :3] > 1e-4, axis=-1)))
                    current_valid_fraction = float(np.mean(np.any(current_arr[..., :3] > 1e-4, axis=-1)))
                    if min(pre_valid_fraction, post_valid_fraction, current_valid_fraction) < 0.75:
                        raise RuntimeError(
                            "The selected date windows do not provide enough Sentinel-2 coverage for this AOI. "
                            f"Try a smaller AOI or nearby dates (pre valid: {pre_valid_fraction:.0%}, "
                            f"post valid: {post_valid_fraction:.0%}, present valid: {current_valid_fraction:.0%})."
                        )

                st.success(
                    f"Successfully fetched imagery! Pre-event: {pre_arr.shape} | "
                    f"Post-event: {post_arr.shape} | Present: {current_arr.shape}"
                )

                with st.spinner("Running LandSentry AI model..."):
                    model, in_channels = load_model(checkpoint_path)

                    if in_channels == 5:
                        import json
                        from utils.preprocessing import compute_ndvi
                        
                        from pathlib import Path
                        
                        project_root = Path(__file__).resolve().parents[2]
                        stats_path = project_root / "data" / "ml" / "kodagu" / "normalization_stats.json"
                        
                        if not stats_path.exists():
                            raise FileNotFoundError(
                                f"LandSentry normalization statistics not found at: "
                                f"{stats_path}"
                            )
                            
                        print(f"DEBUG: Resolved normalization stats path: {stats_path}")
                        
                        with open(stats_path, "r") as f:
                            stats = json.load(f)
                        
                        mean = np.array(stats["mean"], dtype=np.float32)
                        std = np.array(stats["std"], dtype=np.float32)
                        
                        if len(mean) == 8:
                            mean = np.insert(mean, 4, 0.0)
                            mean = np.append(mean, 0.0)
                            std = np.insert(std, 4, 1.0)
                            std = np.append(std, 1.0)
                            
                        pre_mean = mean[:5]
                        pre_std = std[:5]
                        post_mean = mean[5:]
                        post_std = std[5:]
                        
                        pre_ndvi = compute_ndvi(pre_arr.astype(np.float32), red_idx=0, nir_idx=3)
                        post_ndvi = compute_ndvi(post_arr.astype(np.float32), red_idx=0, nir_idx=3)
                        
                        pre_5 = np.concatenate([pre_arr[..., :4].astype(np.float32), pre_ndvi], axis=-1)
                        post_5 = np.concatenate([post_arr[..., :4].astype(np.float32), post_ndvi], axis=-1)
                        
                        pre_n = (pre_5 - pre_mean) / pre_std
                        post_n = (post_5 - post_mean) / post_std

                        st.info(
                            f"**Checkpoint:** `{checkpoint_path}`  \n"
                            f"**Detected in_channels:** 5  \n"
                            f"**Preprocessing mode:** 5-channel RGBN + NDVI  \n"
                            f"**Final tensor shape:** [1, 5, H, W]"
                        )

                    elif in_channels == 4:
                        import json
                        from pathlib import Path

                        project_root = Path(__file__).resolve().parents[2]
                        stats_path = project_root / "data" / "ml" / "kodagu" / "normalization_stats.json"

                        if not stats_path.exists():
                            raise FileNotFoundError(
                                f"LandSentry normalization statistics not found at: {stats_path}"
                            )

                        print(f"DEBUG: stats_path={stats_path}")

                        with open(stats_path, "r") as f_stats:
                            stats4 = json.load(f_stats)

                        mean4 = np.array(stats4["mean"], dtype=np.float32)
                        std4  = np.array(stats4["std"],  dtype=np.float32)

                        # normalization_stats.json: 8 values = [pre_B4,pre_B3,pre_B2,pre_B8, post_B4,post_B3,post_B2,post_B8]
                        pre_mean  = mean4[:4]
                        pre_std   = std4[:4]
                        post_mean = mean4[4:8]
                        post_std  = std4[4:8]

                        pre_n  = (pre_arr[...,  :4].astype(np.float32) - pre_mean)  / pre_std
                        post_n = (post_arr[..., :4].astype(np.float32) - post_mean) / post_std

                        st.info(
                            f"**Checkpoint:** `{checkpoint_path}`  \n"
                            f"**Detected in_channels:** 4  \n"
                            f"**Preprocessing mode:** 4-channel RGBN (no NDVI)  \n"
                            f"**Final tensor shape:** [1, 4, H, W]"
                        )

                    else:
                        raise RuntimeError(
                            f"Unexpected in_channels={in_channels} in checkpoint '{checkpoint_path}'. "
                            f"LandSentry supports 4 (RGBN) or 5 (RGBN+NDVI) only."
                        )

                    pre_t = torch.from_numpy(pre_n).permute(2, 0, 1).unsqueeze(0).float()
                    post_t = torch.from_numpy(post_n).permute(2, 0, 1).unsqueeze(0).float()

                    _, _, h, w = pre_t.shape
                    pad_h = (16 - (h % 16)) % 16
                    pad_w = (16 - (w % 16)) % 16

                    import torch.nn.functional as F
                    pre_t_padded = F.pad(pre_t, (0, pad_w, 0, pad_h), mode='reflect')
                    post_t_padded = F.pad(post_t, (0, pad_w, 0, pad_h), mode='reflect')

                    with torch.no_grad():
                        logits = model(pre_t_padded, post_t_padded)
                        if isinstance(logits, tuple):
                            logits = logits[0]

                    logits = logits[..., :h, :w]
                    prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()
                    active_threshold = 0.5 if in_channels == 5 else threshold
                    change_mask = (prob_map > active_threshold).astype(np.uint8)
                    detection_mode = "ai"

                    # A detector that marks most of the scene is not a useful
                    # landslide localization. Keep only the strongest live model
                    # responses when the requested threshold produces a scene-wide mask.
                    if change_mask.mean() > 0.35:
                        strong_threshold = max(threshold, float(np.quantile(prob_map, 0.90)))
                        change_mask = (prob_map >= strong_threshold).astype(np.uint8)

                    pre_n_2d = pre_n[..., :4]
                    post_n_2d = post_n[..., :4]
                    red_pre = pre_n_2d[..., 0]
                    nir_pre = pre_n_2d[..., 3]
                    red_post = post_n_2d[..., 0]
                    nir_post = post_n_2d[..., 3]

                    pre_ndvi = np.divide(
                        nir_pre - red_pre,
                        nir_pre + red_pre + 1e-6,
                        out=np.zeros_like(red_pre, dtype=np.float32),
                        where=(nir_pre + red_pre) != 0,
                    )
                    post_ndvi = np.divide(
                        nir_post - red_post,
                        nir_post + red_post + 1e-6,
                        out=np.zeros_like(red_post, dtype=np.float32),
                        where=(nir_post + red_post) != 0,
                    )

                    ndvi_change = np.abs(pre_ndvi - post_ndvi)
                    spectral_change = np.mean(np.abs(pre_n_2d - post_n_2d), axis=-1)
                    fallback_mask = ((ndvi_change > 0.04) | (spectral_change > 0.08)).astype(np.uint8)

                    if change_mask.sum() == 0 or change_mask.sum() < 200:
                        if fallback_mask.sum() > 0:
                            change_mask = fallback_mask
                            detection_mode = "spectral_fallback"

                    if change_mask.mean() > 0.35:
                        spectral_rank = np.quantile(spectral_change, 0.95)
                        focused_fallback = (spectral_change >= spectral_rank).astype(np.uint8)
                        if focused_fallback.sum() > 0:
                            change_mask = focused_fallback
                            detection_mode = "spectral_fallback"

                    if change_mask.sum() == 0:
                        spectral_rank = np.quantile(spectral_change, 0.98)
                        fallback_mask = (spectral_change > spectral_rank * 0.8).astype(np.uint8)
                        if fallback_mask.sum() > 0:
                            change_mask = fallback_mask
                            detection_mode = "spectral_fallback"

                    if change_mask.sum() > 0:
                        if detection_mode == "ai":
                            confidence = float(prob_map[change_mask == 1].mean())
                        else:
                            confidence = float(np.abs(pre_n_2d - post_n_2d)[change_mask == 1].mean())
                    else:
                        confidence = 0.0

                    px_area_km2 = geo_utils.pixel_area_km2(post_transform, post_crs)
                    affected_area_km2 = float(change_mask.sum() * px_area_km2)
                    change_regions = tag_regions(
                        geo_utils.extract_change_regions(change_mask, post_transform, post_crs)
                    )
                    raster_bounds = geo_utils.compute_raster_bounds_wgs84(post_transform, post_crs, *change_mask.shape[:2])

                    present_status = compute_present_status(post_arr, current_arr)
                    present_mask = present_status["mask"]
                    present_regions = tag_regions(
                        geo_utils.extract_change_regions(present_mask, current_transform, current_crs)
                    )
                    present_area_km2 = float(present_mask.sum() * geo_utils.pixel_area_km2(current_transform, current_crs))
                    present_bounds = geo_utils.compute_raster_bounds_wgs84(
                        current_transform, current_crs, *present_mask.shape[:2]
                    )

                    # Compute AOI centroid for advisory API calls
                    _bounds = geo_utils.compute_raster_bounds_wgs84(post_transform, post_crs, *change_mask.shape[:2])
                    _advisory_lat = (_bounds["north"] + _bounds["south"]) / 2.0
                    _advisory_lon = (_bounds["east"] + _bounds["west"]) / 2.0

                    with st.spinner("Fetching live official warnings and rainfall data for Present Visit Advisory..."):
                        visit_advisory = compute_visit_advisory(
                            lat=_advisory_lat,
                            lon=_advisory_lon,
                            present_date=present_date,
                            post_arr=post_arr,
                            current_arr=current_arr,
                            present_status=present_status,
                            historical_change_fraction=float(change_mask.mean()),
                            after_date=after_date,
                            before_date=before_date,
                            window=window,
                        )

                    rgb_pre, rgb_post, display_valid = prepare_display_pair(pre_arr, post_arr)
                    _, rgb_current, current_display_valid = prepare_display_pair(post_arr, current_arr)
                st.session_state.last_result = {
                    "confidence": confidence,
                    "affected_area_km2": affected_area_km2,
                    "changed_pixels": int(change_mask.sum()),
                    "change_regions": change_regions,
                    "raster_bounds": raster_bounds,
                    "aoi_geojson": aoi_geojson,
                    "rgb_pre": rgb_pre,
                    "rgb_post": rgb_post,
                    "rgb_current": rgb_current,
                    "pre_arr": pre_arr,
                    "post_arr": post_arr,
                    "current_arr": current_arr,
                    "post_transform": post_transform,
                    "post_crs": post_crs,
                    "display_valid": display_valid,
                    "current_display_valid": current_display_valid,
                    "change_mask": change_mask,
                    "present_mask": present_mask,
                    "present_status": present_status,
                    "present_regions": present_regions,
                    "present_area_km2": present_area_km2,
                    "present_bounds": present_bounds,
                    "before_date": before_date,
                    "after_date": after_date,
                    "present_date": present_date,
                    "window": window,
                    "detection_mode": detection_mode,
                    "visit_advisory": visit_advisory,
                    "error": None,
                }

            except Exception as e:
                st.session_state.last_result = {"error": f"Error during processing: {e}"}

    if st.session_state.last_result is not None:
        result = st.session_state.last_result
        if result.get("error"):
            st.error(result["error"])
        else:
            from utils import feature_utils
            import plotly.graph_objects as go

            st.subheader("2. New-AOI Model Prediction")
            st.info("Ground truth not available — evaluation metrics are not calculated.")
            col1, col2, col3 = st.columns(3)
            col1.metric("Model Confidence", f"{result['confidence'] * 100:.1f}%")
            col2.metric("Estimated Affected Area", f"{result['affected_area_km2']:.4f} km²")
            col3.metric("Changed Pixels", f"{int(result['changed_pixels'])}")

            # --- FEATURE 1: AUTOMATED INCIDENT REPORT EXPORTS ---
            st.markdown("### 📥 Official Incident Export Options")
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                pdf_bytes = feature_utils.generate_pdf_report(result)
                st.download_button(
                    label="🚨 Export Official PDF Incident Report",
                    data=pdf_bytes,
                    file_name=f"LandSentry_Report_{result['after_date']}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with exp_col2:
                geojson_str = feature_utils.generate_geojson_report(
                    result, result["post_transform"], result["post_crs"]
                )
                st.download_button(
                    label="🗺️ Download GeoJSON Polygons (QGIS / ArcGIS)",
                    data=geojson_str,
                    file_name=f"LandSentry_Landslides_{result['after_date']}.geojson",
                    mime="application/geo+json",
                    use_container_width=True
                )

            st.markdown("**Detected Landslide Location**")
            if not result["change_regions"]:
                st.info("No significant affected region detected — geographic coordinates unavailable.")
            else:
                top_region = result["change_regions"][0]
                st.markdown(
                    f"Place/region: **{top_region['place_name']}**  \n"
                    f"Latitude: `{top_region['centroid_lat']:.6f}° N`  \n"
                    f"Longitude: `{top_region['centroid_lon']:.6f}° E`  \n"
                    f"Region affected area: `{top_region['area_km2']:.4f} km²`"
                )
                bbox = top_region["bbox"]
                st.caption(
                    f"Bounding box — North: {bbox['north']:.6f}°, South: {bbox['south']:.6f}°, "
                    f"East: {bbox['east']:.6f}°, West: {bbox['west']:.6f}°"
                )

                if len(result["change_regions"]) > 1:
                    with st.expander(f"{len(result['change_regions'])} distinct affected regions detected"):
                        for i, region in enumerate(result["change_regions"], start=1):
                            st.markdown(
                                f"**Landslide Region {i}**  \n"
                                f"- Place/region: **{region['place_name']}**\n"
                                f"- Latitude: `{region['centroid_lat']:.6f}° N`\n"
                                f"- Longitude: `{region['centroid_lon']:.6f}° E`\n"
                                f"- Area: `{region['area_km2']:.4f} km²`"
                            )

            if result.get("detection_mode") == "spectral_fallback":
                st.markdown("**Detection Source:** Real Sentinel-2 imagery via Google Earth Engine with a live spectral-change fallback")
            else:
                st.markdown("**Detection Source:** Real Sentinel-2 imagery via Google Earth Engine")
            st.markdown(
                f"Pre-event acquisition window: `{result['before_date'] - result['window']}` to `{result['before_date']}`  \n"
                f"Post-event acquisition window: `{result['after_date']}` to `{result['after_date'] + result['window']}`  \n"
                f"Present acquisition window: `{result['present_date'] - result['window']}` to `{result['present_date']}`"
            )

            # --- FEATURE 2: REAL-TIME RAINFALL & WEATHER INTEGRATION ---
            st.subheader("🌧️ Real-Time Rainfall & Weather Integration (Open-Meteo)")
            center_lat = (result["raster_bounds"]["north"] + result["raster_bounds"]["south"]) / 2.0
            center_lon = (result["raster_bounds"]["east"] + result["raster_bounds"]["west"]) / 2.0

            st.caption(f"📍 **Target AOI Coordinates:** `{center_lat:.4f}° N, {center_lon:.4f}° E` (Open-Meteo ER5 / ERA5-Land Reanalysis Grid)")

            rain_scope = st.radio(
                "Select Rainfall Analysis Window:",
                options=["Pre-to-Post Event Window (Before Date ➔ After Date)", "Full Timeline (Before Date ➔ Present Date)"],
                index=0,
                horizontal=True
            )

            # Always read directly from active sidebar input values (before_date, after_date, present_date)
            # falling back to cached result dates if needed
            active_before = before_date if 'before_date' in locals() else result["before_date"]
            active_after = after_date if 'after_date' in locals() else result["after_date"]
            active_present = present_date if 'present_date' in locals() else result["present_date"]

            rain_start = active_before
            rain_end = active_after if "Pre-to-Post" in rain_scope else active_present

            weather_data = feature_utils.fetch_historical_rainfall(
                lat=center_lat,
                lon=center_lon,
                start_date=rain_start,
                end_date=rain_end
            )

            if weather_data["error"]:
                st.warning(weather_data["error"])
            else:
                w_col1, w_col2 = st.columns([1, 2])
                with w_col1:
                    st.metric("Total Cumulative Rainfall", f"{weather_data['total_precip_mm']:.1f} mm")
                    st.metric("Peak 24h Rainfall", f"{weather_data['max_daily_precip_mm']:.1f} mm")
                    if weather_data['max_daily_precip_mm'] > 100:
                        st.error("⛈️ Extreme Precipitation Trigger Detected (>100mm/day)")
                    elif weather_data['max_daily_precip_mm'] > 50:
                        st.warning("🌧️ Heavy Monsoon Precipitation Trigger Detected (>50mm/day)")
                    else:
                        st.info("🌦️ Moderate / Standard Rainfall Levels Observed")

                with w_col2:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=weather_data["dates"],
                        y=weather_data["precipitation_mm"],
                        marker_color="#0284C7",
                        name="Precipitation (mm)"
                    ))
                    fig.update_layout(
                        title=f"Daily Rainfall ({rain_start} ➔ {rain_end})",
                        xaxis_title="Date",
                        yaxis_title="Precipitation (mm)",
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=250,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # --- FEATURE 3: MULTI-TEMPORAL SLIDE PROGRESSION & RECOVERY TRACKING ---
            st.subheader("📈 Multi-Temporal Slide Progression & Recovery Tracking")
            prog_metrics = feature_utils.compute_progression_metrics(
                pre_arr=result["pre_arr"],
                post_arr=result["post_arr"],
                current_arr=result["current_arr"],
                change_mask=result["change_mask"]
            )

            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric("Pre-Event Mean NDVI", f"{prog_metrics['ndvi_pre']:.3f}")
            p_col2.metric("Post-Event Mean NDVI", f"{prog_metrics['ndvi_post']:.3f}", delta=f"{prog_metrics['ndvi_post'] - prog_metrics['ndvi_pre']:.3f}")
            p_col3.metric("Present Mean NDVI", f"{prog_metrics['ndvi_curr']:.3f}", delta=f"{prog_metrics['ndvi_curr'] - prog_metrics['ndvi_post']:.3f}")

            st.markdown(f"**Current Scar Dynamics:** <span style='color:{prog_metrics['status_color']}; font-weight:bold;'>{prog_metrics['status_text']}</span>", unsafe_allow_html=True)

            slider_val = st.select_slider(
                "Compare Slope Condition Over Time:",
                options=["Pre-event Baseline", "Post-event Scar", "Present Condition Screening"],
                value="Post-event Scar"
            )

            if slider_val == "Pre-event Baseline":
                st.image(result["rgb_pre"], caption=f"Pre-event Baseline imagery (NDVI: {prog_metrics['ndvi_pre']:.3f})", use_container_width=True)
            elif slider_val == "Post-event Scar":
                post_overlay = result["rgb_post"].copy()
                post_overlay[result["change_mask"] > 0] = 0.55 * post_overlay[result["change_mask"] > 0] + 0.45 * np.array([1.0, 0.0, 0.0], dtype=np.float32)
                st.image(post_overlay, caption=f"Post-event Landslide Scar (NDVI: {prog_metrics['ndvi_post']:.3f})", use_container_width=True)
            else:
                curr_overlay = result["rgb_current"].copy()
                curr_overlay[result["present_mask"] > 0] = 0.55 * curr_overlay[result["present_mask"] > 0] + 0.45 * np.array([1.0, 0.65, 0.0], dtype=np.float32)
                st.image(curr_overlay, caption=f"Present Condition & Screening (NDVI: {prog_metrics['ndvi_curr']:.3f})", use_container_width=True)

            # ============================================================
            # SECTION 3 — PRESENT VISIT ADVISORY
            # (completely separate from historical change detection)
            # ============================================================
            st.subheader("3. Present Visit Advisory")
            st.caption(
                "This advisory evaluates current official warnings, live weather/rainfall, "
                "and recent satellite change — completely separate from historical change percentage."
            )

            advisory = result.get("visit_advisory")
            present_status = result["present_status"]

            if advisory is None:
                st.warning(
                    "⚪ DATA VERIFICATION REQUIRED — "
                    "Present Visit Advisory could not be generated. "
                    "Check local authority advisories directly."
                )
            else:
                adv_status = advisory.status
                adv_evidence = advisory.evidence
                adv_rain = adv_evidence.rainfall
                adv_official = adv_evidence.official_warning
                adv_sat = adv_evidence.satellite_anomaly
                adv_susc = adv_evidence.susceptibility
                adv_freshness = adv_evidence.freshness
                
                dyn_place_name = result["present_regions"][0]["place_name"] if result.get("present_regions") else "Selected AOI"
                
                # ---- Validation Check ----
                st.caption(f"📍 All current advisory data and rules displayed strictly correspond to the currently selected area: **{dyn_place_name}**")

                # ---- 1. Status Banner ----
                # Map advisory status to official IMD alert style naming
                banner_text = f"{advisory.status_emoji} {advisory.headline}"
                if adv_status == AdvisoryStatus.RED:
                    st.error(f"[IMD ALERT CODE: RED / EVACUATE OR AVOID] {banner_text}")
                elif adv_status == AdvisoryStatus.ORANGE:
                    st.warning(f"[IMD ALERT CODE: ORANGE / BE PREPARED] {banner_text}")
                elif adv_status == AdvisoryStatus.GREEN:
                    st.success(f"[IMD ALERT CODE: GREEN / NO ALERT] {banner_text}")
                elif adv_status == AdvisoryStatus.INFORMATIONAL:
                    st.info(f"[IMD ALERT CODE: YELLOW / WATCH / INFORMATIONAL] {banner_text}")
                else:
                    st.warning(f"[STATUS: UNVERIFIED / DATA REQUIRED] {banner_text}")

                # ---- 2. Data Freshness Indicator Bar ----
                st.markdown("##### 🕒 Data Freshness & Source Verification Status")
                f_col1, f_col2, f_col3 = st.columns(3)
                with f_col1:
                    st.markdown(f"**Official Warning Status:**  \n`{adv_freshness.official_warning_status}`")
                    st.caption(f"Sources: `{adv_freshness.official_warning_sources_summary}`")
                with f_col2:
                    st.markdown(f"**Weather / Rainfall Data:**  \n`{adv_freshness.weather_timestamp}`")
                    st.caption(f"Provider: `{adv_rain.weather_source}`")
                with f_col3:
                    st.markdown(f"**Satellite Acquisition:**  \n`{adv_freshness.satellite_acquisition_dates}`")
                    st.caption(f"Screening checked at: `{advisory.generated_at_ist}`")

                st.markdown("---")

                # ---- 3. Current Conditions Metrics ----
                st.markdown("**CURRENT HAZARD EVIDENCE BREAKDOWN**")
                cond_cols = st.columns(5)

                # Official warning status display
                if adv_official.status == "active":
                    cond_cols[0].metric("Official Warning", "⚠️ ACTIVE ADVISORY", help=f"Source: {adv_official.primary_source}")
                elif adv_official.status == "clear":
                    cond_cols[0].metric("Official Warning", "✅ VERIFIED CLEAR", help=f"Source: {adv_official.primary_source}")
                else:
                    cond_cols[0].metric("Official Warning", "⚪ UNAVAILABLE", help="Official warning feeds could not be reached live")

                # Weather/Rainfall Metrics
                if adv_rain.available:
                    cond_cols[1].metric("Weather (24h est.)", f"{adv_rain.precip_24h_mm:.1f} mm")
                    cond_cols[2].metric("Weather (72h est.)", f"{adv_rain.precip_72h_mm:.1f} mm")
                else:
                    cond_cols[1].metric("Weather (24h est.)", "⚪ N/A")
                    cond_cols[2].metric("Weather (72h est.)", "⚪ N/A")

                # Correctly Renamed Satellite Metric
                cond_cols[3].metric(
                    "Post-event → Current Satellite Image Change",
                    f"{present_status.get('changed_fraction', 0.0) * 100:.1f}%",
                    help="Detected image/land-cover change between the selected periods. It does NOT mean this area is currently affected by landslides."
                )

                # Susceptibility
                cond_cols[4].metric(
                    "Susceptibility Proxy",
                    adv_susc.label if adv_susc.available else "Unknown",
                    help="Spectral proxy only — not a formal landslide susceptibility assessment."
                )

                # ---- 4. Official Warning Detail Box ----
                st.markdown("---")
                if adv_official.available and adv_official.status == "clear":
                    warn_title = "✅ **Official Warning: NO WARNING DETECTED**"
                elif adv_official.available and adv_official.status == "active":
                    warn_title = f"⚠️ **Official Warning: {adv_official.warning_type or 'ACTIVE WARNING'}**"
                else:
                    warn_title = "⚪ **Official Warning: UNAVAILABLE**"
                
                st.markdown(f"### {warn_title}")
                st.markdown(f"**Source:**  \n{adv_official.primary_source}")
                
                if adv_official.timestamp:
                    st.markdown(f"**Issued:**  \n{adv_official.timestamp}")
                
                if adv_official.applicable_date:
                    st.markdown(f"**Applicable:**  \n{adv_official.applicable_date}")
                    
                if adv_official.future_advisory:
                    st.markdown(f"**Future advisory:**  \n{adv_official.future_advisory}")
                st.markdown("<br>", unsafe_allow_html=True)

                # ---- 5. Weather Evidence Detail Box ----
                if adv_rain.available:
                    st.info(
                        f"🌧️ **Current Weather & Rainfall Estimate ({dyn_place_name} / AOI Grid):**  \n"
                        f"• 24h rainfall estimate: `{adv_rain.precip_24h_mm:.1f} mm` | "
                        f"• 72h rainfall estimate: `{adv_rain.precip_72h_mm:.1f} mm` | "
                        f"• 7-day rainfall estimate: `{adv_rain.total_7day_mm:.1f} mm`  \n"
                        f"• Maximum daily estimate: `{adv_rain.max_daily_recent_mm:.1f} mm` | "
                        f"• Next 24h forecast: `{adv_rain.forecast_24h_mm:.1f} mm`"
                    )

                # Explicit Satellite Disclaimer Requirement
                st.caption(
                    "⚠️ **Satellite Change Disclaimer:** "
                    "The change percentage represents satellite-image/land-cover change between the selected historical post-event baseline "
                    "and the current imagery. It is NOT the percentage of area currently affected by landslides."
                )

                # ---- 6. Explanation ----
                st.markdown("**WHY THIS STATUS?**")
                st.markdown(advisory.explanation)

                # ---- 7. Historical Satellite Context ----
                st.markdown("---")
                st.markdown("**HISTORICAL SATELLITE CHANGE CONTEXT**")
                st.caption(
                    "The values below are from the historical bitemporal comparison. "
                    "They describe image/land-cover change between the selected pre-event "
                    "and post-event periods. They are NOT current visit advisory values."
                )
                hist_cols = st.columns(3)
                hist_cols[0].metric(
                    "Present valid coverage",
                    f"{present_status['valid_fraction'] * 100:.1f}%"
                )
                hist_cols[1].metric(
                    "Post-event → Current Satellite Image Change",
                    f"{present_status.get('changed_fraction', 0.0) * 100:.1f}%"
                )
                hist_cols[2].metric(
                    "Changed Area (Post→Current)",
                    f"{result['present_area_km2']:.4f} km²"
                )

                if result["present_regions"]:
                    region = result["present_regions"][0]
                    st.markdown(
                        f"Satellite screening hotspot: **{region['place_name']}** | "
                        f"Latitude `{region['centroid_lat']:.6f}° N`, "
                        f"Longitude `{region['centroid_lon']:.6f}° E`"
                    )

                # ---- 8. Data Transparency Log ----
                with st.expander("🔍 View Evidence & Why This Advisory? (Data Transparency Log)"):
                    st.markdown("### Evidence Trail & Data Freshness Log")
                    st.markdown(f"**Analysis generated at:** {advisory.generated_at_ist}")

                    st.markdown("#### 1. Official Warning Source Breakdown")
                    for s_name, s_det in adv_official.sources.items():
                        st.markdown(f"- **Source:** `{s_det.name}`")
                        st.markdown(f"  - **Status:** `{s_det.status}`")
                        if s_det.timestamp:
                            st.markdown(f"  - **Timestamp:** {s_det.timestamp}")
                        if s_det.future_advisory:
                            st.markdown(f"  - **Future advisory:** {s_det.future_advisory}")
                        if s_det.description:
                            st.markdown(f"  - **Notes:** {s_det.description}")
                        if s_det.error_msg:
                            st.markdown(f"  - **Fetch error:** {s_det.error_msg}")

                    st.markdown("#### 2. Rainfall & Weather Data")
                    st.markdown(f"- **Weather Source:** {adv_rain.weather_source}")
                    if adv_rain.available:
                        st.markdown(f"- **Data Timestamp:** {adv_rain.data_timestamp}")
                        st.markdown(f"- **Rainfall (last 24h):** {adv_rain.precip_24h_mm:.1f} mm")
                        st.markdown(f"- **Rainfall (last 72h):** {adv_rain.precip_72h_mm:.1f} mm")
                        st.markdown(f"- **7-Day Cumulative:** {adv_rain.total_7day_mm:.1f} mm")
                        st.markdown(f"- **Max Daily (7 days):** {adv_rain.max_daily_recent_mm:.1f} mm")
                        st.markdown(f"- **Forecast next 24h:** {adv_rain.forecast_24h_mm:.1f} mm")
                    else:
                        st.warning(f"Rainfall data unavailable: {adv_rain.fetch_error}")

                    st.markdown("#### 3. Satellite Acquisition & Baseline")
                    st.markdown(f"- **Pre-event window:** `{result['before_date'] - result['window']}` to `{result['before_date']}`")
                    st.markdown(f"- **Post-event window:** `{result['after_date']}` to `{result['after_date'] + result['window']}`")
                    st.markdown(f"- **Current window:** `{result['present_date'] - result['window']}` to `{result['present_date']}`")
                    st.markdown(f"- **Post-event vs Current Satellite Change:** `{present_status.get('changed_fraction', 0.0) * 100:.1f}%`")
                    st.markdown(f"- **Disclaimer:** {adv_sat.limitation_note}")

                    st.markdown("#### 4. Historical AI Model Detection")
                    st.markdown(f"- **Historical AI change fraction:** `{adv_evidence.historical_change_fraction * 100:.1f}%`")

                    st.markdown("#### 5. Decision Rules Triggered")
                    for rule in advisory.rules_triggered:
                        st.markdown(f"- {rule}")

                    st.markdown("---")
                    st.caption(
                        "⚠️ **Safety Notice:** LandSentry Present Visit Advisory never guarantees 100% safety. "
                        "Absence of an official warning or low satellite change does not certify absence of landslide risk. "
                        "Always consult local district authorities before travelling."
                    )

            st.subheader("4. Visual Comparison")
            post_with_mask = result["rgb_post"].copy()
            detected = result["change_mask"] > 0
            post_with_mask[detected] = (
                0.55 * result["rgb_post"][detected]
                + 0.45 * np.array([1.0, 0.0, 0.0], dtype=np.float32)
            )
            post_with_mask[~result["display_valid"]] = 0.55
            current_with_mask = result["rgb_current"].copy()
            present_detected = result["present_mask"] > 0
            current_with_mask[present_detected] = (
                0.55 * result["rgb_current"][present_detected]
                + 0.45 * np.array([1.0, 0.65, 0.0], dtype=np.float32)
            )
            current_with_mask[~result["current_display_valid"]] = 0.55
            comparison_cols = st.columns(5, gap="small")
            comparison_images = (
                (result["rgb_pre"], f"Pre-event Sentinel-2\nTaken: {result['before_date'] - result['window']} to {result['before_date']}"),
                (result["rgb_post"], f"Post-event Sentinel-2\nTaken: {result['after_date']} to {result['after_date'] + result['window']}"),
                (post_with_mask, f"Detected Change Overlay\nCompared with post-event: {result['after_date']} to {result['after_date'] + result['window']}"),
                (result["rgb_current"], f"Present Sentinel-2\nTaken: {result['present_date'] - result['window']} to {result['present_date']}"),
                (current_with_mask, f"Present Change Screening\nCompared with post-event: {result['after_date']} to {result['after_date'] + result['window']}"),
            )
            for column, (image, caption) in zip(comparison_cols, comparison_images):
                with column:
                    st.image(image, caption=caption, use_container_width=True)

            if result["changed_pixels"] > 0:
                st.warning("Potential land-cover changes detected within the selected Area of Interest.")
            else:
                st.success("No significant changes detected above the selected threshold.")

            st.subheader("5. Geotagged Change Location Map")
            raster_bounds = result["raster_bounds"]
            map_lat_center = (raster_bounds["north"] + raster_bounds["south"]) / 2
            map_lon_center = (raster_bounds["east"] + raster_bounds["west"]) / 2
            result_map = folium.Map(location=[map_lat_center, map_lon_center], zoom_start=13)

            folium.GeoJson(
                {"type": "Feature", "geometry": result["aoi_geojson"]},
                name="Area of Interest",
                style_function=lambda _: {"color": "blue", "fillOpacity": 0.05},
            ).add_to(result_map)

            if result["changed_pixels"] > 0:
                overlay_rgba = np.zeros((*result["change_mask"].shape, 4), dtype=np.uint8)
                overlay_rgba[result["change_mask"] == 1] = [255, 0, 0, 160]
                folium.raster_layers.ImageOverlay(
                    image=overlay_rgba,
                    bounds=[
                        [raster_bounds["south"], raster_bounds["west"]],
                        [raster_bounds["north"], raster_bounds["east"]],
                    ],
                    opacity=1.0,
                    name="Detected Change Mask",
                ).add_to(result_map)

            if result["present_mask"].sum() > 0:
                present_overlay = np.zeros((*result["present_mask"].shape, 4), dtype=np.uint8)
                present_overlay[result["present_mask"] == 1] = [255, 165, 0, 150]
                folium.raster_layers.ImageOverlay(
                    image=present_overlay,
                    bounds=[
                        [result["present_bounds"]["south"], result["present_bounds"]["west"]],
                        [result["present_bounds"]["north"], result["present_bounds"]["east"]],
                    ],
                    opacity=1.0,
                    name="Present Change Screening",
                ).add_to(result_map)

            for i, region in enumerate(result["change_regions"], start=1):
                folium.Marker(
                    location=[region["centroid_lat"], region["centroid_lon"]],
                    popup=folium.Popup(
                        f"<b>Landslide detected: {region['place_name']}</b><br>"
                        f"Latitude: {region['centroid_lat']:.6f}<br>"
                        f"Longitude: {region['centroid_lon']:.6f}<br>"
                        f"Affected area: {region['area_km2']:.4f} km²",
                        max_width=250,
                    ),
                    tooltip=f"Landslide Region {i}",
                    icon=folium.Icon(color="red", icon="warning-sign"),
                ).add_to(result_map)

            for i, region in enumerate(result["present_regions"], start=1):
                folium.Marker(
                    location=[region["centroid_lat"], region["centroid_lon"]],
                    popup=folium.Popup(
                        f"<b>Present screening hotspot: {region['place_name']}</b><br>"
                        f"Latitude: {region['centroid_lat']:.6f}<br>"
                        f"Longitude: {region['centroid_lon']:.6f}<br>"
                        f"Area: {region['area_km2']:.4f} km²",
                        max_width=250,
                    ),
                    tooltip=f"Present Change Region {i}",
                    icon=folium.Icon(color="orange", icon="info-sign"),
                ).add_to(result_map)

            folium.LayerControl().add_to(result_map)
            result_map.fit_bounds([
                [raster_bounds["south"], raster_bounds["west"]],
                [raster_bounds["north"], raster_bounds["east"]],
            ])
            st_folium(result_map, height=500, width=900, key="result_map")
    else:
        st.info("Draw an AOI, choose your dates, and click the button to run the live detection.")

