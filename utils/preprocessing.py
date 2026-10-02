"""
Image preprocessing utilities: cloud masking, radiometric normalization,
and sub-pixel co-registration between pre- and post-event image pairs.
"""
import numpy as np
import cv2


def mask_clouds_qa60(qa60_band):
    """
    Sentinel-2 QA60 cloud mask.
    Bit 10 = opaque clouds, bit 11 = cirrus clouds (ESA Sentinel-2 QA60 spec).
    Returns a boolean array where True = clear (usable) pixel.
    """
    qa = qa60_band.astype(np.int32)
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    return ((qa & cloud_bit) == 0) & ((qa & cirrus_bit) == 0)


def normalize_percentile(img, lower=2, upper=98):
    """
    Per-band percentile clipping + min-max scaling to [0, 1].
    Reduces the effect of sensor noise, seasonal illumination differences,
    and outlier pixels between the pre- and post-event acquisitions, so the
    model isn't confused by lighting/season rather than genuine terrain change.

    img: [H, W, C] float array.
    """
    out = np.empty_like(img, dtype=np.float32)
    for b in range(img.shape[-1]):
        band = img[..., b]
        lo, hi = np.percentile(band, [lower, upper])
        if hi - lo < 1e-6:
            out[..., b] = 0.0
            continue
        band = np.clip(band, lo, hi)
        out[..., b] = (band - lo) / (hi - lo)
    return out


def compute_ndvi(img, red_idx=0, nir_idx=3):
    """
    Calculates NDVI = (NIR - RED) / (NIR + RED) safely.
    Follows constant indexing: Red (B4) is 0, NIR (B8) is 3 by default.
    Returns an array of shape [H, W, 1].
    """
    red = img[..., red_idx].astype(np.float32)
    nir = img[..., nir_idx].astype(np.float32)
    
    denominator = nir + red
    # Handle division by zero
    ndvi = np.divide((nir - red), denominator, out=np.zeros_like(red), where=(denominator != 0))
    return np.expand_dims(ndvi, axis=-1)


def fuse_features(img_arr, dem_arr=None, use_ndvi=True, use_dem=True):
    """
    Fuses RGB+NIR [H, W, 4] with optional NDVI [H, W, 1] and DEM [H, W, 3].
    Returns the fused feature array.
    """
    features = [img_arr]
    
    if use_ndvi:
        ndvi = compute_ndvi(img_arr)
        features.append(ndvi)
        
    if use_dem and dem_arr is not None:
        features.append(dem_arr)
        
    return np.concatenate(features, axis=-1)


def coregister(post_img, pre_img_ref):
    """
    Sub-pixel co-registration of the post-event image onto the pre-event
    reference frame using ECC (Enhanced Correlation Coefficient) alignment.

    Sentinel-2 L2A products are already geo-referenced, so in practice this
    acts as a refinement step rather than full registration. Falls back to
    the original (unaligned) image if ECC fails to converge, which can
    happen on very low-texture patches (e.g. still water).
    """
    def to_gray(img):
        if img.ndim == 3 and img.shape[-1] >= 3:
            return cv2.cvtColor(img[..., :3].astype(np.float32), cv2.COLOR_RGB2GRAY)
        return (img[..., 0] if img.ndim == 3 else img).astype(np.float32)

    pre_gray = to_gray(pre_img_ref)
    post_gray = to_gray(post_img)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        _, warp_matrix = cv2.findTransformECC(
            pre_gray, post_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria
        )
        aligned = cv2.warpAffine(
            post_img, warp_matrix, (post_img.shape[1], post_img.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        return aligned
    except cv2.error:
        return post_img
