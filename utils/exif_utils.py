import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif_data(image_path):
    """Extract EXIF data into a dictionary."""
    image = Image.open(image_path)
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]
                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value
    return exif_data


def get_decimal_coordinates(info):
    """
    Safely converts EXIF GPSInfo format to explicit decimal degrees.
    """
    for key in ['Latitude', 'Longitude']:
        if 'GPS' + key not in info or 'GPS' + key + 'Ref' not in info:
            return None, None

    def convert_to_degrees(value):
        def to_float(x):
            try:
                return float(x)
            except:
                return x[0] / x[1]
                
        d0 = to_float(value[0])
        d1 = to_float(value[1])
        d2 = to_float(value[2])
        return d0 + (d1 / 60.0) + (d2 / 3600.0)

    try:
        lat = convert_to_degrees(info['GPSLatitude'])
        if info['GPSLatitudeRef'] != "N":                     
            lat = 0 - lat

        lon = convert_to_degrees(info['GPSLongitude'])
        if info['GPSLongitudeRef'] != "E":
            lon = 0 - lon
            
        # Hard limits
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            return None, None

        return lat, lon
    except Exception:
        return None, None


def extract_gps_from_image(image_path):
    """
    Returns (lat, lon) if extracted correctly, else (None, None).
    """
    try:
        exif_data = get_exif_data(image_path)
        if "GPSInfo" in exif_data:
            return get_decimal_coordinates(exif_data["GPSInfo"])
    except Exception:
        pass
    return None, None
