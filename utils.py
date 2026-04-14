import os
import base64
import io
import numpy as np
from PIL import Image
try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

def pdf_to_images(pdf_path, poppler_path=None, dpi=200):
    """
    Converts PDF pages to PIL images.
    """
    if not HAS_PDF2IMAGE:
        raise ImportError("pdf2image library is required. Install with: pip install pdf2image")
    
    return convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)

def pil_to_base64(image, max_dim=1600):
    """
    Converts a PIL image to a base64 encoded string for API consumption.
    """
    # Resize if too large
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
        
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def crop_region(image, box):
    """
    Crops a region from the image based on a normalized bounding box [ymin, xmin, ymax, xmax].
    Coordinates are expected in 0-1000 scale.
    """
    width, height = image.size
    ymin, xmin, ymax, xmax = box
    
    left = (xmin / 1000) * width
    top = (ymin / 1000) * height
    right = (xmax / 1000) * width
    bottom = (ymax / 1000) * height
    
    return image.crop((left, top, right, bottom))

def scale_to_unit_eight(arr):
    """
    Scales a numpy array to 0-255 uint8 for display.
    """
    amin, amax = np.min(arr), np.max(arr)
    if amax == amin:
        return np.zeros_like(arr, dtype=np.uint8)
    return (((arr - amin) / (amax - amin + 1e-6)) * 255).astype(np.uint8)
