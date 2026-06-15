"""
Image helpers for Phase 2 input construction.

P0 scope: open / resize / crop-from-bbox / save (PNG). Background inpainting is a
placeholder (returns the original image with a TODO note).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
from PIL import Image


DEFAULT_LONG_EDGE = 1024


def open_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def save_image(img: Image.Image, path: str | Path, fmt: str = "PNG") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if img.mode == "RGBA" and fmt.upper() == "JPEG":
        img = img.convert("RGB")
    img.save(p, format=fmt)
    return str(p)


def resize_long_edge(img: Image.Image, long_edge: int = DEFAULT_LONG_EDGE) -> Image.Image:
    w, h = img.size
    le = max(w, h)
    if le <= long_edge:
        return img
    scale = long_edge / le
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def crop_normalized_bbox(
    img: Image.Image,
    bbox: Tuple[float, float, float, float],
    pad_ratio: float = 0.05,
) -> Image.Image:
    """Crop using normalized bbox [x0,y0,x1,y1] in [0,1]."""
    w, h = img.size
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    bw, bh = (x1 - x0), (y1 - y0)
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    px0 = max(0.0, x0 - pad_x)
    py0 = max(0.0, y0 - pad_y)
    px1 = min(1.0, x1 + pad_x)
    py1 = min(1.0, y1 + pad_y)
    box = (int(px0 * w), int(py0 * h), int(px1 * w), int(py1 * h))
    if box[2] <= box[0] or box[3] <= box[1]:
        logger.warning(f"Invalid crop box derived from bbox={bbox}, returning full image")
        return img
    return img.crop(box)


def bytes_to_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def image_resolution_ok(img: Image.Image, min_long_edge: int = 384) -> bool:
    w, h = img.size
    return max(w, h) >= min_long_edge


def fetch_url_to_image(url: str, timeout: int = 30) -> Optional[Image.Image]:
    """Download an HTTP(S) image URL to a PIL.Image. Used for T2I provider responses."""
    try:
        import requests  # local import to keep utils light at import time
    except ImportError:
        logger.error("requests not installed; cannot fetch URL image")
        return None
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return bytes_to_image(r.content)
    except Exception as e:
        logger.error(f"Failed to fetch image url {url}: {e}")
        return None
