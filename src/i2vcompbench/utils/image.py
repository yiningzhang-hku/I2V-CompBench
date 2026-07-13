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


DEFAULT_LONG_EDGE = 854  # 480P long edge (854x480, 16:9); see Phase 2 §6.4


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


def resize_long_edge(
    img: Image.Image,
    long_edge: int = DEFAULT_LONG_EDGE,
    enlarge: bool = False,
) -> Image.Image:
    """Resize so that max(W, H) == long_edge, keeping aspect ratio.

    enlarge=False (default): only shrink oversized images, leave small ones intact.
        Used by Phase 1 API upload path where we just want an upper bound.
    enlarge=True: force-resize regardless of original size, including upscaling small
        images (e.g. TIP-I2V 224x126 -> 854x480). Used by Phase 2 input construction
        where benchmark assets must be uniformly 480P-class (long edge = 854).
    """
    w, h = img.size
    le = max(w, h)
    if le == long_edge:
        return img
    if le < long_edge and not enlarge:
        return img
    scale = long_edge / le
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
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


# ----------------------------------------------------------------------------
# 16:9 inference-side adapter (Phase 2 §6.4 dual-track output)
# ----------------------------------------------------------------------------

DEFAULT_INFERENCE_W = 854
DEFAULT_INFERENCE_H = 480
_NEAR_169_TOLERANCE = 0.04  # ±4% 认为已足够接近 16:9，直接 resize不可见形变


def to_16x9_720p(
    img: Image.Image,
    target_w: int = DEFAULT_INFERENCE_W,
    target_h: int = DEFAULT_INFERENCE_H,
    blur_radius: int = 30,
) -> Image.Image:
    """Adapt arbitrary-aspect image to a strict ``target_w x target_h`` (default 854x480) canvas.

    Strategy (selected automatically, no bbox required):
      1. ±4% near 16:9   -> direct resize, distortion <= 4% (imperceptible)
      2. wider than 16:9 -> center crop left/right, then resize
      3. else            -> blur padding: scale to fit + Gaussian-blurred background

    Case 3 uses Gaussian blur padding (industry standard for TikTok/Instagram-style
    I2V preprocessing) instead of black letterbox, avoiding I2V model artifacts from
    pure-black padding regions.

    Used by Phase 2 to produce ``*_16x9.png`` companion files for I2V model inference,
    while keeping the original ``*.png`` (long-edge=854, native ratio) intact for
    evaluator P-axis ground truth.
    """
    from PIL import ImageFilter

    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {(w, h)}")
    src_ratio = w / h
    tgt_ratio = target_w / target_h
    rel_diff = abs(src_ratio - tgt_ratio) / tgt_ratio

    if rel_diff <= _NEAR_169_TOLERANCE:
        # case 1: near-16:9, direct resize (≤ 4% stretch)
        return img.resize((target_w, target_h), Image.LANCZOS)

    if src_ratio > tgt_ratio:
        # case 2: wider than 16:9 -> center crop left/right
        new_w = int(round(h * tgt_ratio))
        x0 = max(0, (w - new_w) // 2)
        cropped = img.crop((x0, 0, x0 + new_w, h))
        return cropped.resize((target_w, target_h), Image.LANCZOS)

    # case 3: narrower than 16:9 (incl. 1:1, 9:16) -> blur padding
    # Create blurred background by scaling image to cover target, then blur
    scale_bg = max(target_w / w, target_h / h)
    bg_w = max(1, int(round(w * scale_bg)))
    bg_h = max(1, int(round(h * scale_bg)))
    bg_img = img.resize((bg_w, bg_h), Image.LANCZOS)
    bx0 = (bg_w - target_w) // 2
    by0 = (bg_h - target_h) // 2
    bg_img = bg_img.crop((bx0, by0, bx0 + target_w, by0 + target_h))
    bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Scale original to fit within target (keep aspect ratio)
    scale_fg = min(target_w / w, target_h / h)
    fg_w = max(1, int(round(w * scale_fg)))
    fg_h = max(1, int(round(h * scale_fg)))
    fg_img = img.resize((fg_w, fg_h), Image.LANCZOS)

    # Paste sharp image centered on blurred background
    x0 = (target_w - fg_w) // 2
    y0 = (target_h - fg_h) // 2
    if fg_img.mode == "RGBA":
        bg_img.paste(fg_img, (x0, y0), fg_img)
    else:
        bg_img.paste(fg_img, (x0, y0))
    return bg_img


def classify_16x9_strategy(img: Image.Image) -> str:
    """Return which strategy ``to_16x9_720p`` would apply: 'resize' / 'crop' / 'blur_pad'.
    Useful for audit / dataset_card stats."""
    w, h = img.size
    src_ratio = w / h
    tgt_ratio = DEFAULT_INFERENCE_W / DEFAULT_INFERENCE_H
    if abs(src_ratio - tgt_ratio) / tgt_ratio <= _NEAR_169_TOLERANCE:
        return "resize"
    return "crop" if src_ratio > tgt_ratio else "blur_pad"


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
