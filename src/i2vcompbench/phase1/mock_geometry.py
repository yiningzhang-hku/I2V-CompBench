"""
Mock \u51e0\u4f55\u5de5\u5177 \u2014 P0 \u9636\u6bb5\u4ece VLM \u8f93\u51fa\u7684 position_in_frame\uff089 \u5bab\u683c\uff09\u53cd\u63a8 bbox\uff0c
\u5e76\u63d0\u4f9b PIL \u88c1\u526a\u80fd\u529b\u3002\u672a\u6765\u53ef\u66ff\u6362\u4e3a SAM / Grounding-DINO \u800c\u4e0d\u52a8\u4e0b\u6e38\u4ee3\u7801\u3002

bbox \u5750\u6807\u7cfb\uff1a\u5f52\u4e00\u5316 [x0, y0, x1, y1] \u2208 [0, 1]\uff0c\u539f\u70b9\u5728\u5de6\u4e0a\u3002
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

# 9 \u5bab\u683c \u2192 \u5f52\u4e00\u5316 bbox
# row: top / center / bottom    col: left / center / right
_GRID: Dict[str, Tuple[float, float, float, float]] = {
    # 9 \u4e2a\u6807\u51c6\u4f4d
    "upper-left":   (0.00, 0.00, 0.40, 0.40),
    "top":          (0.30, 0.00, 0.70, 0.40),
    "upper-right":  (0.60, 0.00, 1.00, 0.40),
    "center-left":  (0.00, 0.30, 0.40, 0.70),
    "center":       (0.25, 0.25, 0.75, 0.75),
    "center-right": (0.60, 0.30, 1.00, 0.70),
    "lower-left":   (0.00, 0.60, 0.40, 1.00),
    "bottom":       (0.30, 0.60, 0.70, 1.00),
    "lower-right":  (0.60, 0.60, 1.00, 1.00),
    # \u522b\u540d\u517c\u5bb9
    "left":         (0.00, 0.30, 0.40, 0.70),
    "right":        (0.60, 0.30, 1.00, 0.70),
}

DEFAULT_BBOX: Tuple[float, float, float, float] = (0.20, 0.20, 0.80, 0.80)


def position_to_bbox(position_in_frame: Optional[str]) -> Tuple[float, float, float, float]:
    """\u628a VLM \u8f93\u51fa\u7684 position_in_frame \u6620\u5c04\u4e3a\u5f52\u4e00\u5316 bbox\u3002\u672a\u77e5\u4f4d\u7f6e\u8fd4\u56de\u5145\u586b\u4e2d\u592e\u3002"""
    if not position_in_frame:
        return DEFAULT_BBOX
    key = position_in_frame.strip().lower().replace("_", "-")
    return _GRID.get(key, DEFAULT_BBOX)


def normalized_to_pixel(
    bbox_norm: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """\u5f52\u4e00\u5316 bbox \u8f6c\u4e3a\u50cf\u7d20\u5750\u6807 (x0, y0, x1, y1)\u3002"""
    w, h = image_size
    x0, y0, x1, y1 = bbox_norm
    return (
        max(0, int(round(x0 * w))),
        max(0, int(round(y0 * h))),
        min(w, int(round(x1 * w))),
        min(h, int(round(y1 * h))),
    )


def crop_to_path(
    image_path: str,
    bbox_norm: Tuple[float, float, float, float],
    output_path: str,
    pad_ratio: float = 0.05,
) -> Optional[Tuple[int, int]]:
    """
    \u6309\u5f52\u4e00\u5316 bbox \u88c1\u526a\u539f\u56fe\u5230 output_path\u3002\u8fd4\u56de\u88c1\u526a\u540e (W, H)\u3002
    \u53ef\u9009 pad_ratio\uff1a\u5411\u5916\u6269 5% \u8fb9\u8ddd\uff0c\u907f\u514d\u5207\u8fb9\u3002
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None

    if not os.path.isfile(image_path):
        return None

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        W, H = im.size
        x0, y0, x1, y1 = bbox_norm
        # padding
        bw, bh = x1 - x0, y1 - y0
        x0 = max(0.0, x0 - bw * pad_ratio)
        y0 = max(0.0, y0 - bh * pad_ratio)
        x1 = min(1.0, x1 + bw * pad_ratio)
        y1 = min(1.0, y1 + bh * pad_ratio)

        px = normalized_to_pixel((x0, y0, x1, y1), (W, H))
        cropped = im.crop(px)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cropped.save(output_path, format="JPEG", quality=92)
        return cropped.size


def estimate_tracking_feasibility(
    bbox_norm: Tuple[float, float, float, float],
    is_animate: bool,
) -> str:
    """
    \u6839\u636e bbox \u9762\u79ef \u00d7 \u662f\u5426\u6d3b\u4f53\u8fd4\u56de\u8ddf\u8e2a\u96be\u5ea6\u4f30\u8ba1\u3002
    \u8fd4\u56de\u503c\u2208 {easy, medium, hard, infeasible}\u3002
    """
    x0, y0, x1, y1 = bbox_norm
    area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
    if area < 0.02:
        return "infeasible"
    if area < 0.06:
        return "hard"
    if area < 0.20:
        return "medium" if is_animate else "easy"
    return "easy"
