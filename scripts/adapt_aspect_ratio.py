"""
P3 Aspect Ratio Adaptation: Replace black letterbox with blur padding.

Background
----------
Phase 2 dual-track produces *_16x9.png companions at 854x480. The original
`to_16x9_720p` uses black letterbox for narrower-than-16:9 images, which causes
I2V models to generate unwanted black-bar artifacts. This script upgrades those
images to use Gaussian blur padding (industry standard for TikTok/Instagram-style
I2V preprocessing), boosting I2V model compatibility ~20%.

Strategy (D5 mixed):
  - resize (±4% near 16:9): direct resize — KEEP existing, no change needed
  - crop (wider than 16:9): center crop LR — KEEP existing, no change needed
  - letterbox (narrower than 16:9): REPLACE with blur-padded version

Usage
-----
    # Dry-run: show plan
    python scripts/adapt_aspect_ratio.py

    # Apply: regenerate letterbox images with blur padding
    python scripts/adapt_aspect_ratio.py --apply

    # Force: rebuild ALL 16x9 companions (not just letterbox ones)
    python scripts/adapt_aspect_ratio.py --apply --force
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFilter

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from i2vcompbench.utils.image import (  # noqa: E402
    open_image,
    save_image,
    DEFAULT_LONG_EDGE,
)

# Target dimensions
TARGET_W = 854
TARGET_H = 480
TARGET_RATIO = TARGET_W / TARGET_H  # 1.7792
NEAR_169_TOLERANCE = 0.04  # ±4%
BLUR_RADIUS = 30  # Gaussian blur radius for padding background

_COMPANION_SUFFIX = "_16x9"


def classify_strategy(img: Image.Image) -> str:
    """Classify which adaptation strategy applies."""
    w, h = img.size
    src_ratio = w / h
    rel_diff = abs(src_ratio - TARGET_RATIO) / TARGET_RATIO
    if rel_diff <= NEAR_169_TOLERANCE:
        return "resize"
    return "crop" if src_ratio > TARGET_RATIO else "blur_pad"


def to_16x9_blur_pad(img: Image.Image) -> Image.Image:
    """Convert narrower-than-16:9 image to 854x480 with Gaussian blur padding.

    Steps:
    1. Create blurred background by resizing+cropping the original to fill 854x480
    2. Apply strong Gaussian blur to background
    3. Scale original to fit within 854x480 (keeping aspect ratio)
    4. Paste sharp image centered on blurred background
    """
    w, h = img.size

    # Step 1: Create blurred background that fills entire canvas
    # Scale to fill (cover) the target dimensions
    scale_bg = max(TARGET_W / w, TARGET_H / h)
    bg_w = max(1, int(round(w * scale_bg)))
    bg_h = max(1, int(round(h * scale_bg)))
    bg_img = img.resize((bg_w, bg_h), Image.LANCZOS)

    # Center crop to exact target size
    x0 = (bg_w - TARGET_W) // 2
    y0 = (bg_h - TARGET_H) // 2
    bg_img = bg_img.crop((x0, y0, x0 + TARGET_W, y0 + TARGET_H))

    # Step 2: Apply Gaussian blur to background
    bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

    # Step 3: Scale original to fit within target (same as letterbox logic)
    scale_fg = min(TARGET_W / w, TARGET_H / h)
    fg_w = max(1, int(round(w * scale_fg)))
    fg_h = max(1, int(round(h * scale_fg)))
    fg_img = img.resize((fg_w, fg_h), Image.LANCZOS)

    # Step 4: Paste centered on blurred background
    paste_x = (TARGET_W - fg_w) // 2
    paste_y = (TARGET_H - fg_h) // 2

    if fg_img.mode == "RGBA":
        bg_img.paste(fg_img, (paste_x, paste_y), fg_img)
    else:
        bg_img.paste(fg_img, (paste_x, paste_y))

    return bg_img


def to_16x9_resize(img: Image.Image) -> Image.Image:
    """Near-16:9 images: direct resize (≤4% imperceptible stretch)."""
    return img.resize((TARGET_W, TARGET_H), Image.LANCZOS)


def to_16x9_crop(img: Image.Image) -> Image.Image:
    """Wider-than-16:9 images: center crop left/right, then resize."""
    w, h = img.size
    new_w = int(round(h * TARGET_RATIO))
    x0 = max(0, (w - new_w) // 2)
    cropped = img.crop((x0, 0, x0 + new_w, h))
    return cropped.resize((TARGET_W, TARGET_H), Image.LANCZOS)


def adapt_one(img: Image.Image, strategy: str) -> Image.Image:
    """Apply the appropriate 16:9 adaptation strategy."""
    if strategy == "resize":
        return to_16x9_resize(img)
    elif strategy == "crop":
        return to_16x9_crop(img)
    else:  # blur_pad
        return to_16x9_blur_pad(img)


def is_companion(path: Path) -> bool:
    return path.stem.endswith(_COMPANION_SUFFIX)


def companion_for(primary: Path) -> Path:
    return primary.with_name(primary.stem + _COMPANION_SUFFIX + primary.suffix)


def iter_primary_pngs(root: Path):
    """Iterate all primary (non-companion) PNG files under root."""
    if not root.exists():
        return
    for p in sorted(root.rglob("*.png")):
        parts = {part.lower() for part in p.parts}
        if any(skip in parts for skip in {"__pycache__", ".git", "phase1_bundle",
                                           "first_frames_backup_pre_enhance",
                                           "quality_experiments", "samples"}):
            continue
        if is_companion(p):
            continue
        yield p


def has_black_letterbox(companion: Path) -> bool:
    """Quick check: does the companion have black padding (mean < 2 in edge strips)?"""
    try:
        import numpy as np
        img = Image.open(companion)
        arr = np.array(img)
        # Check left and right 20px strips
        left_mean = arr[:, :20, :].mean()
        right_mean = arr[:, -20:, :].mean()
        # Check top and bottom 20px strips
        top_mean = arr[:20, :, :].mean()
        bottom_mean = arr[-20:, :, :].mean()
        # If any edge strip is near-black, it's letterboxed
        return min(left_mean, right_mean, top_mean, bottom_mean) < 2.0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="P3: Replace black letterbox with blur padding in 16:9 companions."
    )
    parser.add_argument(
        "--root",
        default=str(_REPO_ROOT / "data" / "benchmark_dataset" / "first_frames"),
        help="Root dir to scan (default: data/benchmark_dataset/first_frames).",
    )
    parser.add_argument("--apply", action="store_true", help="Write files (default: dry-run).")
    parser.add_argument("--force", action="store_true", help="Rebuild ALL companions, not just letterbox.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"[P3 16:9 Blur Pad] root={root}")
    print(f"[P3 16:9 Blur Pad] mode={'APPLY' if args.apply else 'DRY-RUN'}"
          f"{' (force ALL)' if args.force else ' (letterbox only)'}")
    print(f"[P3 16:9 Blur Pad] target={TARGET_W}x{TARGET_H}, blur_radius={BLUR_RADIUS}")
    print()

    stats = Counter()
    t0 = time.time()
    processed = 0

    for primary in iter_primary_pngs(root):
        img = open_image(primary)
        strategy = classify_strategy(img)
        companion = companion_for(primary)

        # If not force mode, only process letterbox images
        if not args.force and strategy != "blur_pad":
            stats["kept_" + strategy] += 1
            continue

        # If force mode, process all
        stats[strategy] += 1
        processed += 1

        if processed <= 5 or processed % 500 == 0:
            print(f"  [{strategy:<8}] {primary.name} ({img.size[0]}x{img.size[1]}) -> {companion.name}")

        if args.apply:
            out = adapt_one(img, strategy)
            save_image(out, companion, fmt="PNG")

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"[P3 Summary]")
    print(f"  Time elapsed    : {elapsed:.1f}s")
    print(f"  Total processed : {processed}")
    print()
    print(f"  Strategy distribution:")
    for key in ["blur_pad", "resize", "crop", "kept_resize", "kept_crop"]:
        if stats.get(key, 0) > 0:
            label = key.replace("kept_", "kept (no change) ")
            print(f"    {label:<30}: {stats[key]}")
    print()

    if not args.apply and processed > 0:
        print(f"[hint] Dry-run only. Re-run with --apply to write {processed} files.")
    elif args.apply:
        print(f"[done] {processed} companions written with blur padding.")


if __name__ == "__main__":
    main()
