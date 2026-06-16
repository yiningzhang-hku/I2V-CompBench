"""
Migrate existing benchmark dataset images to 720P-class (long edge = 1280).

Background
----------
Phase 2 §6.4 (revised) requires every input image (first_frames/, ref_images/) to be
resized to long_edge=1280 with aspect ratio preserved (path A1: equal-ratio upscale,
no letterbox, no crop). Images already produced before this rule are typically
224x126 / 224x224 / 126x224 (TIP-I2V native low-res frames) and must be migrated.

Usage
-----
    # dry-run (default): only print stats, no file write
    python scripts/migrate_images_to_720p.py

    # apply in-place overwrite
    python scripts/migrate_images_to_720p.py --apply

    # custom root / pattern
    python scripts/migrate_images_to_720p.py --root data/benchmark_dataset --apply

Idempotency
-----------
Already-1280-long-edge images are skipped. Safe to rerun.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow running this script directly without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from PIL import Image  # noqa: E402

from i2vcompbench.utils.image import (  # noqa: E402
    DEFAULT_LONG_EDGE,
    open_image,
    resize_long_edge,
    save_image,
)


def iter_target_pngs(root: Path):
    """Yield every PNG under root that should follow the 720P rule.

    by_dimension/<dim>/<question_id>/first_frame.png
    by_dimension/<dim>/<question_id>/ref_images/*.png
    """
    if not root.exists():
        return
    for p in root.rglob("*.png"):
        # only target benchmark inputs; skip caches / thumbnails / docs
        parts = {part.lower() for part in p.parts}
        if any(skip in parts for skip in {"__pycache__", ".git", "phase1_bundle"}):
            continue
        yield p


def migrate_one(path: Path, long_edge: int, apply: bool) -> str:
    """Return migration status for stats: skipped / resized / error."""
    try:
        img = open_image(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [error] open failed: {path} -> {e}")
        return "error"

    w, h = img.size
    if max(w, h) == long_edge:
        return "skipped"

    new_img = resize_long_edge(img, long_edge=long_edge, enlarge=True)
    new_w, new_h = new_img.size
    print(f"  [resize] {path.relative_to(path.anchor)}: {w}x{h} -> {new_w}x{new_h}")

    if apply:
        save_image(new_img, path, fmt="PNG")
    return "resized"


def main():
    parser = argparse.ArgumentParser(description="Migrate benchmark images to 720P long edge.")
    parser.add_argument(
        "--root",
        default=str(_REPO_ROOT / "data" / "benchmark_dataset"),
        help="Root dir to scan (default: data/benchmark_dataset).",
    )
    parser.add_argument(
        "--long-edge",
        type=int,
        default=DEFAULT_LONG_EDGE,
        help=f"Target long edge (default: {DEFAULT_LONG_EDGE}).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually overwrite files. Without this flag, only print plan (dry-run).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    long_edge = int(args.long_edge)
    print(f"[migrate] root={root}")
    print(f"[migrate] long_edge={long_edge}")
    print(f"[migrate] mode={'APPLY (overwrite)' if args.apply else 'DRY-RUN'}")
    print()

    stats = Counter()
    new_size_dist = Counter()
    for p in iter_target_pngs(root):
        status = migrate_one(p, long_edge, args.apply)
        stats[status] += 1
        if status == "resized":
            with Image.open(p) as im:
                # If apply=True we just overwrote; size matches new. If dry-run,
                # size on disk is still old, so derive predicted size manually.
                if args.apply:
                    new_size_dist[im.size] += 1
                else:
                    w, h = im.size
                    le = max(w, h)
                    scale = long_edge / le
                    new_size_dist[(int(round(w * scale)), int(round(h * scale)))] += 1

    print()
    print("=" * 60)
    print(f"[summary] total scanned : {sum(stats.values())}")
    print(f"[summary] skipped       : {stats['skipped']} (already at long_edge={long_edge})")
    print(f"[summary] resized       : {stats['resized']}")
    print(f"[summary] errors        : {stats['error']}")
    if new_size_dist:
        print("[summary] post-resize size distribution (top 10):")
        for size, n in new_size_dist.most_common(10):
            print(f"           {size[0]:>5} x {size[1]:<5} : {n}")
    if not args.apply and stats["resized"] > 0:
        print()
        print("[hint] dry-run only. Re-run with --apply to overwrite the files in place.")


if __name__ == "__main__":
    main()
