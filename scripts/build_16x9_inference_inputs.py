"""
Generate 854x480 strict 16:9 inference companions for every benchmark image.

Background
----------
Phase 2 §6.4 dual-track output:
  * <name>.png       — native-aspect 480P (long edge = 854), evaluator P-axis truth
  * <name>_16x9.png  — strict 854x480 16:9, fed to I2V models that require 16:9 input

This script scans data/benchmark_dataset/, finds every primary PNG (skipping any that
already end with `_16x9`) and produces the companion file alongside it. Idempotent:
re-running skips inputs whose companion already exists and is up-to-date.

Strategy applied (utils.image.to_16x9_720p):
  * ratio in 16:9 ± 4%   -> direct resize (≤4% imperceptible stretch)
  * wider than 16:9      -> center crop left/right
  * narrower than 16:9   -> letterbox (equal-ratio fit + black canvas)

Usage
-----
    # dry-run (default): print plan + strategy distribution
    python scripts/build_16x9_inference_inputs.py

    # apply
    python scripts/build_16x9_inference_inputs.py --apply

    # rebuild even if companion already exists
    python scripts/build_16x9_inference_inputs.py --apply --force
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from i2vcompbench.utils.image import (  # noqa: E402
    classify_16x9_strategy,
    open_image,
    save_image,
    to_16x9_720p,
)

_COMPANION_SUFFIX = "_16x9"


def is_companion(path: Path) -> bool:
    return path.stem.endswith(_COMPANION_SUFFIX)


def companion_for(primary: Path) -> Path:
    return primary.with_name(primary.stem + _COMPANION_SUFFIX + primary.suffix)


def iter_primary_pngs(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*.png"):
        parts = {part.lower() for part in p.parts}
        if any(skip in parts for skip in {"__pycache__", ".git", "phase1_bundle"}):
            continue
        if is_companion(p):
            continue
        yield p


def process_one(primary: Path, apply: bool, force: bool) -> str:
    companion = companion_for(primary)
    if companion.exists() and not force:
        return "skipped_exists"

    try:
        img = open_image(primary)
    except Exception as e:  # noqa: BLE001
        print(f"  [error] open failed: {primary} -> {e}")
        return "error"

    strategy = classify_16x9_strategy(img)
    print(f"  [{strategy:<10}] {primary.name} ({img.size[0]}x{img.size[1]}) -> {companion.name} (854x480)")

    if apply:
        out = to_16x9_720p(img)
        save_image(out, companion, fmt="PNG")
    return strategy


def main():
    parser = argparse.ArgumentParser(description="Build 854x480 16:9 companions for benchmark images.")
    parser.add_argument(
        "--root",
        default=str(_REPO_ROOT / "data" / "benchmark_dataset"),
        help="Root dir to scan (default: data/benchmark_dataset).",
    )
    parser.add_argument("--apply", action="store_true", help="Actually write files (default: dry-run).")
    parser.add_argument("--force", action="store_true", help="Rebuild companions even if they exist.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"[16x9] root={root}")
    print(f"[16x9] mode={'APPLY' if args.apply else 'DRY-RUN'}{' (force rebuild)' if args.force else ''}")
    print()

    stats = Counter()
    for primary in iter_primary_pngs(root):
        stats[process_one(primary, args.apply, args.force)] += 1

    total = sum(stats.values())
    print()
    print("=" * 60)
    print(f"[summary] primary scanned : {total}")
    print(f"[summary] resize          : {stats.get('resize', 0)}    (near 16:9, direct resize)")
    print(f"[summary] crop            : {stats.get('crop', 0)}    (wider than 16:9, center crop LR)")
    print(f"[summary] blur_pad        : {stats.get('blur_pad', 0)}    (narrower than 16:9, blur padding)")
    print(f"[summary] skipped_exists  : {stats.get('skipped_exists', 0)}")
    print(f"[summary] errors          : {stats.get('error', 0)}")
    if not args.apply and (stats.get('resize', 0) + stats.get('crop', 0) + stats.get('blur_pad', 0)) > 0:
        print()
        print("[hint] dry-run only. Re-run with --apply to write companions.")


if __name__ == "__main__":
    main()
