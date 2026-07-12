"""
Fix: populate correct first_frame_16x9.png in by_dimension/ from first_frames/ intermediate dir.

Steps:
1. Delete all existing first_frame_16x9.png in by_dimension (V1 remnants, possibly stale)
2. For each question dir, copy the correct <qid>_16x9.png from first_frames/ as first_frame_16x9.png
"""
import shutil
from pathlib import Path

BY_DIM = Path("data/benchmark_dataset/by_dimension")
FIRST_FRAMES = Path("data/benchmark_dataset/first_frames")
DIMS = ["attribute_binding", "action_binding", "motion_binding", "background_dynamics", "view_transformation"]


def main():
    # Step 1: Remove all existing 16x9 in by_dimension (V1 remnants)
    removed = 0
    for dim in DIMS:
        dim_dir = BY_DIM / dim
        if not dim_dir.exists():
            continue
        for qdir in dim_dir.iterdir():
            if not qdir.is_dir():
                continue
            old = qdir / "first_frame_16x9.png"
            if old.exists():
                old.unlink()
                removed += 1
    print(f"[step1] Removed {removed} stale V1 remnant 16x9 files")

    # Step 2: Copy correct V2 16x9 from first_frames/<qid>_16x9.png
    copied = 0
    missing_src = 0
    for dim in DIMS:
        dim_dir = BY_DIM / dim
        if not dim_dir.exists():
            continue
        for qdir in sorted(dim_dir.iterdir()):
            if not qdir.is_dir():
                continue
            qid = qdir.name
            src = FIRST_FRAMES / f"{qid}_16x9.png"
            dst = qdir / "first_frame_16x9.png"
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing_src += 1
                print(f"  [warn] no source 16x9 for {qid}")

    print(f"[step2] Copied {copied} correct V2 16x9 files to by_dimension/")
    if missing_src:
        print(f"  [warn] {missing_src} questions had no source 16x9 in first_frames/")

    # Verify
    total_ok = 0
    for dim in DIMS:
        dim_dir = BY_DIM / dim
        if not dim_dir.exists():
            continue
        for qdir in dim_dir.iterdir():
            if not qdir.is_dir():
                continue
            if (qdir / "first_frame_16x9.png").exists():
                total_ok += 1
    print(f"[verify] {total_ok}/650 question dirs now have first_frame_16x9.png")


if __name__ == "__main__":
    main()
