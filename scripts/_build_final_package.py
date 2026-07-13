"""
生成最终数据集包 final_1500/
组织结构：
  final_1500/
  ├── metadata.jsonl          # 1500条完整元数据
  ├── statistics.json         # 统计报告（已生成）
  ├── by_dimension/
  │   ├── attribute_binding/  # 300条元数据 + 图像
  │   ├── action_binding/
  │   ├── motion_binding/
  │   ├── background_dynamics/
  │   └── view_transformation/
  └── images/                 # 1500张16:9图像（复制）
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = PROJECT_ROOT / "data" / "benchmark_dataset"
FINAL_JSONL = BENCHMARK_ROOT / "final_benchmark_1500.jsonl"
OUTPUT_ROOT = BENCHMARK_ROOT / "final_1500"

FORMAL_DIMENSIONS = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "background_dynamics",
    "view_transformation",
]


def main():
    print("=" * 60)
    print("  Generating final_1500/ dataset package")
    print("=" * 60)

    # Load final selection
    rows = []
    with open(FINAL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} selected candidates")

    # Create directory structure
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    images_dir = OUTPUT_ROOT / "images"
    images_dir.mkdir(exist_ok=True)

    for dim in FORMAL_DIMENSIONS:
        dim_dir = OUTPUT_ROOT / "by_dimension" / dim
        dim_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata.jsonl (same as final_benchmark_1500.jsonl)
    metadata_path = OUTPUT_ROOT / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Written: {metadata_path}")

    # Write per-dimension JSONL files
    by_dim = {}
    for row in rows:
        dim = row["dimension"]
        by_dim.setdefault(dim, []).append(row)

    for dim in FORMAL_DIMENSIONS:
        dim_rows = by_dim.get(dim, [])
        dim_path = OUTPUT_ROOT / "by_dimension" / dim / "metadata.jsonl"
        with open(dim_path, "w", encoding="utf-8") as f:
            for row in dim_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {dim}: {len(dim_rows)} items -> {dim_path.name}")

    # Copy images to images/ directory
    print("\nCopying images...")
    copied = 0
    missing = 0
    for row in rows:
        ff_path = row.get("first_frame_path", "")
        if not ff_path:
            missing += 1
            continue

        src = PROJECT_ROOT / ff_path.replace("\\", "/")
        if not src.exists():
            missing += 1
            continue

        # Copy to images/ dir with question_id as name
        dst = images_dir / f"{row['question_id']}_16x9.png"
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1

    print(f"  Copied: {copied} images")
    if missing > 0:
        print(f"  Missing: {missing} images")

    # Also copy to by_dimension/dim/ dirs
    print("\nCopying to per-dimension dirs...")
    for dim in FORMAL_DIMENSIONS:
        dim_dir = OUTPUT_ROOT / "by_dimension" / dim
        dim_rows = by_dim.get(dim, [])
        dim_copied = 0
        for row in dim_rows:
            ff_path = row.get("first_frame_path", "")
            if not ff_path:
                continue
            src = PROJECT_ROOT / ff_path.replace("\\", "/")
            if not src.exists():
                continue
            dst = dim_dir / f"{row['question_id']}_16x9.png"
            if not dst.exists():
                shutil.copy2(src, dst)
            dim_copied += 1
        print(f"  {dim}: {dim_copied} images")

    # Verify statistics.json exists
    stats_path = OUTPUT_ROOT / "statistics.json"
    if stats_path.exists():
        print(f"\nStatistics: {stats_path} (exists)")
    else:
        print(f"\n[WARN] Statistics file not found!")

    # Print final summary
    total_images = len(list(images_dir.glob("*.png")))
    print(f"\n{'='*60}")
    print(f"  PACKAGE COMPLETE")
    print(f"{'='*60}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  metadata.jsonl: {len(rows)} rows")
    print(f"  images/: {total_images} files")
    for dim in FORMAL_DIMENSIONS:
        dim_dir = OUTPUT_ROOT / "by_dimension" / dim
        dim_imgs = len(list(dim_dir.glob("*.png")))
        dim_meta = 1 if (dim_dir / "metadata.jsonl").exists() else 0
        print(f"  by_dimension/{dim}/: {dim_imgs} images, {dim_meta} metadata")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
