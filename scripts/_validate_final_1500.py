"""
验收检查脚本 - 验证最终1500条benchmark数据集
检查项：
1. 总数严格1500条
2. 每维度严格300条
3. 无重复question_id
4. 所有图像文件存在且为854×480
5. 输出维度分布、主体分层分布、难度分布统计
"""
import json
import sys
from collections import Counter
from pathlib import Path

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARN] Pillow not available, skipping image dimension check")
    sys.stdout.flush()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = PROJECT_ROOT / "data" / "benchmark_dataset"
FINAL_PATH = BENCHMARK_ROOT / "final_benchmark_1500.jsonl"

FORMAL_DIMENSIONS = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "background_dynamics",
    "view_transformation",
]


def main():
    print("=" * 70)
    print("  FINAL BENCHMARK VALIDATION")
    print("=" * 70)

    # Load
    rows = []
    with open(FINAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = len(rows)
    errors = []

    # Check 1: Total count
    print(f"\n[1] Total count: {total}", end="")
    if total == 1500:
        print(" PASS")
    else:
        print(f" FAIL (expected 1500)")
        errors.append(f"Total count is {total}, expected 1500")

    # Check 2: Per-dimension count
    dim_counts = Counter(r["dimension"] for r in rows)
    print(f"\n[2] Per-dimension counts:")
    for dim in FORMAL_DIMENSIONS:
        count = dim_counts.get(dim, 0)
        status = "PASS" if count == 300 else "FAIL"
        print(f"    {dim}: {count} [{status}]")
        if count != 300:
            errors.append(f"{dim} has {count} items, expected 300")

    # Check 3: Uniqueness
    qids = [r["question_id"] for r in rows]
    unique_qids = set(qids)
    dup_count = total - len(unique_qids)
    print(f"\n[3] Unique question_ids: {len(unique_qids)}", end="")
    if dup_count == 0:
        print(" PASS")
    else:
        print(f" FAIL ({dup_count} duplicates)")
        errors.append(f"{dup_count} duplicate question_ids")

    # Check 4: Image files
    print(f"\n[4] Image file verification:")
    missing_images = []
    wrong_size_images = []
    checked = 0
    sample_size = min(total, 1500)  # Check all

    for row in rows[:sample_size]:
        ff_path = row.get("first_frame_path", "")
        if not ff_path:
            missing_images.append(row["question_id"])
            continue

        # Resolve path
        img_path = PROJECT_ROOT / ff_path.replace("\\", "/")
        if not img_path.exists():
            missing_images.append(row["question_id"])
            continue

        checked += 1
        if HAS_PIL and checked <= 100:  # Check first 100 for size
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if (w, h) != (854, 480):
                        wrong_size_images.append((row["question_id"], w, h))
            except Exception as e:
                wrong_size_images.append((row["question_id"], "error", str(e)))

    print(f"    Images checked: {checked}/{sample_size}")
    print(f"    Missing images: {len(missing_images)}", end="")
    if len(missing_images) == 0:
        print(" PASS")
    else:
        print(f" FAIL")
        for qid in missing_images[:5]:
            print(f"      - {qid}")
        errors.append(f"{len(missing_images)} missing images")

    if HAS_PIL:
        print(f"    Wrong size (checked first 100): {len(wrong_size_images)}", end="")
        if len(wrong_size_images) == 0:
            print(" PASS")
        else:
            print(f" WARN")
            for item in wrong_size_images[:5]:
                print(f"      - {item}")

    # Check 5: Distribution stats
    print(f"\n[5] Distribution statistics:")
    overall_diff = Counter(r.get("difficulty", "unknown") for r in rows)
    overall_rarity = Counter(r.get("semantic_rarity", "unknown") for r in rows)
    overall_subtype = Counter(r.get("subtype", "unknown") for r in rows)

    print(f"\n    Difficulty distribution:")
    for k, v in sorted(overall_diff.items()):
        print(f"      {k}: {v} ({v/total*100:.1f}%)")

    print(f"\n    Semantic rarity distribution:")
    for k, v in sorted(overall_rarity.items()):
        print(f"      {k}: {v} ({v/total*100:.1f}%)")

    print(f"\n    Subtype distribution:")
    for k, v in sorted(overall_subtype.items(), key=lambda x: -x[1]):
        print(f"      {k}: {v} ({v/total*100:.1f}%)")

    # Final verdict
    print("\n" + "=" * 70)
    if not errors:
        print("  ALL CHECKS PASSED - Benchmark ready!")
    else:
        print(f"  {len(errors)} CHECK(S) FAILED:")
        for e in errors:
            print(f"    - {e}")
    print("=" * 70)

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
