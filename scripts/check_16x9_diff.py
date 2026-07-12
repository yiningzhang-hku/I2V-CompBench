"""Check how first_frame.png differs from first_frame_16x9.png across all questions."""
from PIL import Image
from pathlib import Path

root = Path("data/benchmark_dataset/by_dimension")
dims = ["attribute_binding", "action_binding", "motion_binding", "background_dynamics", "view_transformation"]
stats = {"resize": 0, "crop": 0, "letterbox": 0}
crop_examples = []
letterbox_examples = []

for dim in dims:
    dim_dir = root / dim
    if not dim_dir.exists():
        continue
    for qdir in sorted(dim_dir.iterdir()):
        if not qdir.is_dir():
            continue
        ff = qdir / "first_frame.png"
        ff16 = qdir / "first_frame_16x9.png"
        if not ff.exists() or not ff16.exists():
            continue
        img = Image.open(ff)
        w, h = img.size
        ratio = w / h
        tgt = 854 / 480
        rel = abs(ratio - tgt) / tgt
        if rel <= 0.04:
            stats["resize"] += 1
        elif ratio > tgt:
            stats["crop"] += 1
            lost_pct = (w - h * tgt) / w * 100
            if len(crop_examples) < 8:
                crop_examples.append(f"  {dim}/{qdir.name}: {w}x{h} ratio={ratio:.3f} -> lost {lost_pct:.1f}% width (center crop)")
        else:
            stats["letterbox"] += 1
            if len(letterbox_examples) < 5:
                letterbox_examples.append(f"  {dim}/{qdir.name}: {w}x{h} ratio={ratio:.3f} -> letterbox (black bars)")

total = sum(stats.values())
print(f"Total questions with both images: {total}")
print(f"  resize (near 16:9, visually same): {stats['resize']} ({stats['resize']/total*100:.0f}%)")
print(f"  crop (wider than 16:9, edges lost): {stats['crop']} ({stats['crop']/total*100:.0f}%)")
print(f"  letterbox (narrower, black bars):   {stats['letterbox']} ({stats['letterbox']/total*100:.0f}%)")
print()
if crop_examples:
    print("CROP examples (prompt describes full image, 16x9 lost left/right edges):")
    for ex in crop_examples:
        print(ex)
    print()
if letterbox_examples:
    print("LETTERBOX examples (content preserved but has black bars):")
    for ex in letterbox_examples:
        print(ex)
