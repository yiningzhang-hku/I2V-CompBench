from pathlib import Path
root = Path("data/benchmark_dataset/by_dimension")
dims = ["attribute_binding","action_binding","motion_binding","background_dynamics","view_transformation"]
total = has16x9 = no16x9 = 0
per_dim = {}
for d in dims:
    dim_dir = root / d
    if not dim_dir.exists(): continue
    c_has = c_no = 0
    for q in sorted(dim_dir.iterdir()):
        if not q.is_dir(): continue
        total += 1
        if (q / "first_frame_16x9.png").exists():
            has16x9 += 1
            c_has += 1
        else:
            no16x9 += 1
            c_no += 1
    per_dim[d] = (c_has, c_no)

print(f"Total dirs: {total}  |  has 16x9: {has16x9}  |  no 16x9: {no16x9}")
for d, (h, n) in per_dim.items():
    print(f"  {d}: has={h} no={n}")

# Also check: what files exist in a dir WITHOUT 16x9?
print("\nSample dir WITHOUT 16x9:")
for d in dims:
    for q in sorted((root / d).iterdir()):
        if q.is_dir() and not (q / "first_frame_16x9.png").exists():
            print(f"  {q.name}: {[f.name for f in sorted(q.iterdir())]}")
            break
    break

# And in a dir WITH 16x9:
print("\nSample dir WITH 16x9:")
for d in dims:
    for q in sorted((root / d).iterdir()):
        if q.is_dir() and (q / "first_frame_16x9.png").exists():
            print(f"  {q.name}: {[f.name for f in sorted(q.iterdir())]}")
            break
    break
