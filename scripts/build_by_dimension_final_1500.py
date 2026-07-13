"""
将 final_benchmark_1500.jsonl 中的1500条最终记录
按维度组织到 by_dimension_final_1500/ 目录结构中。

对每条记录：
- 优先从现有 by_dimension/ 中复制整个题目目录
- 若不存在，则从 first_frames/ 复制图像并生成 metadata.json
"""

import json
import shutil
from pathlib import Path
from collections import Counter

# ===== 路径设置 =====
BASE_DIR = Path(r"d:\projects\I2V-CompBench\data\benchmark_dataset")
JSONL_PATH = BASE_DIR / "final_benchmark_1500.jsonl"
SRC_BY_DIM = BASE_DIR / "by_dimension"
DST_BY_DIM = BASE_DIR / "by_dimension_final_1500"
FIRST_FRAMES_DIR = BASE_DIR / "first_frames"

# ===== 读取 JSONL =====
records = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"读取到 {len(records)} 条记录")

# 统计维度分布
dim_counter = Counter(r["dimension"] for r in records)
print("维度分布:")
for dim, count in sorted(dim_counter.items()):
    print(f"  {dim}: {count}")

# ===== 创建目标目录 =====
if DST_BY_DIM.exists():
    shutil.rmtree(DST_BY_DIM)
DST_BY_DIM.mkdir(parents=True)

# 创建维度子目录
dimensions = ["action_binding", "attribute_binding", "motion_binding",
              "background_dynamics", "view_transformation"]
for dim in dimensions:
    (DST_BY_DIM / dim).mkdir()

# ===== 处理每条记录 =====
copied_from_existing = 0
created_new = 0
missing_images = []

for record in records:
    qid = record["question_id"]
    dim = record["dimension"]
    
    src_dir = SRC_BY_DIM / dim / qid
    dst_dir = DST_BY_DIM / dim / qid
    
    if src_dir.exists():
        # 从现有目录复制
        shutil.copytree(src_dir, dst_dir)
        copied_from_existing += 1
    else:
        # 创建新目录
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制 16x9 图像作为 first_frame_16x9.png
        frame_16x9_name = f"{qid}_16x9.png"
        frame_16x9_src = FIRST_FRAMES_DIR / frame_16x9_name
        
        # 复制原始图像作为 first_frame.png
        frame_orig_name = f"{qid}.png"
        frame_orig_src = FIRST_FRAMES_DIR / frame_orig_name
        
        has_16x9 = False
        has_orig = False
        
        if frame_16x9_src.exists():
            shutil.copy2(frame_16x9_src, dst_dir / "first_frame_16x9.png")
            has_16x9 = True
        
        if frame_orig_src.exists():
            shutil.copy2(frame_orig_src, dst_dir / "first_frame.png")
            has_orig = True
        
        if not has_16x9 and not has_orig:
            missing_images.append(qid)
        
        # 生成 prompt.json (从 JSONL 记录中提取元数据)
        metadata = {
            "question_id": qid,
            "dimension": dim,
            "subtype": record.get("subtype", ""),
            "input_mode": record.get("input_mode", "single_image"),
            "difficulty": record.get("difficulty", ""),
            "rarity": record.get("semantic_rarity", ""),
            "prompt": record.get("prompt", ""),
            "evaluator_tools": record.get("evaluator_tools", []),
            "expected_failure_modes": record.get("expected_failure_modes", []),
            "preservation_set": record.get("preservation_set", []),
            "target_subjects": record.get("target_subjects", []),
            "target_relation": record.get("target_relation"),
            "contrastive_pair_id": record.get("contrastive_pair_id"),
            "contrastive_role": record.get("contrastive_role", "original"),
            "input_files": ["first_frame.png"]
        }
        
        with open(dst_dir / "prompt.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        created_new += 1

# ===== 验证 =====
print(f"\n===== 处理完成 =====")
print(f"从现有目录复制: {copied_from_existing}")
print(f"新创建目录: {created_new}")
print(f"总计: {copied_from_existing + created_new}")

if missing_images:
    print(f"\n警告: {len(missing_images)} 个题目缺少图像文件:")
    for qid in missing_images[:10]:
        print(f"  - {qid}")

# 验证每个维度的目录数
print(f"\n===== 验证维度目录数 =====")
all_ok = True
for dim in dimensions:
    dim_dir = DST_BY_DIM / dim
    count = len([d for d in dim_dir.iterdir() if d.is_dir()])
    status = "OK" if count == 300 else "MISMATCH"
    if count != 300:
        all_ok = False
    print(f"  {dim}: {count} 个目录 [{status}]")

# 验证文件完整性
print(f"\n===== 验证文件完整性 =====")
incomplete_dirs = []
for dim in dimensions:
    dim_dir = DST_BY_DIM / dim
    for item_dir in sorted(dim_dir.iterdir()):
        if item_dir.is_dir():
            files = [f.name for f in item_dir.iterdir() if f.is_file()]
            # 至少需要 prompt.json 和某种形式的图像
            has_prompt = "prompt.json" in files
            has_image = "first_frame.png" in files or "first_frame_16x9.png" in files
            if not has_prompt or not has_image:
                incomplete_dirs.append(str(item_dir.relative_to(DST_BY_DIM)))

if incomplete_dirs:
    print(f"  {len(incomplete_dirs)} 个目录文件不完整:")
    for d in incomplete_dirs[:10]:
        print(f"    - {d}")
else:
    print(f"  全部 1500 个目录文件完整!")

if all_ok and not incomplete_dirs:
    print(f"\n全部验证通过! by_dimension_final_1500/ 目录结构已就绪。")
