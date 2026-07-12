import json

lines = open('E:/I2V-CompBench/outputs/image_analysis/image_parse.jsonl', 'r', encoding='utf-8').readlines()
ids = set()
invalid = 0
empty = 0
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        ids.add(obj["sample_id"])
        if not obj.get("parse_success", False) and not obj.get("raw_vlm_output", "").strip():
            empty += 1
    except Exception:
        invalid += 1

print(f"Unique processed IDs: {len(ids)}")
print(f"Empty raw entries: {empty}")
print(f"Invalid lines: {invalid}")
print(f"Remaining to process: {9897 - len(ids)}")
