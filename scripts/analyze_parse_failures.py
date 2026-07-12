import json
from collections import Counter

lines = open('E:/I2V-CompBench/outputs/image_analysis/image_parse.jsonl', 'r', encoding='utf-8').readlines()
print(f"Total lines: {len(lines)}")

success = 0
fail = 0
errors = Counter()
empty_raw = 0
invalid = 0
for i, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception as e:
        invalid += 1
        if invalid <= 3:
            print(f"Invalid JSON at line {i+1}: {line[:100]}... ({e})")
        continue
    if obj["parse_success"]:
        success += 1
    else:
        fail += 1
        err = obj.get("parse_error", "unknown")
        errors[err] += 1
        if not obj.get("raw_vlm_output", "").strip():
            empty_raw += 1

print(f"Valid entries: {success + fail}")
print(f"Success: {success}")
print(f"Fail: {fail}")
print(f"Invalid JSON lines: {invalid}")
print(f"Empty raw (likely 429): {empty_raw}")
print("\nError breakdown:")
for err, cnt in errors.most_common():
    print(f"  {err}: {cnt}")
