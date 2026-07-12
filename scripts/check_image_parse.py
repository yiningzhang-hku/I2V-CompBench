import json

lines = open('E:/I2V-CompBench/outputs/image_analysis/image_parse.jsonl', 'r', encoding='utf-8').readlines()
print(f"Total lines: {len(lines)}")

# Check last 5
for line in lines[-5:]:
    obj = json.loads(line)
    sid = obj["sample_id"]
    ok = obj["parse_success"]
    err = obj.get("parse_error", "")
    raw = obj.get("raw_vlm_output", "")[:80]
    print(f"  {sid}: success={ok}, error={err}, raw_preview={raw}")
