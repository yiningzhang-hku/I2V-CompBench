import json

input_path = 'E:/I2V-CompBench/outputs/image_analysis/image_parse.jsonl'
backup_path = input_path + '.backup_before_cleanup'

with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Backup
with open(backup_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print(f"Backup saved to {backup_path}")

kept = []
removed_empty = 0
removed_invalid = 0
for i, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        removed_invalid += 1
        continue
    # Remove entries with empty VLM response (likely 429 failures)
    if not obj.get("parse_success", False) and not obj.get("raw_vlm_output", "").strip():
        removed_empty += 1
        continue
    kept.append(line)

with open(input_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(kept) + '\n')

print(f"Removed {removed_empty} empty-response entries")
print(f"Removed {removed_invalid} invalid JSON lines")
print(f"Kept {len(kept)} entries")
