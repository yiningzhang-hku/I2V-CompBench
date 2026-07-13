"""Check pans distribution."""
import json, re, sys
from wordfreq import zipf_frequency as zf

recs = [json.loads(l) for l in open('data/benchmark_dataset/phase3_manifest.jsonl', 'r', encoding='utf-8')]
pans_only = 0
pans_total = 0
view_pans = 0

for r in recs:
    tokens = re.findall(r'\b[a-zA-Z]+\b', r['prompt'])
    rare = []
    seen = set()
    for i, t in enumerate(tokens):
        if len(t) <= 2: continue
        if t.isupper(): continue
        if i > 0 and t[0].isupper(): continue
        low = t.lower()
        if low in seen: continue
        if zf(low, 'en') < 3.5:
            rare.append(low)
            seen.add(low)
    
    has_pans = 'pans' in rare
    if has_pans:
        pans_total += 1
        if r['dimension'] == 'view_transformation':
            view_pans += 1
        if len(rare) == 1:
            pans_only += 1

print(f'Records with pans: {pans_total}')
print(f'  - in view_transformation: {view_pans}')
print(f'  - with ONLY pans as rare word: {pans_only}')
print(f'  - with pans + other rare words: {pans_total - pans_only}')

# Also check zooms
zooms_count = sum(1 for r in recs if 'zooms' in re.findall(r'\b\w+\b', r['prompt'].lower()))
print(f'\nRecords with "zooms": {zooms_count}')
