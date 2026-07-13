"""Analyze remaining rare words after rule-based fixes."""
import json, re, sys
from collections import Counter
from wordfreq import zipf_frequency as zf

recs = [json.loads(l) for l in open('data/benchmark_dataset/phase3_manifest.jsonl', 'r', encoding='utf-8')]
freq = Counter()

for r in recs:
    tokens = re.findall(r'\b[a-zA-Z]+\b', r['prompt'])
    for i, t in enumerate(tokens):
        if len(t) <= 2:
            continue
        if t.isupper():
            continue
        if i > 0 and t[0].isupper():
            continue
        low = t.lower()
        score = zf(low, 'en')
        if score < 3.5:
            freq[low] += 1

print(f'Total remaining rare word hits: {sum(freq.values())}')
print(f'Unique rare words: {len(freq)}')
print(f'\nTop 80 remaining rare words:')
for w, c in freq.most_common(80):
    print(f'  {w}: {c} (zipf={zf(w, "en"):.2f})')
