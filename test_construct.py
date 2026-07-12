"""Test construct on first 3 question plans."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from i2vcompbench.phase2.construct_inputs import construct_inputs
from i2vcompbench.utils.io import load_config, benchmark_paths, iter_jsonl
from i2vcompbench.utils.templates import TemplateRegistry

cfg = load_config("configs/phase2.yaml")
paths = benchmark_paths(cfg["output_dir"])

print("Reading first 3 plans...")
plans = list(iter_jsonl(paths["question_plans"]))[:3]
for p in plans:
    print(f"  {p['question_id']} dim={p['dimension']} input_mode={p.get('input_mode')} recipe_id={p.get('recipe_id')}")
    dim = p["dimension"]
    template = TemplateRegistry().get(dim)
    subtype_block = template.find_subtype(p.get("subtype"), p.get("input_mode"))
    print(f"    subtype={p.get('subtype')} found={subtype_block is not None}")
    target_subjects = (p.get("target_plan") or {}).get("target_subjects") or []
    print(f"    target_subjects={len(target_subjects)}")
    required = (p.get("input_plan") or {}).get("required_images") or []
    print(f"    required_images={len(required)}: {[(r.get('role'), r.get('source_preference')) for r in required]}")

print("\nRunning construct_inputs on first 3 plans only...")
# Monkey-patch iter_jsonl to return only first 3 plans
from i2vcompbench.phase2 import construct_inputs as ci_mod
orig_iter_jsonl = ci_mod.iter_jsonl

def limited_iter_jsonl(path):
    for i, row in enumerate(orig_iter_jsonl(path)):
        if i < 50:
            yield row
        else:
            break

ci_mod.iter_jsonl = limited_iter_jsonl

manifests = construct_inputs(cfg)
print(f"Success: {len(manifests)} manifests")
for m in manifests:
    print(f"  {m.question_id}: {len(m.assets)} assets")
