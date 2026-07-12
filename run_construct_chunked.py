"""Run Phase 2 construct with per-plan progress and chunking."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from i2vcompbench.phase2.construct_inputs import _construct_for_question, _extract_reference_subject
from i2vcompbench.utils.api_client import Phase2SiliconFlowClient
from i2vcompbench.utils.io import load_config, benchmark_paths, iter_jsonl, write_jsonl, Phase1Bundle
from i2vcompbench.utils.templates import TemplateRegistry
from i2vcompbench.schemas.phase2 import InputAssetManifest

cfg = load_config("configs/phase2.yaml")
paths = benchmark_paths(cfg["output_dir"])

plans = list(iter_jsonl(paths["question_plans"]))
print(f"Loaded {len(plans)} question plans")

sampled_path = paths["sampled_recipes"]
sampled_recipe_lookup = {}
for row in iter_jsonl(sampled_path):
    rid = (row.get("recipe") or {}).get("recipe_id")
    if rid:
        sampled_recipe_lookup[rid] = row.get("recipe") or {}
print(f"Loaded {len(sampled_recipe_lookup)} sampled recipes")

bundle = Phase1Bundle(cfg["phase1_bundle_dir"])
construct_cfg = cfg.get("construct", {})
long_edge = int(construct_cfg.get("long_edge", 854))
enable_t2i = bool(construct_cfg.get("enable_t2i", True))
client = None
if enable_t2i:
    try:
        client = Phase2SiliconFlowClient(cfg)
    except Exception as e:
        print(f"T2I client init failed; disabled: {e}")
        client = None

registry = TemplateRegistry()

out = []
start = time.time()
for i, plan in enumerate(plans):
    qid = plan["question_id"]
    if i % 10 == 0:
        elapsed = time.time() - start
        print(f"[{i}/{len(plans)}] {qid} (elapsed {elapsed:.1f}s, success {len(out)})")
    dim = plan["dimension"]
    template = registry.get(dim)
    subtype_block = template.find_subtype(plan.get("subtype"), plan.get("input_mode"))
    target_subjects = (plan.get("target_plan") or {}).get("target_subjects") or []
    primary = target_subjects[0] if target_subjects else {}
    secondary = target_subjects[1] if len(target_subjects) >= 2 else {}
    slots = {
        "target_subject": primary.get("description") or "the subject",
        "reference_subject": secondary.get("description") or _extract_reference_subject(plan),
        "direction": "to the right",
        "target_relation": (((plan.get("target_plan") or {}).get("target_relation") or {}).get("value") or ((plan.get("target_plan") or {}).get("target_relation") or {}).get("relation") or "right_of"),
        "background": "neutral",
    }
    manifest = _construct_for_question(
        plan=plan,
        bundle=bundle,
        template_subtype_block=subtype_block,
        client=client,
        paths=paths,
        long_edge=long_edge,
        enable_t2i=enable_t2i,
        slots=slots,
        sampled_recipe_lookup=sampled_recipe_lookup,
    )
    if manifest is not None:
        out.append(manifest)

print(f"Constructed inputs for {len(out)} questions")
write_jsonl(paths["input_assets_manifest"], [m.model_dump() for m in out])
print(f"Wrote {len(out)} manifests -> {paths['input_assets_manifest']}")
