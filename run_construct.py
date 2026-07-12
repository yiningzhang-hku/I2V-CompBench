"""Run Phase 2 construct directly with progress logging."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from i2vcompbench.phase2.construct_inputs import construct_inputs
from i2vcompbench.utils.io import load_config, benchmark_paths, write_jsonl

cfg = load_config("configs/phase2.yaml")
paths = benchmark_paths(cfg["output_dir"])

print("Starting construct...")
manifests = construct_inputs(cfg)
print(f"Constructed {len(manifests)} manifests")

manifests_data = [m.model_dump() for m in manifests]
write_jsonl(paths["input_assets_manifest"], manifests_data)
print(f"Wrote {len(manifests_data)} manifests -> {paths['input_assets_manifest']}")
