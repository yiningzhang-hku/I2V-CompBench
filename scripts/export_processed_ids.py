"""
Export all processed sample IDs from Phase 1 manifest to a text file.

After Phase 1 completes, run this script to produce a flat list of all
sample_id values that have been processed (from manifest_clean.jsonl).
This serves as a dedup registry for future expansion runs.

Usage:
    python scripts/export_processed_ids.py
    python scripts/export_processed_ids.py --out processed_ids.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from i2vcompbench.utils.io import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Export all processed sample IDs.")
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "configs" / "phase1.yaml"),
        help="Path to phase1.yaml",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output txt file path (default: <manifest_dir>/processed_ids.txt)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_dir = Path(cfg["paths"]["manifest_dir"])
    manifest_clean = manifest_dir / "manifest_clean.jsonl"

    if not manifest_clean.exists():
        print(f"[error] manifest_clean.jsonl not found at {manifest_clean}")
        return 1

    # Read all sample IDs
    ids = []
    with open(manifest_clean, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("sample_id", "")
            if sid:
                ids.append(sid)

    # Also include bad_samples (they were processed but failed validation)
    bad_path = manifest_dir / "bad_samples.jsonl"
    if bad_path.exists():
        with open(bad_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = row.get("sample_id", "")
                if sid:
                    ids.append(sid)

    # Deduplicate and sort
    ids = sorted(set(ids))

    # Write output
    out_path = Path(args.out) if args.out else (manifest_dir / "processed_ids.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Phase 1 processed sample IDs (total: {len(ids)})\n")
        f.write(f"# Generated from: {manifest_clean}\n")
        f.write(f"# Use this file to avoid re-processing in future expansion runs.\n")
        f.write("\n")
        for sid in ids:
            f.write(sid + "\n")

    print(f"[export] Wrote {len(ids)} unique sample IDs to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
