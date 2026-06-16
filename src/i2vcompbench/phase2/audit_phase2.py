"""
Phase 2 · Step 8: audit benchmark_dataset and emit dataset_card.md.

Checks:
  - phase3_manifest.jsonl line count == sum(samples/<dim>.jsonl)
  - per (dimension / input_mode / subtype / difficulty / semantic_rarity / source_type) actual vs quota
  - contrastive pair coverage on enabled dimensions（§4.8 step 4：每对 ≥1 original + ≥1 baseline_*）
  - multi_image reference_quality distribution（读 samples/<dim>.jsonl 的 _audit.multi_reference_quality）
  - QC status histogram
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from ..schemas.phase2 import DIMENSIONS_V2
from ..utils.io import (
    benchmark_paths,
    iter_jsonl,
    load_config,
    read_json,
)


def _samples_iter(paths: Dict[str, Path]):
    for dim in DIMENSIONS_V2:
        path = paths["samples"] / f"{dim}.jsonl"
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            yield row


def _format_table(headers: List[str], rows: List[List[Any]]) -> str:
    out_lines = []
    out_lines.append("| " + " | ".join(headers) + " |")
    out_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out_lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out_lines)


def audit(config: Dict[str, Any]) -> str:
    paths = benchmark_paths(config["output_dir"])

    # 1) line count consistency
    per_dim_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_per_dim = 0
    for row in _samples_iter(paths):
        per_dim_rows[row["dimension"]].append(row)
        total_per_dim += 1
    manifest_rows = (
        list(iter_jsonl(paths["phase3_manifest"]))
        if paths["phase3_manifest"].exists()
        else []
    )
    consistent = len(manifest_rows) == total_per_dim

    # 2) quota gap
    quota = read_json(paths["quota_plan"]) if paths["quota_plan"].exists() else {"buckets": []}
    quota_target_by_key: Dict[str, int] = {}
    for b in quota.get("buckets", []):
        key = (
            b["dimension"]
            + "::"
            + b["input_mode_or_subtype"]
            + "::"
            + b["difficulty"]
            + "::"
            + b["rarity"]
        )
        quota_target_by_key[key] = b["target_count"]

    actual_by_key: Counter = Counter()
    for row in manifest_rows:
        sub = row.get("subtype") or row.get("input_mode")
        key = f'{row["dimension"]}::{sub}::{row.get("difficulty","medium")}::{row.get("semantic_rarity","common")}'
        actual_by_key[key] += 1

    quota_rows: List[List[Any]] = []
    for k, target in sorted(quota_target_by_key.items()):
        actual = actual_by_key.get(k, 0)
        quota_rows.append([k, target, actual, target - actual])
    quota_table = _format_table(["bucket", "target", "actual", "shortfall"], quota_rows)

    # 3) per-dimension stats
    dim_stats_rows = []
    for dim in DIMENSIONS_V2:
        rows = per_dim_rows.get(dim, [])
        n = len(rows)
        n_single = sum(1 for r in rows if r.get("input_mode") == "single_image")
        n_multi = n - n_single
        dim_stats_rows.append([dim, n, n_single, n_multi])
    dim_table = _format_table(
        ["dimension", "total", "single_image", "multi_image"], dim_stats_rows
    )

    # 4) contrastive coverage——§4.8 step 4：每对 pair 要求 ≥1 original + ≥1 baseline_*
    enabled_dims = set(
        (config.get("quota", {}).get("contrastive_pair") or {}).get("enabled_dimensions") or []
    )
    pair_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"original": 0, "baseline": 0})
    )
    for row in manifest_rows:
        if row["dimension"] not in enabled_dims:
            continue
        pid = row.get("contrastive_pair_id")
        if not pid:
            continue
        role = str(row.get("contrastive_role") or "")
        if role == "original":
            pair_counts[row["dimension"]][pid]["original"] += 1
        elif role.startswith("baseline_"):
            pair_counts[row["dimension"]][pid]["baseline"] += 1
    pair_rows = []
    for dim in sorted(enabled_dims):
        pairs = pair_counts.get(dim, {})
        complete_pairs = sum(
            1 for v in pairs.values() if v["original"] >= 1 and v["baseline"] >= 1
        )
        missing_orig = sum(1 for v in pairs.values() if v["original"] == 0)
        missing_base = sum(1 for v in pairs.values() if v["baseline"] == 0)
        pair_rows.append([dim, len(pairs), complete_pairs, missing_orig, missing_base])
    pair_table = _format_table(
        ["dimension", "pairs_seen", "complete(>=1O+>=1B)", "missing_original", "missing_baseline"],
        pair_rows,
    )

    # 5) multi-image quality histogram——读 samples/<dim>.jsonl 的 _audit.multi_reference_quality
    crop_hist: Counter = Counter()
    id_hist: Counter = Counter()
    for row in _samples_iter(paths):
        audit_block = row.get("_audit") or row.get("audit") or {}
        mq = (
            audit_block.get("multi_reference_quality")
            if isinstance(audit_block, dict)
            else None
        )
        if not mq:
            continue
        crop_hist[mq.get("crop_leakage_risk", "unknown")] += 1
        id_hist[mq.get("identity_visibility", "unknown")] += 1
    quality_lines = []
    quality_lines.append(
        "**crop_leakage_risk**: " + ", ".join(f"{k}={v}" for k, v in crop_hist.items())
    )
    quality_lines.append(
        "**identity_visibility**: " + ", ".join(f"{k}={v}" for k, v in id_hist.items())
    )
    quality_block = "\n".join(quality_lines)

    # 6) QC histogram (peek qc_reports)
    qc_hist: Counter = Counter()
    if paths["qc_reports"].exists():
        for p in paths["qc_reports"].glob("*.json"):
            try:
                data = read_json(p)
                qc_hist[data.get("qc_status", "unknown")] += 1
            except Exception:
                continue

    # ---- compose markdown ----
    md = []
    md.append(f"# Benchmark Dataset Card (mode={config.get('mode','pilot')})")
    md.append("")
    md.append(f"- Total samples (per-dim files): **{total_per_dim}**")
    md.append(f"- phase3_manifest.jsonl rows: **{len(manifest_rows)}**")
    md.append(f"- Consistency check: **{'PASS' if consistent else 'FAIL'}**")
    md.append("")
    md.append("## 1. Per-dimension counts")
    md.append("")
    md.append(dim_table)
    md.append("")
    md.append("## 2. Quota vs actual")
    md.append("")
    md.append(quota_table)
    md.append("")
    md.append("## 3. Contrastive pair coverage")
    md.append("")
    md.append(pair_table)
    md.append("")
    md.append("## 4. Multi-image reference quality")
    md.append("")
    md.append(quality_block)
    md.append("")
    md.append("## 5. QC status histogram")
    md.append("")
    md.append(", ".join(f"{k}={v}" for k, v in qc_hist.items()) or "(no qc_reports/)")
    md.append("")

    md_text = "\n".join(md)
    paths["dataset_card"].write_text(md_text, encoding="utf-8")
    logger.info(f"Wrote dataset card -> {paths['dataset_card']}")
    return md_text


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 audit_phase2")
    parser.add_argument("--config", required=False)
    parser.add_argument("--dataset", required=False, help="Path to benchmark_dataset/")
    args = parser.parse_args(argv)

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = {
            "mode": "pilot",
            "output_dir": args.dataset or "data/benchmark_dataset",
            "quota": {},
        }
    audit(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
