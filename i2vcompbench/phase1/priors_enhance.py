"""
Phase 1 Step 6 \u2014 \u5728\u73b0\u6709 dimension_priors / global_distributions \u4e0a\u589e\u5f3a\u4ea7\u51fa\uff1a
1. frequency_tiers.json   \u2014\u2014 \u4e3b\u4f53/\u52a8\u4f5c/\u5c5e\u6027\u7684 head/torso/long_tail \u4e09\u6863
2. subject_pair_distribution.json \u2014\u2014 \u591a\u4e3b\u4f53\u5171\u73b0\u9891\u8c31
3. multi_reference_priors.json \u2014\u2014 \u591a\u56fe\u8d44\u4ea7\u7684\u5fc3\u8df3\u9891\u7387\u4e0e\u53ef\u7528\u6027
4. compatibility_matrix.json \u2014\u2014 7\u00d77 \u7ef4\u5ea6\u517c\u5bb9\u77e9\u9635\uff08\u542f\u53d1\u5f0f\uff09

\u8f93\u5165\uff1a
    aligned_instances.jsonl
    text_parse_v2.jsonl
    \uff08\u53ef\u9009\uff09 dimension_priors.jsonl  \u2014 \u4ec5\u7528\u4e8e\u51b3\u5b9a tier \u9608\u503c
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List

from loguru import logger

from i2vcompbench.utils.io_utils import ensure_dir, read_jsonl

# 7 \u7ef4\u5ea6
DIMENSIONS_V2 = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "spatial_composition",
    "background_dynamics",
    "view_transformation",
    "interaction_reasoning",
]


# ---------------------------------------------------------------------------
# 1. frequency tiers
# ---------------------------------------------------------------------------

def _split_tiers(counter: Counter, head_pct: float = 0.5, torso_pct: float = 0.85) -> Dict[str, List[Dict]]:
    """\u6309\u7d2f\u8ba1\u9891\u7387\u5212\u5206 head/torso/long_tail \u4e09\u6863\u3002"""
    items = counter.most_common()
    total = sum(counter.values()) or 1
    cum = 0
    tiers = {"head": [], "torso": [], "long_tail": []}
    for name, cnt in items:
        cum += cnt
        ratio = cum / total
        entry = {"name": name, "count": cnt, "pct": round(cnt / total * 100, 2)}
        if ratio <= head_pct:
            tiers["head"].append(entry)
        elif ratio <= torso_pct:
            tiers["torso"].append(entry)
        else:
            tiers["long_tail"].append(entry)
    return tiers


def build_frequency_tiers(text_rows: List[dict]) -> Dict:
    subj_ctr = Counter()
    verb_ctr = Counter()
    attr_to_value_ctr = defaultdict(Counter)

    for t in text_rows:
        for slot in t.get("attribute_change_slots", []) or []:
            subj_ctr[(slot.get("target_subject") or "").lower()] += 1
            attr_type = (slot.get("attribute_type") or "").lower()
            to_val = (slot.get("to_value") or "").lower()
            if attr_type and to_val:
                attr_to_value_ctr[attr_type][to_val] += 1
        for slot in t.get("action_slots", []) or []:
            subj_ctr[(slot.get("target_subject") or "").lower()] += 1
            verb_ctr[(slot.get("action_verb") or "").lower()] += 1

    return {
        "subjects": _split_tiers(subj_ctr),
        "action_verbs": _split_tiers(verb_ctr),
        "attribute_values": {
            attr: _split_tiers(c) for attr, c in attr_to_value_ctr.items()
        },
    }


# ---------------------------------------------------------------------------
# 2. subject pair distribution
# ---------------------------------------------------------------------------

def build_subject_pair_distribution(aligned_rows: List[dict]) -> Dict:
    pair_ctr = Counter()
    for row in aligned_rows:
        names = sorted({
            (a.get("image_subject_name") or "").lower()
            for a in row.get("aligned_subjects", [])
            if a.get("image_subject_name")
        })
        for a, b in combinations(names, 2):
            pair_ctr[(a, b)] += 1

    total = sum(pair_ctr.values()) or 1
    pairs = [
        {"pair": [a, b], "count": cnt, "pct": round(cnt / total * 100, 2)}
        for (a, b), cnt in pair_ctr.most_common(200)
    ]
    return {"total_unique_pairs": len(pair_ctr), "top_pairs": pairs}


# ---------------------------------------------------------------------------
# 3. multi-reference priors
# ---------------------------------------------------------------------------

def build_multi_reference_priors(aligned_rows: List[dict], assets_rows: List[dict]) -> Dict:
    """\u7edf\u8ba1\u53ef\u8d70 multi_image \u8def\u7ebf\u7684\u6837\u672c\u5360\u6bd4\u4e0e asset \u8986\u76d6\u3002"""
    multi_eligible = 0
    total = len(aligned_rows)
    for row in aligned_rows:
        if len(row.get("aligned_subjects", [])) >= 2:
            multi_eligible += 1

    asset_per_sample = Counter()
    asset_type_ctr = Counter()
    for a in assets_rows:
        sid = (a.get("provenance") or {}).get("source_sample_id")
        if sid:
            asset_per_sample[sid] += 1
        asset_type_ctr[a.get("asset_type", "")] += 1

    return {
        "total_samples": total,
        "multi_eligible_samples": multi_eligible,
        "multi_eligible_ratio": round(multi_eligible / total, 4) if total else 0.0,
        "asset_type_distribution": dict(asset_type_ctr),
        "avg_assets_per_sample": (
            round(sum(asset_per_sample.values()) / len(asset_per_sample), 2)
            if asset_per_sample else 0.0
        ),
        "samples_with_any_asset": len(asset_per_sample),
    }


# ---------------------------------------------------------------------------
# 4. compatibility matrix
# ---------------------------------------------------------------------------

# \u542f\u53d1\u5f0f\u517c\u5bb9\u8868\uff1a"compatible" / "leakage_risk" / "incompatible"
# \u5bf9\u89d2\u7ebf\u4e3a self\uff0c\u8bbe\u4e3a "self"
COMPAT_MATRIX = {
    "attribute_binding": {
        "action_binding": "leakage_risk",
        "motion_binding": "incompatible",
        "spatial_composition": "compatible",
        "background_dynamics": "compatible",
        "view_transformation": "incompatible",
        "interaction_reasoning": "leakage_risk",
    },
    "action_binding": {
        "motion_binding": "leakage_risk",
        "spatial_composition": "compatible",
        "background_dynamics": "compatible",
        "view_transformation": "incompatible",
        "interaction_reasoning": "compatible",
    },
    "motion_binding": {
        "spatial_composition": "leakage_risk",
        "background_dynamics": "compatible",
        "view_transformation": "incompatible",
        "interaction_reasoning": "compatible",
    },
    "spatial_composition": {
        "background_dynamics": "compatible",
        "view_transformation": "leakage_risk",
        "interaction_reasoning": "compatible",
    },
    "background_dynamics": {
        "view_transformation": "leakage_risk",
        "interaction_reasoning": "compatible",
    },
    "view_transformation": {
        "interaction_reasoning": "incompatible",
    },
}


def build_compatibility_matrix() -> Dict:
    matrix: Dict[str, Dict[str, str]] = {d: {} for d in DIMENSIONS_V2}
    for i, d1 in enumerate(DIMENSIONS_V2):
        for d2 in DIMENSIONS_V2:
            if d1 == d2:
                matrix[d1][d2] = "self"
            else:
                rel = COMPAT_MATRIX.get(d1, {}).get(d2) or COMPAT_MATRIX.get(d2, {}).get(d1)
                matrix[d1][d2] = rel or "compatible"
    return {
        "dimensions": DIMENSIONS_V2,
        "matrix": matrix,
        "legend": {
            "self": "\u540c\u4e00\u7ef4\u5ea6",
            "compatible": "\u53ef\u540c\u65f6\u51fa\u73b0",
            "leakage_risk": "\u5bb9\u6613\u6df7\u6dc6\u4e3b\u7ef4\u5ea6\uff0c\u9700\u51fa\u73b0\u5728 forbidden_dimension_leakage",
            "incompatible": "\u4e0d\u5141\u8bb8\u540c\u65f6\u4e3a\u4e3b\u7ef4\u5ea6 + \u9644\u52a0\u7ef4\u5ea6",
        },
    }


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def enhance_priors(config: dict) -> None:
    output_base = Path(config["paths"]["output_dir"])
    aligned_path = output_base / "phase1" / "aligned_instances.jsonl"
    assets_path = output_base / "phase1" / "assets.jsonl"
    text_path = output_base / "text_analysis" / "text_parse_v2.jsonl"
    out_dir = output_base / "phase1" / "priors"
    ensure_dir(str(out_dir))

    aligned_rows = read_jsonl(str(aligned_path)) if aligned_path.exists() else []
    text_rows = read_jsonl(str(text_path)) if text_path.exists() else []
    assets_rows = read_jsonl(str(assets_path)) if assets_path.exists() else []

    if not text_rows:
        logger.warning("\u7f3a\u5c11 text_parse_v2.jsonl\uff0c\u90e8\u5206\u8f93\u51fa\u4f1a\u4e3a\u7a7a\u3002")

    freq_tiers = build_frequency_tiers(text_rows)
    pair_dist = build_subject_pair_distribution(aligned_rows)
    multi_ref = build_multi_reference_priors(aligned_rows, assets_rows)
    compat = build_compatibility_matrix()

    for name, payload in [
        ("frequency_tiers.json", freq_tiers),
        ("subject_pair_distribution.json", pair_dist),
        ("multi_reference_priors.json", multi_ref),
        ("compatibility_matrix.json", compat),
    ]:
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"\u5199\u5165 {path}")
