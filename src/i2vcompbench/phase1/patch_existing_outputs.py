"""
\u5728\u4e0d\u91cd\u8dd1 VLM/LLM \u7684\u524d\u63d0\u4e0b\uff0c\u7ed9\u73b0\u6709 image_parse.jsonl / text_parse.jsonl \u8865\u9f50 Phase 1 \u6240\u9700\u65b0\u5b57\u6bb5\u3002

\u8865\u9f50\u7b56\u7565\uff1a
* image_parse.jsonl\uff1a\u9760 mock_geometry \u4ece position_in_frame \u53cd\u63a8 bbox\uff1b\u7528\u542f\u53d1\u5f0f\u751f\u6210
  reference_potential\u3002
* text_parse.jsonl\uff1a\u7528\u542f\u53d1\u5f0f\u4ece involves_* \u5e03\u5c14\u5b57\u6bb5\u63a8\u5bfc primary_dimension /
  candidate_dimensions / forbidden_dimension_leakage\uff1b\u4ece spatial / motion / action \u7684\u5b58\u5728\u63a8\u5bfc
  interaction_slots\uff08\u4ec5\u5f53 prompt \u540c\u65f6\u4f53\u73b0\u591a\u4e3b\u4f53 + \u63a5\u89e6\u4e8b\u4ef6\u65f6\uff09\u3002

\u4f7f\u7528\u65b9\u5f0f\uff1a
    python -m src.phase1.patch_existing_outputs --config configs/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from i2vcompbench.phase1.mock_geometry import (
    DEFAULT_BBOX,
    estimate_tracking_feasibility,
    position_to_bbox,
)
from i2vcompbench.utils.io_utils import load_config, read_jsonl, write_jsonl

# 7 \u7ef4\u5ea6\u96c6\u5408
DIMENSIONS_V2 = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "spatial_composition",
    "background_dynamics",
    "view_transformation",
    "interaction_reasoning",
]

# \u4e0e\u4e3b\u7ef4\u5ea6\u5bb9\u6613\u6df7\u6dc6\u3001\u9700\u8981\u9632\u6cc4\u6f0f\u7684\u7ef4\u5ea6\u8868
LEAKAGE_TABLE: Dict[str, List[str]] = {
    "attribute_binding":      ["motion_binding", "view_transformation"],
    "action_binding":         ["motion_binding", "view_transformation"],
    "motion_binding":         ["view_transformation", "action_binding"],
    "spatial_composition":    ["motion_binding", "view_transformation"],
    "background_dynamics":    ["view_transformation"],
    "view_transformation":    ["motion_binding"],
    "interaction_reasoning":  ["motion_binding", "view_transformation"],
}


# ---------------------------------------------------------------------------
# Image patch
# ---------------------------------------------------------------------------

def _patch_image_subject(subj: dict) -> dict:
    """\u4e3a\u5355\u4e2a image \u4e3b\u4f53\u8865 instance_id / bbox / segmentation_quality / tracking_feasibility\u3002"""
    if not subj.get("instance_id"):
        subj["instance_id"] = subj.get("id") or "subj_unknown"
    if subj.get("bbox") is None:
        subj["bbox"] = list(position_to_bbox(subj.get("position_in_frame")))
    if not subj.get("segmentation_quality"):
        subj["segmentation_quality"] = "low"  # P0 mock
    if not subj.get("tracking_feasibility"):
        bb = tuple(subj["bbox"]) if subj.get("bbox") else DEFAULT_BBOX
        subj["tracking_feasibility"] = estimate_tracking_feasibility(
            bb, bool(subj.get("is_animate", False))
        )
    return subj


def _infer_reference_potential(image_row: dict) -> dict:
    """
    \u542f\u53d1\u5f0f\u63a8\u5bfc reference_potential\uff1a
      - subject\uff1a\u6709\u53ef\u8bc6\u522b\u4e3b\u4f53 \u4e14 separability != low \u2192 True
      - attribute\uff1a\u4e3b\u4f53\u5c5e\u6027\u975e\u7a7a \u2192 True
      - object\uff1a\u4e3b\u4f53\u4e2d\u5b58\u5728 is_animate=False \u7684\u53ef\u62ff\u53d6\u7269\u4f53 \u2192 True
      - scene_reference_original\uff1a\u521a\u6027\u5e8c\u666f\u5360\u6bd4 high/medium \u2192 True
      - scene_reference_inpainted\uff1aP0 \u9636\u6bb5\u4e00\u5f8b False\uff08\u4e0d\u505a inpainting\uff09
    """
    subjects = image_row.get("subjects", [])
    bg = image_row.get("background", {}) or {}
    sep = (bg.get("foreground_background_separability") or "").lower()
    rigid_ratio = (bg.get("rigid_background_ratio") or "").lower()

    has_subject = bool(subjects) and sep in ("high", "medium")
    has_attr = any(
        (s.get("attributes") or {})
        and any(
            (s["attributes"].get(k) or [])
            for k in ("color", "material_texture", "state", "wearing")
        )
        for s in subjects
    )
    has_object = any(not s.get("is_animate", False) for s in subjects)
    scene_orig = rigid_ratio in ("high", "medium")

    reasons = []
    if has_subject:
        reasons.append("subject_extractable")
    if scene_orig:
        reasons.append("rigid_background_present")

    return {
        "subject": has_subject,
        "attribute": has_attr,
        "object": has_object,
        "scene_reference_original": scene_orig,
        "scene_reference_inpainted": False,  # P0 \u4e0d\u505a inpainting
        "reason": "; ".join(reasons) if reasons else "no_extractable_assets",
    }


def patch_image_parse(image_path: str, output_path: str) -> Tuple[int, int]:
    """\u5bf9 image_parse.jsonl \u9010\u884c\u8865\u9f50\u3002\u8fd4\u56de (total, patched)\u3002"""
    rows = read_jsonl(image_path)
    patched = 0
    for row in rows:
        if not row.get("parse_success", True):
            continue
        for s in row.get("subjects", []):
            _patch_image_subject(s)
        if not row.get("reference_potential"):
            row["reference_potential"] = _infer_reference_potential(row)
        patched += 1
    write_jsonl(output_path, rows)
    return len(rows), patched


# ---------------------------------------------------------------------------
# Text patch
# ---------------------------------------------------------------------------

def _infer_dimension_routing(text_row: dict) -> Tuple[Optional[str], List[str], List[str]]:
    """
    \u4ece\u73b0\u6709 involves_* \u5e03\u5c14\u63a8\u5bfc\u4e3b\u7ef4\u5ea6\u4e0e\u5019\u9009\u7ef4\u5ea6\u3002
    \u4f18\u5148\u7ea7\uff08\u4e3b\u7ef4\u5ea6\u53d6\u9996\u4e2a\u547d\u4e2d\uff09\uff1a
        interaction (\u4e3b\u4f53>=2 \u4e14\u591a\u4e2a\u4e3b\u4f53\u52a8\u4f5c)
        > attribute_binding > action_binding > motion_binding
        > spatial_composition > background_dynamics > view_transformation
    """
    candidates: List[str] = []

    if text_row.get("involves_attribute_change"):
        candidates.append("attribute_binding")
    if text_row.get("involves_action"):
        candidates.append("action_binding")
    if text_row.get("involves_directed_motion"):
        candidates.append("motion_binding")
    if text_row.get("involves_spatial_relation_change"):
        candidates.append("spatial_composition")
    if text_row.get("involves_background_change"):
        candidates.append("background_dynamics")
    if text_row.get("involves_camera_movement"):
        candidates.append("view_transformation")

    # interaction \u63a8\u5bfc\uff1aaction_slots \u4e2d\u51fa\u73b0 >=2 \u4e2a\u4e0d\u540c target_subject \u4e14
    # spatial_relation_slots \u975e\u7a7a\uff0c\u8868\u660e\u5b58\u5728\u591a\u4e3b\u4f53\u52a8\u4f5c\u4e0e\u63a5\u89e6\u4e8b\u4ef6\u3002
    actions = text_row.get("action_slots") or []
    spatials = text_row.get("spatial_relation_slots") or []
    distinct_actors = {a.get("target_subject", "").lower() for a in actions if a.get("target_subject")}
    if len(distinct_actors) >= 2 and spatials:
        candidates.insert(0, "interaction_reasoning")

    primary = candidates[0] if candidates else None
    forbidden = LEAKAGE_TABLE.get(primary, []) if primary else []
    # \u4e0d\u8981\u628a\u672c\u8eab\u5217\u8fdb forbidden
    forbidden = [d for d in forbidden if d != primary]
    # \u53bb\u91cd\u4f46\u4fdd\u6301\u987a\u5e8f
    seen = set()
    candidates_unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            candidates_unique.append(c)
    return primary, candidates_unique, forbidden


def _infer_interaction_slots(text_row: dict) -> List[dict]:
    """\u4ece spatial_relation_slots + action_slots \u63a8\u5bfc\u4e00\u6761 contact_event \u578b interaction\u3002"""
    spatials = text_row.get("spatial_relation_slots") or []
    actions = text_row.get("action_slots") or []
    if not spatials or not actions:
        return []
    # \u53d6\u7b2c\u4e00\u4e2a\u7a7a\u95f4\u5173\u7cfb\u4f5c\u4e3a agent/patient
    sp = spatials[0]
    a, b = sp.get("subject_a", ""), sp.get("subject_b", "")
    if not a or not b or a == b:
        return []
    return [{
        "interaction_type": "contact_event",
        "agent_subject": a,
        "patient_subject": b,
        "relation_phrase": sp.get("target_predicate", ""),
        "expected_outcome": "",
    }]


def _patch_text_row(text_row: dict) -> bool:
    """\u4e3a\u5355\u884c text_parse \u8865\u9f50 Phase 1 \u65b0\u5b57\u6bb5\u3002\u8fd4\u56de\u662f\u5426\u53d1\u751f\u4fee\u6539\u3002"""
    if not text_row.get("parse_success", True):
        return False
    changed = False

    if not text_row.get("primary_dimension"):
        primary, cands, forb = _infer_dimension_routing(text_row)
        text_row["primary_dimension"] = primary
        text_row["candidate_dimensions"] = cands
        text_row["forbidden_dimension_leakage"] = forb
        text_row.setdefault("transform_ontology_confidence", 0.6 if primary else 0.0)
        text_row.setdefault("routing_reason", "heuristic_from_involves_flags")
        changed = True

    if "involves_interaction" not in text_row:
        slots = _infer_interaction_slots(text_row)
        text_row["involves_interaction"] = bool(slots)
        text_row["interaction_slots"] = slots
        # \u82e5\u88c5\u4e0a\u4e86 interaction_slots \u4e14 primary \u672a\u8bbe\uff0c\u63d0\u5347\u4e3a interaction_reasoning
        if slots and not text_row.get("primary_dimension"):
            text_row["primary_dimension"] = "interaction_reasoning"
            text_row["candidate_dimensions"] = (
                text_row.get("candidate_dimensions") or []
            ) + ["interaction_reasoning"]
            text_row["forbidden_dimension_leakage"] = LEAKAGE_TABLE["interaction_reasoning"]
        changed = True

    return changed


def patch_text_parse(text_path: str, output_path: str) -> Tuple[int, int]:
    rows = read_jsonl(text_path)
    patched = 0
    for row in rows:
        if _patch_text_row(row):
            patched += 1
    write_jsonl(output_path, rows)
    return len(rows), patched


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def patch_outputs(config: dict) -> None:
    output_base = Path(config["paths"]["output_dir"])
    image_in = str(output_base / "image_analysis" / "image_parse.jsonl")
    text_in = str(output_base / "text_analysis" / "text_parse.jsonl")
    image_out = str(output_base / "image_analysis" / "image_parse_v2.jsonl")
    text_out = str(output_base / "text_analysis" / "text_parse_v2.jsonl")

    if Path(image_in).exists():
        total, patched = patch_image_parse(image_in, image_out)
        logger.info(f"image_parse \u8865\u9f50: total={total} patched={patched} \u2192 {image_out}")
    else:
        logger.warning(f"image_parse \u4e0d\u5b58\u5728\uff0c\u8df3\u8fc7\uff1a{image_in}")

    if Path(text_in).exists():
        total, patched = patch_text_parse(text_in, text_out)
        logger.info(f"text_parse \u8865\u9f50: total={total} patched={patched} \u2192 {text_out}")
    else:
        logger.warning(f"text_parse \u4e0d\u5b58\u5728\uff0c\u8df3\u8fc7\uff1a{text_in}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    patch_outputs(load_config(args.config))
