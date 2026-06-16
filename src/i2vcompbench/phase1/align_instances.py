"""
Phase 1 Step 4 \u2014 \u56fe-\u6587\u5b9e\u4f8b\u5bf9\u9f50\u3002

\u8f93\u5165\uff1a
    image_parse_v2.jsonl   \uff08patch \u540e\u542b bbox / segmentation_quality \u7684\u56fe\u50cf\u89e3\u6790\u7ed3\u679c\uff09
    text_parse_v2.jsonl    \uff08patch \u540e\u542b primary_dimension \u7b49\u5b57\u6bb5\u7684\u6587\u672c\u89e3\u6790\u7ed3\u679c\uff09
\u8f93\u51fa\uff1a
    aligned_instances.jsonl \uff08\u4e00\u884c = \u4e00\u4e2a sample\uff0c\u542b instance \u6620\u5c04 + 7 \u7ef4\u53ef\u8bc4\u6d4b\u6027 + \u9632\u6cc4\u6f0f\uff09

\u8bbe\u8ba1\u8981\u70b9\uff1a
* instance_id \u91c7\u7528 ``inst_{sample_id}_{idx}`` \u683c\u5f0f\uff0c\u6587\u672c\u4fa7\u4e3b\u4f53 \u2192 \u6309\u540d\u79f0\u8f6f\u5339\u914d\u3002
* \u6c47\u603b\u51fa 7 \u7ef4\u53ef\u8bc4\u6d4b\u6027\u8bca\u65ad\uff0c\u542b tool_status\uff08valid / low_confidence / tool_uncertain / invalid_input\uff09\u3002
* dimension_isolation \u4ece text_row.forbidden_dimension_leakage \u7ee7\u627f\uff1bP0 \u4e0d\u72ec\u7acb\u63a8\u5bfc\u3002
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from i2vcompbench.utils.io_utils import ensure_dir, read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# instance \u5bf9\u9f50
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def _collect_text_target_subjects(text_row: dict) -> List[str]:
    targets: List[str] = []
    for slot in text_row.get("attribute_change_slots", []) or []:
        if slot.get("target_subject"):
            targets.append(_normalize_name(slot["target_subject"]))
    for slot in text_row.get("action_slots", []) or []:
        if slot.get("target_subject"):
            targets.append(_normalize_name(slot["target_subject"]))
    for slot in text_row.get("motion_slots", []) or []:
        if slot.get("target_subject"):
            targets.append(_normalize_name(slot["target_subject"]))
    for slot in text_row.get("spatial_relation_slots", []) or []:
        if slot.get("subject_a"):
            targets.append(_normalize_name(slot["subject_a"]))
        if slot.get("subject_b"):
            targets.append(_normalize_name(slot["subject_b"]))
    for slot in text_row.get("interaction_slots", []) or []:
        if slot.get("agent_subject"):
            targets.append(_normalize_name(slot["agent_subject"]))
        if slot.get("patient_subject"):
            targets.append(_normalize_name(slot["patient_subject"]))
    # \u53bb\u91cd\u4fdd\u987a
    seen = set()
    out = []
    for t in targets:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _align_one_sample(image_row: dict, text_row: dict) -> Dict:
    sample_id = image_row.get("sample_id") or text_row.get("sample_id")
    image_subjects = image_row.get("subjects", []) or []
    text_targets = _collect_text_target_subjects(text_row)

    aligned_subjects: List[Dict] = []
    matched_text: List[str] = []
    matched_image: List[str] = []

    for idx, s in enumerate(image_subjects):
        instance_id = s.get("instance_id") or f"inst_{sample_id}_{idx}"
        s_name = _normalize_name(s.get("name"))
        s_desc = _normalize_name(s.get("instance_description"))

        text_hit = None
        for t in text_targets:
            if t and (t == s_name or t in s_desc or s_name in t):
                text_hit = t
                break

        if text_hit:
            matched_text.append(text_hit)
            confidence = 1.0 if text_hit == s_name else 0.7
        else:
            confidence = 0.0

        matched_image.append(s_name)
        aligned_subjects.append({
            "instance_id": instance_id,
            "image_subject_id": s.get("id", f"subj_{idx}"),
            "image_subject_name": s_name,
            "text_subject_name": text_hit,
            "alignment_confidence": confidence,
            "geometry": {
                "instance_id": instance_id,
                "bbox": s.get("bbox"),
                "mask_path": s.get("mask_path"),
                "segmentation_quality": s.get("segmentation_quality", "low"),
                "tracking_feasibility": s.get("tracking_feasibility", "medium"),
            },
        })

    unmatched_text = [t for t in text_targets if t not in matched_text]
    unmatched_image = [s for s in matched_image if s not in matched_text]
    overlap_ratio = (
        len(matched_text) / len(text_targets) if text_targets else 0.0
    )

    feasibility = _diagnose_feasibility(image_row, text_row, aligned_subjects)
    isolation = _build_isolation(text_row)

    return {
        "sample_id": sample_id,
        "aligned_subjects": aligned_subjects,
        "unmatched_text_subjects": unmatched_text,
        "unmatched_image_subjects": unmatched_image,
        "overlap_ratio": round(overlap_ratio, 4),
        "evaluator_feasibility": feasibility,
        "dimension_isolation": isolation,
    }


# ---------------------------------------------------------------------------
# 7 \u7ef4\u53ef\u8bc4\u6d4b\u6027\u8bca\u65ad
# ---------------------------------------------------------------------------

def _diagnose_feasibility(
    image_row: dict, text_row: dict, aligned_subjects: List[Dict]
) -> List[Dict]:
    """
    \u8f93\u51fa 7 \u4e2a\u7ef4\u5ea6\u5404\u81ea\u7684 EvaluatorFeasibility (dict \u5f62\u5f0f)\u3002
    tool_status \u63a8\u5bfc\u89c4\u5219 (P0)\uff1a
        - feasible \u4e14 segmentation_quality \u2208 {high,medium}\uff1avalid
        - feasible \u4f46\u5168\u90e8\u5b9e\u4f8b\u4e3a low \u51e0\u4f55\uff1alow_confidence
        - text_request \u4f46 image \u4e0d\u652f\u6301\uff1ainvalid_input
        - \u5176\u4ed6\uff1atool_uncertain
    """
    bg = image_row.get("background", {}) or {}
    cam = image_row.get("camera_baseline", {}) or {}
    has_animate = any(s.get("is_animate") for s in image_row.get("subjects", []) or [])
    has_multi = bool(image_row.get("has_multiple_subjects"))
    distinguishable = bool(image_row.get("subjects_clearly_distinguishable", True))
    sep_ok = (bg.get("foreground_background_separability") or "").lower() in ("high", "medium")
    rigid_ref = bool(cam.get("has_rigid_reference_structure"))

    seg_qualities = [
        (a.get("geometry") or {}).get("segmentation_quality", "low")
        for a in aligned_subjects
    ]
    any_high_or_medium_seg = any(q in ("high", "medium") for q in seg_qualities)

    rules: Dict[str, Tuple[bool, bool, List[str], List[str]]] = {
        # name: (text_request, image_support, reasons, required_tools)
        "attribute_binding": (
            bool(text_row.get("involves_attribute_change")),
            len(image_row.get("subjects", []) or []) > 0,
            [], ["VLM_attribute_check"],
        ),
        "action_binding": (
            bool(text_row.get("involves_action")),
            has_animate,
            [], ["action_recognizer"],
        ),
        "motion_binding": (
            bool(text_row.get("involves_directed_motion")),
            len(image_row.get("subjects", []) or []) > 0,
            [], ["object_tracker", "optical_flow"],
        ),
        "spatial_composition": (
            bool(text_row.get("involves_spatial_relation_change")),
            has_multi and distinguishable,
            [], ["object_tracker"],
        ),
        "background_dynamics": (
            bool(text_row.get("involves_background_change")),
            sep_ok,
            [], ["VLM_scene_compare"],
        ),
        "view_transformation": (
            bool(text_row.get("involves_camera_movement")),
            rigid_ref,
            [], ["camera_motion_estimator"],
        ),
        "interaction_reasoning": (
            bool(text_row.get("involves_interaction"))
            or bool(text_row.get("interaction_slots")),
            has_multi and distinguishable,
            [], ["object_tracker", "VLM_interaction_check"],
        ),
    }

    out = []
    for dim, (text_req, img_sup, reasons, tools) in rules.items():
        feasible = text_req and img_sup
        if feasible and any_high_or_medium_seg:
            tool_status = "valid"
        elif feasible:
            tool_status = "low_confidence"
            reasons.append("only_mock_geometry_available")
        elif text_req and not img_sup:
            tool_status = "invalid_input"
            reasons.append("image_does_not_support")
        elif not text_req and img_sup:
            tool_status = "tool_uncertain"
            reasons.append("text_does_not_request")
        else:
            tool_status = "invalid_input"
            reasons.append("neither_text_nor_image")

        out.append({
            "dimension": dim,
            "feasible": feasible,
            "tool_status": tool_status,
            "reasons": reasons,
            "required_tools": tools,
        })
    return out


def _build_isolation(text_row: dict) -> Optional[Dict]:
    """构建维度隔离信息（单个 dict，与文档 §8 AlignedSample schema 对齐）。"""
    primary = text_row.get("primary_dimension")
    if not primary:
        return None
    return {
        "primary_dimension": primary,
        "forbidden_dimensions": text_row.get("forbidden_dimension_leakage") or [],
        "leakage_risk_notes": text_row.get("routing_reason") or "",
    }


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def align_instances(config: dict) -> None:
    output_base = Path(config["paths"]["output_dir"])
    image_path = str(output_base / "image_analysis" / "image_parse_v2.jsonl")
    text_path = str(output_base / "text_analysis" / "text_parse_v2.jsonl")
    out_dir = output_base / "phase1"
    ensure_dir(str(out_dir))
    out_path = str(out_dir / "aligned_instances.jsonl")

    image_rows = read_jsonl(image_path)
    text_rows = read_jsonl(text_path)
    if not image_rows or not text_rows:
        logger.error(
            "\u7f3a\u5c11 patch \u540e\u7684 image_parse_v2 / text_parse_v2\uff0c\u8bf7\u5148\u8fd0\u884c\u3010phase1.patch\u3011\u3002"
        )
        return

    image_map = {r.get("sample_id"): r for r in image_rows if r.get("parse_success", True)}
    text_map = {r.get("sample_id"): r for r in text_rows if r.get("parse_success", True)}
    common = sorted(set(image_map) & set(text_map))
    logger.info(
        f"align_instances: image={len(image_map)} text={len(text_map)} common={len(common)}"
    )

    out_rows = [_align_one_sample(image_map[sid], text_map[sid]) for sid in common]
    write_jsonl(out_path, out_rows)
    logger.info(f"aligned_instances \u2192 {out_path}")
