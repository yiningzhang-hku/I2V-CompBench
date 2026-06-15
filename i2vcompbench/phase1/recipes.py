"""
Phase 1 Step 7 \u2014 Candidate Recipe \u751f\u6210\u3002

\u8f93\u5165\uff1a
    aligned_instances.jsonl    (Phase 1 Step 4 \u4ea7\u51fa)
    assets.jsonl               (Phase 1 Step 5 \u4ea7\u51fa)
    text_parse_v2.jsonl        (Phase 1 patch \u4ea7\u51fa\uff0c\u542b primary_dimension / forbidden_*)
    manifest_clean.jsonl

\u8f93\u51fa\uff1a
    candidate_recipes.jsonl    (\u4e00\u884c = \u4e00\u4efd CandidateRecipe)

\u751f\u6210\u7b56\u7565 (P0)\uff1a
* \u4ec5\u4e3a feasible \u4e14 tool_status \u2208 {valid, low_confidence} \u7684\u7ef4\u5ea6\u751f\u6210 recipe\u3002
* \u4ee5\u6837\u672c\u7684 primary_dimension \u4e3a\u4e3b\uff0c\u9047\u5230\u6ca1\u8bbe\u4e3b\u7ef4\u5ea6\u7684\u8df3\u8fc7\u3002
* input_mode \u4e0e source_type \u63a8\u5bfc\u89c4\u5219\u89c1\u4e0b\u65b9 _decide_input_mode\u3002
* contrastive_spec \u9ed8\u8ba4 4 \u7c7b baseline\uff1astatic_copy / random_motion / global_filter / camera_pan_cheat\u3002
* preserve_constraints \u6839\u636e\u4e3b\u7ef4\u5ea6\u578b\u522b\u9ed8\u8ba4\u586b\u5145\uff08\u5982 attribute_binding \u9700\u4fdd\u6301 identity \u4e0e camera_framing\uff09\u3002
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from i2vcompbench.utils.io_utils import ensure_dir, read_jsonl, write_jsonl

DIMENSIONS_V2 = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "spatial_composition",
    "background_dynamics",
    "view_transformation",
    "interaction_reasoning",
]

# \u4e3b\u7ef4\u5ea6 \u2192 \u9ed8\u8ba4 preserve_constraints
PRESERVE_DEFAULTS: Dict[str, List[Dict]] = {
    "attribute_binding": [
        {"target": "primary_subject", "aspect": "identity", "note": "\u4fdd\u6301\u4e3b\u4f53\u8eab\u4efd\u4e0d\u53d8"},
        {"target": "scene", "aspect": "scene_context", "note": "\u4fdd\u6301\u573a\u666f\u4e0a\u4e0b\u6587"},
        {"target": "camera", "aspect": "camera_framing", "note": "\u4fdd\u6301\u955c\u5934\u6784\u56fe"},
    ],
    "action_binding": [
        {"target": "primary_subject", "aspect": "identity", "note": "\u4fdd\u6301\u52a8\u4f5c\u53d1\u8d77\u8005\u8eab\u4efd"},
        {"target": "scene", "aspect": "scene_context", "note": "\u4fdd\u6301\u573a\u666f\u4e0a\u4e0b\u6587"},
        {"target": "camera", "aspect": "camera_framing", "note": "\u4fdd\u6301\u955c\u5934\u6784\u56fe"},
    ],
    "motion_binding": [
        {"target": "primary_subject", "aspect": "identity", "note": "\u4fdd\u6301\u8fd0\u52a8\u4e3b\u4f53\u8eab\u4efd"},
        {"target": "scene", "aspect": "scene_context", "note": "\u4fdd\u6301\u573a\u666f\u4e0a\u4e0b\u6587"},
        {"target": "camera", "aspect": "camera_framing", "note": "\u4fdd\u6301\u955c\u5934\u6784\u56fe\uff08\u907f\u514d view_transformation \u6c61\u67d3\uff09"},
    ],
    "spatial_composition": [
        {"target": "primary_subject", "aspect": "identity", "note": "\u4fdd\u6301 A/B \u8eab\u4efd"},
        {"target": "primary_subject", "aspect": "attribute_color", "note": "\u4fdd\u6301\u4e3b\u4f53\u989c\u8272"},
        {"target": "camera", "aspect": "camera_framing", "note": "\u4fdd\u6301\u955c\u5934\u6784\u56fe"},
    ],
    "background_dynamics": [
        {"target": "primary_subject", "aspect": "identity", "note": "\u4fdd\u6301\u524d\u666f\u4e3b\u4f53\u8eab\u4efd"},
        {"target": "camera", "aspect": "camera_framing", "note": "\u4fdd\u6301\u955c\u5934\u6784\u56fe"},
    ],
    "view_transformation": [
        {"target": "scene", "aspect": "scene_context", "note": "\u573a\u666f\u4e3b\u4f53\u4e0d\u53d8"},
        {"target": "primary_subject", "aspect": "spatial_position", "note": "\u4e3b\u4f53\u4f4d\u7f6e\u4e0d\u53d8"},
    ],
    "interaction_reasoning": [
        {"target": "primary_subject", "aspect": "identity", "note": "\u4fdd\u6301 agent / patient \u8eab\u4efd"},
        {"target": "scene", "aspect": "scene_context", "note": "\u4fdd\u6301\u573a\u666f\u4e0a\u4e0b\u6587"},
        {"target": "camera", "aspect": "camera_framing", "note": "\u4fdd\u6301\u955c\u5934\u6784\u56fe"},
    ],
}

CONTRAST_DEFAULT = [
    {"contrast_type": "static_copy", "description": "\u9996\u5e27\u590d\u5236\u4f5c\u4e3a\u9759\u6001 baseline"},
    {"contrast_type": "random_motion", "description": "\u968f\u673a\u8fd0\u52a8 baseline"},
    {"contrast_type": "global_filter", "description": "\u5168\u5c40\u6ee4\u955c baseline\uff08\u4e0d\u4ea7\u751f\u4e3b\u7ef4\u5ea6\u4e0a\u7684\u53d8\u5316\uff09"},
    {"contrast_type": "camera_pan_cheat", "description": "以运镜伪造主体运动 baseline"},
]

# subject_swap 仅在 spatial_composition / interaction_reasoning 维度条件追加（§5.6.7）
CONTRAST_SUBJECT_SWAP = {
    "contrast_type": "subject_swap", "description": "主体互换 baseline（仅多主体维度）"
}
_SUBJECT_SWAP_DIMENSIONS = {"spatial_composition", "interaction_reasoning"}


def _index_assets_by_sample(assets: List[dict]) -> Dict[str, List[dict]]:
    by_sid: Dict[str, List[dict]] = defaultdict(list)
    for a in assets:
        sid = (a.get("provenance") or {}).get("source_sample_id")
        if sid:
            by_sid[sid].append(a)
    return by_sid


def _decide_input_mode_and_source(
    primary_dim: str,
    sample_assets: List[dict],
    aligned: dict,
) -> Optional[Dict]:
    """
    \u6839\u636e\u4e3b\u7ef4\u5ea6\u4e0e\u53ef\u7528\u8d44\u4ea7\u51b3\u5b9a input_mode + source_type\u3002\u8fd4\u56de None \u8868\u793a\u4e0d\u8db3\u4ee5\u751f\u6210\u3002
    \u89c4\u5219 (P0)\uff1a
    * view_transformation / background_dynamics \u2192 single_image + observed_single_image\u3002
    * spatial_composition / interaction_reasoning \u2192 \u4f18\u5148 multi_image (\u6709 \u22652 subject asset \u65f6) +
        derived_multi_reference\uff1b\u5426\u5219 single_image + observed_single_image\u3002
    * \u5176\u4ed6\u7ef4\u5ea6 \u2192 single_image + observed_single_image\u3002
    """
    subj_assets = [a for a in sample_assets if a.get("asset_type") == "subject"]
    aligned_count = len(aligned.get("aligned_subjects") or [])

    if primary_dim in ("view_transformation", "background_dynamics"):
        return {"input_mode": "single_image", "source_type": "observed_single_image",
                "reference_asset_ids": []}

    if primary_dim in ("spatial_composition", "interaction_reasoning"):
        if len(subj_assets) >= 2 and aligned_count >= 2:
            return {
                "input_mode": "multi_image",
                "source_type": "derived_multi_reference",
                "reference_asset_ids": [a["asset_id"] for a in subj_assets[:2]],
            }
        return {"input_mode": "single_image", "source_type": "observed_single_image",
                "reference_asset_ids": []}

    # attribute / action / motion
    if subj_assets:
        return {
            "input_mode": "multi_image" if len(subj_assets) >= 2 else "single_image",
            "source_type": "derived_single_image" if subj_assets else "observed_single_image",
            "reference_asset_ids": [subj_assets[0]["asset_id"]],
        }
    return {"input_mode": "single_image", "source_type": "observed_single_image",
            "reference_asset_ids": []}


def _draft_prompt(primary_dim: str, text_row: dict, manifest: dict) -> str:
    """生成 base_prompt_draft —— P0 以 clean_prompt_text 为起点，仅加上维度 tag。"""
    base = manifest.get("clean_prompt_text") or text_row.get("prompt_text") or ""
    return f"[dim={primary_dim}] {base}".strip()


def _infer_subtype(primary_dim: str, input_mode: str) -> str:
    """基于维度和 input_mode 推导 subtype。

    P0 简单规则：维度名 + input_mode 缩写。
    Phase 2 template 会进一步细化。
    """
    mode_suffix = "single" if input_mode == "single_image" else "multi"
    subtype_map = {
        ("motion_binding", "single_image"): "type_a_absolute_single",
        ("motion_binding", "multi_image"): "type_c_multi_motion",
        ("attribute_binding", "single_image"): "attribute_change_single",
        ("attribute_binding", "multi_image"): "attribute_reference_multi",
        ("action_binding", "single_image"): "action_single",
        ("action_binding", "multi_image"): "action_with_object_multi",
        ("spatial_composition", "single_image"): "spatial_single",
        ("spatial_composition", "multi_image"): "spatial_multi",
        ("background_dynamics", "single_image"): "background_change_single",
        ("background_dynamics", "multi_image"): "subject_into_scene_multi",
        ("view_transformation", "single_image"): "camera_motion_single",
        ("view_transformation", "multi_image"): "orbit_around_subject_multi",
        ("interaction_reasoning", "single_image"): "interaction_single",
        ("interaction_reasoning", "multi_image"): "interaction_multi",
    }
    return subtype_map.get((primary_dim, input_mode), f"{primary_dim}_{mode_suffix}")


def _infer_rarity(text_row: dict) -> str:
    """基于 text_parse 中的先验信息推导 semantic_rarity。

    如果存在 frequency_tier 且为 rare/very_rare，返回 'rare'；否则 'common'。
    """
    tier = text_row.get("frequency_tier") or text_row.get("semantic_rarity") or ""
    if tier.lower() in ("rare", "very_rare"):
        return "rare"
    return "common"


def _collect_quality_flags(sample_assets: List[dict]) -> List[str]:
    """收集资产质量标记。"""
    flags: List[str] = []
    for a in sample_assets:
        q = a.get("quality") or {}
        for f in q.get("quality_flags") or []:
            if f and f not in flags:
                flags.append(f)
    if not sample_assets:
        flags.append("no_assets_available")
    return flags


def _expected_difficulty(feasibility_dict: Dict, primary_dim: str, sample_assets: List[dict]) -> str:
    """§5.6.5 三档难度推导。

    - valid + 均分 quality_score >= 0.85 → easy
    - valid + quality_score < 0.85 → medium
    - low_confidence / tool_uncertain → hard
    """
    info = feasibility_dict.get(primary_dim) or {}
    status = info.get("tool_status", "tool_uncertain")
    if status == "valid":
        # 检查资产平均质量分
        if sample_assets:
            avg_quality = sum(
                (a.get("quality") or {}).get("quality_score", 0.0)
                for a in sample_assets
            ) / len(sample_assets)
        else:
            avg_quality = 0.0
        return "easy" if avg_quality >= 0.85 else "medium"
    return "hard"


def build_recipes(config: dict) -> None:
    output_base = Path(config["paths"]["output_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    aligned_path = output_base / "phase1" / "aligned_instances.jsonl"
    assets_path = output_base / "phase1" / "assets.jsonl"
    text_path = output_base / "text_analysis" / "text_parse_v2.jsonl"
    manifest_path = manifest_dir / "manifest_clean.jsonl"

    aligned_rows = read_jsonl(str(aligned_path))
    text_rows = read_jsonl(str(text_path))
    assets_rows = read_jsonl(str(assets_path)) if assets_path.exists() else []
    manifest_rows = read_jsonl(str(manifest_path))

    if not aligned_rows or not text_rows:
        logger.error("\u7f3a\u5c11 aligned_instances.jsonl \u6216 text_parse_v2.jsonl\uff0c\u65e0\u6cd5\u751f\u6210 recipes\u3002")
        return

    aligned_map = {r["sample_id"]: r for r in aligned_rows}
    text_map = {r["sample_id"]: r for r in text_rows}
    manifest_map = {r["sample_id"]: r for r in manifest_rows}
    assets_by_sid = _index_assets_by_sample(assets_rows)

    out_dir = output_base / "phase1"
    ensure_dir(str(out_dir))
    out_path = str(out_dir / "candidate_recipes.jsonl")

    recipes: List[Dict] = []
    skipped_no_primary = 0
    skipped_infeasible = 0

    for sid, aligned in aligned_map.items():
        text_row = text_map.get(sid)
        manifest = manifest_map.get(sid, {})
        if not text_row:
            continue
        primary = text_row.get("primary_dimension")
        if not primary:
            skipped_no_primary += 1
            continue

        feas_list = aligned.get("evaluator_feasibility") or []
        feas_dict = {f["dimension"]: f for f in feas_list}
        f_info = feas_dict.get(primary)
        if not f_info or not f_info.get("feasible") or f_info.get("tool_status") == "invalid_input":
            skipped_infeasible += 1
            continue

        sample_assets = assets_by_sid.get(sid, [])
        decision = _decide_input_mode_and_source(primary, sample_assets, aligned)
        if decision is None:
            continue

        # 构建 contrastive_spec：默认 4 类 + 条件性 subject_swap
        contrastive = list(CONTRAST_DEFAULT)
        if primary in _SUBJECT_SWAP_DIMENSIONS:
            contrastive.append(CONTRAST_SUBJECT_SWAP)

        # 推导 subtype（基于维度+input_mode 的简单规则）
        subtype = _infer_subtype(primary, decision["input_mode"])

        # 质量标记
        quality_flags = _collect_quality_flags(sample_assets)

        recipe = {
            "recipe_id": f"recipe_{sid}_{primary}",
            "source_sample_id": sid,
            "target_dimension": primary,
            "input_mode": decision["input_mode"],
            "source_type": decision["source_type"],
            "reference_asset_ids": decision["reference_asset_ids"],
            "base_prompt_draft": _draft_prompt(primary, text_row, manifest),
            "preserve_constraints": list(PRESERVE_DEFAULTS.get(primary, [])),
            "contrastive_spec": contrastive,
            "dimension_isolation": {
                "primary_dimension": primary,
                "forbidden_dimensions": text_row.get("forbidden_dimension_leakage") or [],
                "leakage_risk_notes": text_row.get("routing_reason") or "",
            },
            "expected_difficulty": _expected_difficulty(feas_dict, primary, sample_assets),
            "subtype": subtype,
            "semantic_rarity": _infer_rarity(text_row),
            "quality_flags": quality_flags,
            "contrastive_pair_id": None,
            "contrastive_role": "original",
            "notes": "P0_mock_geometry",
        }
        recipes.append(recipe)

    write_jsonl(out_path, recipes)
    logger.info(
        f"recipes \u751f\u6210={len(recipes)} \u8df3\u8fc7\u65e0\u4e3b\u7ef4\u5ea6={skipped_no_primary} "
        f"\u8df3\u8fc7\u4e0d\u53ef\u8bc4\u6d4b={skipped_infeasible} \u2192 {out_path}"
    )
