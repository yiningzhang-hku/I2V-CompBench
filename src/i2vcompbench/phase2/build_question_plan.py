"""
Phase 2 · Step 3: build QuestionPlan from sampled recipes.

For each SampledRecipe we:
  1. Locate its dimension template (configs/templates/<dimension>.yaml)
  2. Resolve a subtype (explicit -> input_mode-matching -> first)
  3. Merge slots from Phase 1 text_parse + image_parse + recipe metadata
  4. Render prompt_draft + evaluator E/P/C patterns
  5. Build input_plan / target_plan / preserve_plan / dimension_isolation
  6. Allocate a question_id (deterministic across runs in pilot)

Output: data/benchmark_dataset/question_plans.jsonl  (List[QuestionPlan])
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from loguru import logger

from ..schemas.phase2 import (
    DIM_SHORT,
    DimensionIsolationPlan,
    InputPlan,
    PreserveItem,
    QuestionPlan,
    RequiredImageSpec,
    SubjectRef,
    TargetPlan,
    TargetRelation,
    default_failure_modes_for,
    required_tools_for,
)
from ..utils.ids import next_question_id, reset_counters
from ..utils.io import (
    Phase1Bundle,
    benchmark_paths,
    iter_jsonl,
    load_config,
    write_jsonl,
)
from ..utils.templates import (
    DimensionTemplate,
    TemplateRegistry,
    count_words,
    render_template,
)


# ============================================================
# Slot resolution
# ============================================================

def _slots_from_text_parse(
    text_parse_row: Dict[str, Any],
    dimension: str,
) -> Dict[str, Any]:
    """Extract dimension-specific slots from Phase 1 text_parse row, with safe fallbacks."""
    slots_out: Dict[str, Any] = {}
    if not text_parse_row:
        return slots_out
    # multiple naming schemes from different Phase 1 versions
    slots_block = (
        text_parse_row.get("slots")
        or text_parse_row.get("dimension_slots")
        or {}
    )
    if isinstance(slots_block, dict):
        # if keyed by dimension, pick the relevant block
        if dimension in slots_block and isinstance(slots_block[dimension], dict):
            slots_out.update(slots_block[dimension])
        else:
            # flat dict -> take all
            for k, v in slots_block.items():
                if isinstance(v, (str, int, float, bool)):
                    slots_out[k] = v

    # Phase 1 的主 schema 将各维度槽位保存在顶层列表中，而不是统一的
    # `slots` 节点。早期 Phase 2 只读取后者，导致主体、动作、属性和运镜
    # 槽位全部回退为空值。这里显式兼容 Phase 1 Step 3 的真实字段。
    slot_field_by_dimension = {
        "attribute_binding": "attribute_change_slots",
        "action_binding": "action_slots",
        "motion_binding": "motion_slots",
        "background_dynamics": "background_change_slots",
        "view_transformation": "camera_movement_slots",
    }
    raw_items = text_parse_row.get(slot_field_by_dimension.get(dimension, "")) or []
    item = raw_items[0] if isinstance(raw_items, list) and raw_items else {}
    if isinstance(item, dict):
        target = item.get("target_subject")
        if target:
            slots_out["target_subject"] = target
            slots_out["target_subject_noun"] = target

        if dimension == "attribute_binding":
            before = item.get("from_value") or item.get("attribute_before") or ""
            after = item.get("to_value") or item.get("attribute_after") or ""
            slots_out.update({
                "attribute_before": before,
                "attribute_after": after,
                "attribute_type": item.get("attribute_type") or "attribute",
                "operation": "gradually changes to",
            })
        elif dimension == "action_binding":
            verb = str(item.get("action_verb") or "").strip()
            detail = str(item.get("action_detail") or "").strip()
            slots_out["action_phrase"] = " ".join(x for x in (verb, detail) if x)
        elif dimension == "motion_binding":
            slots_out["direction"] = item.get("direction") or ""
        elif dimension == "background_dynamics":
            region = item.get("target_region") or "background"
            change = str(item.get("change_type") or "").strip()
            state = str(item.get("to_state") or "").strip()
            slots_out["scene_phrase"] = region
            slots_out["scene_dynamic"] = " ".join(x for x in (change, state) if x)
        elif dimension == "view_transformation":
            slots_out["camera_motion"] = item.get("command") or ""
    return slots_out


def _slots_from_aligned(aligned_row: Dict[str, Any]) -> Dict[str, Any]:
    """Pull target_subject / reference_subject descriptions from aligned_instances."""
    out: Dict[str, Any] = {}
    if not aligned_row:
        return out
    targets = aligned_row.get("target_instances") or []
    refs = aligned_row.get("reference_instances") or []
    # Phase 1 AlignedSample 的正式字段名是 aligned_subjects。
    if not targets:
        targets = aligned_row.get("aligned_subjects") or []
    if targets:
        first = targets[0]
        target_name = (
            first.get("text_subject_name")
            or first.get("image_subject_name")
            or first.get("description")
            or first.get("subject_id")
        )
        if target_name:
            out.setdefault("target_subject", target_name)
            out.setdefault("target_subject_noun", target_name)
        out.setdefault(
            "target_subject_id",
            first.get("instance_id") or first.get("subject_id") or "s1",
        )
    if refs:
        first = refs[0]
        out.setdefault("reference_subject", first.get("description") or first.get("subject_id") or "the reference")
        out.setdefault("reference_subject_id", first.get("subject_id", "s2"))
    return out


def _slots_from_image_parse(image_parse_row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not image_parse_row:
        return out
    bg = image_parse_row.get("background")
    if isinstance(bg, dict):
        out["background"] = bg.get("category") or bg.get("description") or ""
    elif isinstance(bg, str):
        out["background"] = bg
    subjects = image_parse_row.get("subjects") or []
    if isinstance(subjects, list) and subjects:
        first = subjects[0] if isinstance(subjects[0], dict) else {}
        name = first.get("name") or first.get("instance_description")
        if name:
            out.setdefault("target_subject", name)
            out.setdefault("target_subject_noun", name)
    return out


def _resolve_slots(
    sampled: Dict[str, Any],
    bundle: Phase1Bundle,
) -> Dict[str, Any]:
    recipe = sampled["recipe"]
    sid = recipe.get("source_sample_id")
    text_row = bundle.text_parse.get(sid, {}) if sid else {}
    aligned_row = bundle.aligned.get(sid, {}) if sid else {}
    image_row = bundle.image_parse.get(sid, {}) if sid else {}

    slots: Dict[str, Any] = {}
    slots.update(_slots_from_image_parse(image_row))
    slots.update(_slots_from_aligned(aligned_row))
    slots.update(_slots_from_text_parse(text_row, sampled["dimension"]))

    # ensure essential keys exist
    slots.setdefault("target_subject", "")
    slots.setdefault("reference_subject", "the reference subject")
    slots.setdefault("direction", "")
    slots.setdefault("target_relation", "right_of")
    slots.setdefault("start_position", "left")
    slots.setdefault("background", "neutral background")
    slots.setdefault("attribute_change", "")
    slots.setdefault("attribute_before", "")
    slots.setdefault("attribute_after", "")
    slots.setdefault("operation", "gradually changes to")
    slots.setdefault("action_phrase", "")
    slots.setdefault("scene_phrase", slots.get("background", ""))
    slots.setdefault("scene_dynamic", "")
    slots.setdefault("camera_motion", "")
    slots.setdefault("subtype", sampled.get("subtype", ""))
    return slots


# ============================================================
# Plan assembly
# ============================================================

def _resolve_subtype(template: DimensionTemplate, sampled: Dict[str, Any]) -> Dict[str, Any]:
    explicit = sampled.get("subtype") or ""
    return template.find_subtype(explicit or None, sampled.get("input_mode"))


def _build_input_plan(subtype_block: Dict[str, Any]) -> InputPlan:
    items: List[RequiredImageSpec] = []
    for it in subtype_block.get("required_images") or []:
        items.append(
            RequiredImageSpec(
                role=str(it.get("role") or "first_frame"),
                description=str(it.get("description") or ""),
                source_preference=list(
                    it.get("source_preference") or ["tip_derived_reference", "t2i_generated"]
                ),
            )
        )
    if not items:
        items = [RequiredImageSpec(role="first_frame")]
    return InputPlan(required_images=items)


def _build_target_plan(
    subtype_block: Dict[str, Any],
    slots: Dict[str, Any],
    sampled: Dict[str, Any],
) -> TargetPlan:
    op = str(subtype_block.get("operation") or subtype_block.get("id") or "transform")
    expected_pattern = (
        subtype_block.get("expected_final_state_pattern")
        or subtype_block.get("evaluator_E_target_pattern")
        or "{target_subject} performs {operation}"
    )
    expected_state = render_template(expected_pattern, {**slots, "operation": op})

    # §4.3 step 6：target_subjects 稳定 id (s1, s2, ...) + multi_image 的 ref_image_idx
    input_mode = sampled.get("input_mode", "single_image")
    target_subjects: List[SubjectRef] = []
    seen_ids: set[str] = set()

    def _push(desc: str, noun: Optional[str] = None) -> SubjectRef:
        sid = f"s{len(target_subjects) + 1}"
        seen_ids.add(sid)
        ref_idx: Optional[int] = (
            len(target_subjects) if input_mode == "multi_image" else None
        )
        sr = SubjectRef(id=sid, description=desc, noun=noun, ref_image_idx=ref_idx)
        target_subjects.append(sr)
        return sr

    primary_desc = str(slots.get("target_subject") or "")
    primary_noun = slots.get("target_subject_noun")
    _push(primary_desc, str(primary_noun) if primary_noun else None)

    # 多主体场景：spatial / interaction 维度 + multi_image 都可能需要 s2
    secondary_desc = slots.get("reference_subject") or slots.get("object_subject") or ""
    if (
        secondary_desc
        and secondary_desc != primary_desc
        and (
            input_mode == "multi_image"
            or sampled.get("dimension") in {"spatial_composition", "interaction_reasoning"}
        )
    ):
        _push(str(secondary_desc), None)

    # target_relation 作为通用结构化变化字段。五个正式维度同样需要显式
    # value，避免下游只能重新解析自然语言 prompt 才知道要评什么。
    relation_str = str(slots.get("target_relation") or "")
    target_relation: Optional[TargetRelation] = None
    dim = sampled.get("dimension")
    if relation_str and dim in {"spatial_composition", "interaction_reasoning"}:
        subj = "s1" if "s1" in seen_ids else None
        obj = "s2" if "s2" in seen_ids else None
        rel_type = "spatial" if dim == "spatial_composition" else "interaction"
        target_relation = TargetRelation(
            type=rel_type,
            value=relation_str,
            subj=subj,
            obj=obj,
        )
    elif expected_state.strip():
        relation_type_by_dim = {
            "attribute_binding": "attribute",
            "action_binding": "action",
            "motion_binding": "motion",
            "background_dynamics": "background",
            "view_transformation": "view",
        }
        target_relation = TargetRelation(
            type=relation_type_by_dim.get(str(dim), "transform"),
            value=expected_state,
            subj="s1",
            obj=None,
        )

    return TargetPlan(
        target_subjects=target_subjects,
        target_relation=target_relation,
        operation=op,
        attribute_source=slots.get("attribute_source"),
        expected_final_state=expected_state,
    )


def _build_preserve_plan(
    recipe: Dict[str, Any],
    subtype_block: Dict[str, Any],
) -> List[PreserveItem]:
    items: List[PreserveItem] = []
    for pc in recipe.get("preserve_constraints") or []:
        items.append(
            PreserveItem(
                scope=str(pc.get("target") or pc.get("scope") or "background"),
                constraint=str(pc.get("aspect") or pc.get("constraint") or "preserve"),
            )
        )
    # template-level extras (guarantee we have at least background+camera)
    if not any(p.scope == "background" for p in items):
        items.append(PreserveItem(scope="background", constraint="stable"))
    if subtype_block.get("camera_constraint") == "forbidden" and not any(
        p.scope == "camera" for p in items
    ):
        items.append(PreserveItem(scope="camera", constraint="fixed"))
    return items


def _build_evaluator_tools(
    dimension: str,
    input_mode: str,
    subtype_block: Dict[str, Any],
) -> List[str]:
    """§4.3 step 7：强枚举映射覆写。任何模板自由文本 evaluator_tools 都被覆写。"""
    tools = required_tools_for(dimension, input_mode)
    # 仅允许 9 项枚举；multi_image 追加的 dinov2/clip/grounding 已在 helper 里处理
    return tools


def _build_dimension_isolation(
    subtype_block: Dict[str, Any],
) -> DimensionIsolationPlan:
    return DimensionIsolationPlan(
        forbidden_words=list(subtype_block.get("forbidden_words") or []),
        camera_constraint=str(subtype_block.get("camera_constraint") or "forbidden"),  # type: ignore[arg-type]
    )


def _build_expected_failure_modes(
    dimension: str,
    subtype_block: Dict[str, Any],
) -> List[str]:
    """§4.3 step 8：强枚举默认填充；模板可追加但必须 ⊂ 15 项 FAILURE_MODES。"""
    from ..schemas.phase2 import FAILURE_MODES

    base = default_failure_modes_for(dimension)
    extras = list(subtype_block.get("expected_failure_modes") or [])
    out: List[str] = list(base)
    for x in extras:
        if x in FAILURE_MODES and x not in out:
            out.append(x)
    return out


def _build_question_plan(
    sampled: Dict[str, Any],
    template: DimensionTemplate,
    slots: Dict[str, Any],
    question_id: str,
) -> QuestionPlan:
    subtype_block = _resolve_subtype(template, sampled)
    input_plan = _build_input_plan(subtype_block)
    target_plan = _build_target_plan(subtype_block, slots, sampled)
    preserve_plan = _build_preserve_plan(sampled["recipe"], subtype_block)
    evaluator_tools = _build_evaluator_tools(
        sampled["dimension"], sampled["input_mode"], subtype_block
    )
    expected_failure_modes = _build_expected_failure_modes(
        sampled["dimension"], subtype_block
    )
    dim_iso = _build_dimension_isolation(subtype_block)

    prompt_pattern = str(subtype_block.get("prompt_pattern") or "")
    prompt_draft = render_template(prompt_pattern, slots) or str(
        sampled["recipe"].get("base_prompt_draft") or ""
    )

    return QuestionPlan(
        question_id=question_id,
        recipe_id=str(sampled["recipe"].get("recipe_id") or ""),
        dimension=sampled["dimension"],
        input_mode=sampled["input_mode"],
        subtype=str(subtype_block.get("id") or sampled.get("subtype") or ""),
        difficulty=sampled["difficulty"],
        semantic_rarity=sampled.get("semantic_rarity", "common"),
        contrastive_pair_id=sampled.get("contrastive_pair_id"),
        contrastive_role=sampled.get("contrastive_role", "original"),
        input_plan=input_plan,
        target_plan=target_plan,
        preserve_plan=preserve_plan,
        dimension_isolation=dim_iso,
        evaluator_tools=evaluator_tools,
        expected_failure_modes=expected_failure_modes,
        prompt_draft=prompt_draft,
    )


# ============================================================
# Driver
# ============================================================

def build_question_plans(config: Dict[str, Any]) -> List[QuestionPlan]:
    paths = benchmark_paths(config["output_dir"])
    sampled_path = paths["sampled_recipes"]
    if not sampled_path.exists():
        raise FileNotFoundError(
            f"sampled_recipes.jsonl not found at {sampled_path}; run sample_recipes first"
        )

    bundle = Phase1Bundle(config["phase1_bundle_dir"])
    registry = TemplateRegistry()
    reset_counters()

    plans: List[QuestionPlan] = []
    for row in iter_jsonl(sampled_path):
        dim = row["dimension"]
        template = registry.get(dim)
        slots = _resolve_slots(row, bundle)
        dim_short = DIM_SHORT.get(dim, dim[:5])
        mode_short = "single" if row["input_mode"] == "single_image" else "multi"
        qid = next_question_id(dim_short, mode_short)
        plan = _build_question_plan(row, template, slots, qid)
        plans.append(plan)

    logger.info(f"Built {len(plans)} question plans")
    return plans


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 build_question_plan")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = benchmark_paths(cfg["output_dir"])
    plans = build_question_plans(cfg)
    write_jsonl(paths["question_plans"], [p.model_dump() for p in plans])
    logger.info(f"Wrote {len(plans)} question plans -> {paths['question_plans']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
