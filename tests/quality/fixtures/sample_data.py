"""Factory functions for generating minimal test data."""

from __future__ import annotations

from typing import Optional


def make_sample_candidate(
    question_id: str = "attr_single_0001",
    dimension: str = "attribute_binding",
    difficulty: str = "easy",
    semantic_rarity: str = "common",
    prompt: str = "The cartoon elephant gently lifts its trunk upward.",
    noun: Optional[str] = "elephant",
    target_relation_value: str = "lifts trunk upward",
    has_preservation: bool = True,
) -> dict:
    """生成一条符合 phase3_manifest 格式的候选。"""
    target_subjects = []
    if noun is not None:
        target_subjects.append({
            "id": "s1",
            "description": f"a {noun} character",
            "noun": noun,
        })

    target_relation = {
        "type": _dimension_to_relation_type(dimension),
        "value": target_relation_value,
        "subj": "s1",
        "obj": None,
    }

    preservation_set = []
    if has_preservation:
        preservation_set = [
            {"scope": "background", "constraint": "stable"},
            {"scope": "target_identity", "constraint": "preserve"},
        ]

    return {
        "question_id": question_id,
        "dimension": dimension,
        "input_mode": "single_image",
        "first_frame_path": f"data/benchmark_dataset/first_frames/{question_id}.png",
        "input_image_paths": [],
        "prompt": prompt,
        "target_subjects": target_subjects,
        "target_relation": target_relation,
        "preservation_set": preservation_set,
        "contrastive_pair_id": None,
        "contrastive_role": "original",
        "evaluator_tools": ["grounding", "vlm_attribute"],
        "expected_failure_modes": ["wrong_attribute"],
        "subtype": "",
        "difficulty": difficulty,
        "semantic_rarity": semantic_rarity,
        "source_type": "tip_i2v_synthetic_first_frame",
        "source_sample_id": f"source_{question_id}",
    }


def make_fixture_candidates() -> list[dict]:
    """生成10条fixture候选（5维×2条），覆盖不同difficulty和rarity。"""
    configs = [
        # attribute_binding: 2 条
        ("attr_single_0001", "attribute_binding", "easy", "common",
         "The cartoon elephant gently lifts its trunk upward.", "elephant", "lifts trunk upward"),
        ("attr_single_0002", "attribute_binding", "hard", "rare",
         "The crystalline fox gradually changes its fur color to emerald green.", "fox", "changes fur color to emerald green"),
        # action_binding: 2 条
        ("act_single_0001", "action_binding", "easy", "common",
         "The athlete sprints across the finish line with determination.", "athlete", "sprints across the finish line"),
        ("act_single_0002", "action_binding", "medium", "rare",
         "The mechanical owl flutters its wings and ascends into twilight.", "owl", "flutters wings and ascends"),
        # motion_binding: 2 条
        ("motion_single_0001", "motion_binding", "medium", "common",
         "The red sports car drifts sharply around the tight corner.", "car", "drifts around corner"),
        ("motion_single_0002", "motion_binding", "hard", "rare",
         "The origami crane unfolds its paper wings and glides forward.", "crane", "unfolds wings and glides"),
        # background_dynamics: 2 条
        ("bg_single_0001", "background_dynamics", "easy", "common",
         "The clouds behind the mountains gradually shift from white to orange.", "clouds", "shift from white to orange"),
        ("bg_single_0002", "background_dynamics", "medium", "rare",
         "The aurora borealis intensifies above the frozen tundra landscape.", "aurora", "intensifies above tundra"),
        # view_transformation: 2 条
        ("view_single_0001", "view_transformation", "easy", "common",
         "The camera pans slowly to the right revealing the distant valley.", "valley", "camera pans right"),
        ("view_single_0002", "view_transformation", "hard", "rare",
         "The camera zooms in on the intricate clockwork mechanism steadily.", "mechanism", "camera zooms in"),
    ]

    candidates = []
    for qid, dim, diff, rarity, prompt, noun, relation_val in configs:
        candidates.append(make_sample_candidate(
            question_id=qid,
            dimension=dim,
            difficulty=diff,
            semantic_rarity=rarity,
            prompt=prompt,
            noun=noun,
            target_relation_value=relation_val,
        ))
    return candidates


def make_input_asset_manifest_row(
    question_id: str,
    source_ref_id: str = "test-source-001",
) -> dict:
    """生成一条 input_assets_manifest 格式的记录。"""
    return {
        "question_id": question_id,
        "assets": [
            {
                "asset_id": f"{question_id}_ff",
                "role": "first_frame",
                "path": f"data/benchmark_dataset/first_frames/{question_id}.png",
                "source_type": "tip_derived_reference",
                "source_ref_id": source_ref_id,
                "ref_image_idx": None,
                "quality": {
                    "identity_visibility": "high",
                    "crop_leakage_risk": "low",
                    "resolution_ok": True,
                    "notes": "",
                },
            }
        ],
    }


def _dimension_to_relation_type(dimension: str) -> str:
    """Map dimension to target_relation.type."""
    mapping = {
        "attribute_binding": "attribute",
        "action_binding": "action",
        "motion_binding": "motion",
        "background_dynamics": "background",
        "view_transformation": "view",
    }
    return mapping.get(dimension, "attribute")
