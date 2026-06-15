"""
Step 4: Joint Analysis + Deep Prior Extraction.

Combines text and image analysis results to:
1. Determine per-dimension evaluability for each sample (retained from v1).
2. Extract per-dimension concept distributions, visual composition priors,
   structural prompt templates, and seed examples.
3. Integrate Pika camera/motion parameters into camera dimension priors.
"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from i2vcompbench.utils.io_utils import (
    ensure_dir,
    read_jsonl,
    write_csv,
    write_freq_csv,
    write_jsonl,
)
from i2vcompbench.schemas.phase1_legacy import (
    ConceptDistribution,
    DimensionEvaluability,
    DimensionPrior,
    ImageAnalysisResult,
    JointAnalysisResult,
    ManifestItem,
    SeedExample,
    TextAnalysisResult,
    VisualCompositionPrior,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "attribute_binding",
    "motion_binding",
    "spatial_relation",
    "action_binding",
    "scene_dynamics",
    "camera_transformation",
]

DIM_DISPLAY_NAMES = {
    "attribute_binding": "Subject Attribute Binding",
    "motion_binding": "Subject Motion Binding",
    "spatial_relation": "Subject Spatial Relation Composition",
    "action_binding": "Subject Action Binding",
    "scene_dynamics": "Scene / Background Dynamics",
    "camera_transformation": "Camera / View Transformation",
}

# Per-dimension constraint rules (static knowledge)
DIM_CONSTRAINTS = {
    "attribute_binding": {
        "min_subjects": 1,
        "requires_explicit_change": True,
        "valid_attribute_types": ["color", "texture", "state", "wearing", "size"],
        "description": "Prompt must specify an attribute change on a subject visible in the image.",
    },
    "motion_binding": {
        "min_subjects": 1,
        "requires_absolute_direction": True,
        "valid_directions": [
            "leftward", "rightward", "upward", "downward",
            "toward_camera", "away_from_camera", "forward", "backward",
        ],
        "description": "Prompt must specify absolute-direction motion for a subject present in the image.",
    },
    "spatial_relation": {
        "min_subjects": 2,
        "requires_relation_change": True,
        "valid_predicates": [
            "left_of", "right_of", "above", "below",
            "in_front_of", "behind", "next_to",
        ],
        "description": "Prompt must specify a spatial relation change between two distinguishable subjects.",
    },
    "action_binding": {
        "min_subjects": 1,
        "requires_animate_subject": True,
        "description": "Prompt must assign a semantic action to an animate subject in the image.",
    },
    "scene_dynamics": {
        "requires_background_change": True,
        "requires_separable_fg_bg": True,
        "valid_change_types": ["weather", "lighting", "state", "texture", "phenomenon"],
        "description": "Prompt must specify an environmental change in the background region.",
    },
    "camera_transformation": {
        "requires_camera_command": True,
        "requires_rigid_reference": True,
        "valid_commands": [
            "zoom_in", "zoom_out", "pan_left", "pan_right",
            "tilt_up", "tilt_down", "push_in", "pull_out",
            "orbit", "tracking", "static_camera",
        ],
        "description": "Prompt must specify a camera movement with a rigid reference structure in the scene.",
    },
}


# ===================================================================
# Part A: Evaluability Assessment (retained from v1)
# ===================================================================

def _compute_subject_overlap(
    text_result: TextAnalysisResult, image_result: ImageAnalysisResult
) -> Tuple[list, list, list, float]:
    """Compute overlap between text-target subjects and image-detected subjects."""
    text_subjects = set()
    for slot in text_result.attribute_change_slots:
        text_subjects.add(slot.target_subject.lower())
    for slot in text_result.action_slots:
        text_subjects.add(slot.target_subject.lower())
    for slot in text_result.motion_slots:
        text_subjects.add(slot.target_subject.lower())
    for slot in text_result.spatial_relation_slots:
        text_subjects.add(slot.subject_a.lower())
        text_subjects.add(slot.subject_b.lower())

    image_subjects = set()
    for s in image_result.subjects:
        image_subjects.add(s.name.lower())

    overlap = text_subjects & image_subjects
    ratio = len(overlap) / len(text_subjects) if text_subjects else 0.0
    return list(text_subjects), list(image_subjects), list(overlap), round(ratio, 4)


def _evaluate_dimension(
    text_result: TextAnalysisResult, image_result: ImageAnalysisResult
) -> Dict[str, DimensionEvaluability]:
    """Evaluate all 6 dimensions for a single sample."""
    dims = {}

    # 1. Attribute Binding
    text_req = text_result.involves_attribute_change
    img_sup = any(
        bool(s.attributes.color or s.attributes.state or s.attributes.wearing
             or s.attributes.material_texture or s.attributes.size)
        for s in image_result.subjects
    ) if image_result.subjects else False
    dims["attribute_binding"] = DimensionEvaluability(
        text_requests=text_req, image_supports=img_sup,
        evaluable=text_req and img_sup,
        reason=_make_reason("attribute_binding", text_req, img_sup),
    )

    # 2. Motion Binding
    text_req = text_result.involves_directed_motion
    img_sup = image_result.subject_count >= 1
    dims["motion_binding"] = DimensionEvaluability(
        text_requests=text_req, image_supports=img_sup,
        evaluable=text_req and img_sup,
        reason=_make_reason("motion_binding", text_req, img_sup),
    )

    # 3. Spatial Relation
    text_req = text_result.involves_spatial_relation_change
    img_sup = (
        image_result.has_multiple_subjects
        and image_result.subjects_clearly_distinguishable
    )
    dims["spatial_relation"] = DimensionEvaluability(
        text_requests=text_req, image_supports=img_sup,
        evaluable=text_req and img_sup,
        reason=_make_reason("spatial_relation", text_req, img_sup),
    )

    # 4. Action Binding
    text_req = text_result.involves_action
    img_sup = any(s.is_animate for s in image_result.subjects) if image_result.subjects else False
    dims["action_binding"] = DimensionEvaluability(
        text_requests=text_req, image_supports=img_sup,
        evaluable=text_req and img_sup,
        reason=_make_reason("action_binding", text_req, img_sup),
    )

    # 5. Scene / Background Dynamics
    text_req = text_result.involves_background_change
    img_sup = image_result.background.foreground_background_separability in ("high", "medium")
    dims["scene_dynamics"] = DimensionEvaluability(
        text_requests=text_req, image_supports=img_sup,
        evaluable=text_req and img_sup,
        reason=_make_reason("scene_dynamics", text_req, img_sup),
    )

    # 6. Camera / View Transformation
    text_req = text_result.involves_camera_movement
    img_sup = image_result.camera_baseline.has_rigid_reference_structure
    dims["camera_transformation"] = DimensionEvaluability(
        text_requests=text_req, image_supports=img_sup,
        evaluable=text_req and img_sup,
        reason=_make_reason("camera_transformation", text_req, img_sup),
    )

    return dims


def _make_reason(dim: str, text_req: bool, img_sup: bool) -> str:
    if text_req and img_sup:
        return f"{dim}: text requests and image supports"
    elif text_req and not img_sup:
        return f"{dim}: text requests but image does not support"
    elif not text_req and img_sup:
        return f"{dim}: text does not request (image could support)"
    else:
        return f"{dim}: neither text requests nor image supports"


# ===================================================================
# Part B: Per-Dimension Concept Distribution Extraction
# ===================================================================

def _counter_to_entries(counter: Counter, top_n: int = 30) -> List[dict]:
    """Convert Counter to sorted entry list with count and percentage."""
    total = sum(counter.values())
    entries = []
    for name, count in counter.most_common(top_n):
        entries.append({
            "name": str(name),
            "count": count,
            "pct": round(count / total * 100, 2) if total else 0,
        })
    return entries


def _extract_concept_distributions(
    dim_name: str,
    text_results: List[TextAnalysisResult],
    top_n: int = 30,
) -> List[ConceptDistribution]:
    """Extract concept distributions for a given dimension from its evaluable text results."""
    total = len(text_results)
    dists = []

    if dim_name == "attribute_binding":
        subj_ctr = Counter()
        attr_type_ctr = Counter()
        change_pattern_ctr = Counter()
        for t in text_results:
            for slot in t.attribute_change_slots:
                subj_ctr[slot.target_subject.lower()] += 1
                attr_type_ctr[slot.attribute_type.lower()] += 1
                if slot.from_value and slot.to_value:
                    change_pattern_ctr[f"{slot.from_value} -> {slot.to_value}"] += 1
        dists.append(ConceptDistribution(
            category="target_subject", entries=_counter_to_entries(subj_ctr, top_n), total_samples=total))
        dists.append(ConceptDistribution(
            category="attribute_type", entries=_counter_to_entries(attr_type_ctr, top_n), total_samples=total))
        if change_pattern_ctr:
            dists.append(ConceptDistribution(
                category="change_pattern", entries=_counter_to_entries(change_pattern_ctr, top_n), total_samples=total))

    elif dim_name == "action_binding":
        subj_ctr = Counter()
        verb_ctr = Counter()
        detail_ctr = Counter()
        for t in text_results:
            for slot in t.action_slots:
                subj_ctr[slot.target_subject.lower()] += 1
                verb_ctr[slot.action_verb.lower()] += 1
                if slot.action_detail:
                    detail_ctr[slot.action_detail.lower()] += 1
        dists.append(ConceptDistribution(
            category="target_subject", entries=_counter_to_entries(subj_ctr, top_n), total_samples=total))
        dists.append(ConceptDistribution(
            category="action_verb", entries=_counter_to_entries(verb_ctr, top_n), total_samples=total))
        if detail_ctr:
            dists.append(ConceptDistribution(
                category="action_detail", entries=_counter_to_entries(detail_ctr, top_n), total_samples=total))

    elif dim_name == "motion_binding":
        subj_ctr = Counter()
        dir_ctr = Counter()
        for t in text_results:
            for slot in t.motion_slots:
                subj_ctr[slot.target_subject.lower()] += 1
                dir_ctr[slot.direction.lower()] += 1
        dists.append(ConceptDistribution(
            category="target_subject", entries=_counter_to_entries(subj_ctr, top_n), total_samples=total))
        dists.append(ConceptDistribution(
            category="direction", entries=_counter_to_entries(dir_ctr, top_n), total_samples=total))

    elif dim_name == "spatial_relation":
        subj_ctr = Counter()
        pred_ctr = Counter()
        for t in text_results:
            for slot in t.spatial_relation_slots:
                subj_ctr[slot.subject_a.lower()] += 1
                subj_ctr[slot.subject_b.lower()] += 1
                pred_ctr[slot.target_predicate.lower()] += 1
        dists.append(ConceptDistribution(
            category="subject", entries=_counter_to_entries(subj_ctr, top_n), total_samples=total))
        dists.append(ConceptDistribution(
            category="predicate", entries=_counter_to_entries(pred_ctr, top_n), total_samples=total))

    elif dim_name == "scene_dynamics":
        region_ctr = Counter()
        change_type_ctr = Counter()
        to_state_ctr = Counter()
        for t in text_results:
            for slot in t.background_change_slots:
                region_ctr[slot.target_region.lower()] += 1
                change_type_ctr[slot.change_type.lower()] += 1
                to_state_ctr[slot.to_state.lower()] += 1
        dists.append(ConceptDistribution(
            category="target_region", entries=_counter_to_entries(region_ctr, top_n), total_samples=total))
        dists.append(ConceptDistribution(
            category="change_type", entries=_counter_to_entries(change_type_ctr, top_n), total_samples=total))
        dists.append(ConceptDistribution(
            category="to_state", entries=_counter_to_entries(to_state_ctr, top_n), total_samples=total))

    elif dim_name == "camera_transformation":
        cmd_ctr = Counter()
        target_ctr = Counter()
        for t in text_results:
            for slot in t.camera_movement_slots:
                cmd_ctr[slot.command.lower()] += 1
                if slot.target_subject:
                    target_ctr[slot.target_subject.lower()] += 1
        dists.append(ConceptDistribution(
            category="camera_command", entries=_counter_to_entries(cmd_ctr, top_n), total_samples=total))
        if target_ctr:
            dists.append(ConceptDistribution(
                category="camera_target_subject", entries=_counter_to_entries(target_ctr, top_n), total_samples=total))

    return dists


# ===================================================================
# Part C: Per-Dimension Visual Composition Prior
# ===================================================================

def _extract_visual_prior(
    image_results: List[ImageAnalysisResult],
    top_n: int = 30,
) -> VisualCompositionPrior:
    """Aggregate visual composition statistics from image analysis results."""
    if not image_results:
        return VisualCompositionPrior()

    subj_count_ctr = Counter()
    scene_ctr = Counter()
    shot_ctr = Counter()
    angle_ctr = Counter()
    sep_ctr = Counter()
    subj_cat_ctr = Counter()

    for r in image_results:
        subj_count_ctr[r.subject_count] += 1
        # Scene = lighting + weather + time_of_day
        bg = r.background
        scene_key = f"{bg.lighting}/{bg.weather}/{bg.time_of_day}"
        scene_ctr[scene_key] += 1
        # Camera
        cam = r.camera_baseline
        if cam.shot_type:
            shot_ctr[cam.shot_type] += 1
        if cam.camera_angle:
            angle_ctr[cam.camera_angle] += 1
        # Separability
        if bg.foreground_background_separability:
            sep_ctr[bg.foreground_background_separability] += 1
        # Subject categories
        for s in r.subjects:
            subj_cat_ctr[s.name.lower()] += 1

    total = len(image_results)

    def _to_dist(ctr: Counter, n: int = top_n) -> List[dict]:
        return [
            {"value": str(k), "count": v, "pct": round(v / total * 100, 2)}
            for k, v in ctr.most_common(n)
        ]

    return VisualCompositionPrior(
        subject_count_distribution=_to_dist(subj_count_ctr),
        scene_type_distribution=_to_dist(scene_ctr, top_n),
        shot_type_distribution=_to_dist(shot_ctr),
        camera_angle_distribution=_to_dist(angle_ctr),
        background_separability_distribution=_to_dist(sep_ctr),
        typical_subject_categories=_to_dist(subj_cat_ctr, top_n),
    )


# ===================================================================
# Part D: Structural Template Extraction
# ===================================================================

def _extract_structural_templates(
    text_results: List[TextAnalysisResult],
    min_count: int = 2,
) -> List[dict]:
    """
    Extract prompt structural templates by replacing concrete nouns/verbs/adjectives
    with placeholders, then counting pattern frequencies.
    """
    pattern_ctr = Counter()

    for t in text_results:
        prompt = t.clean_prompt if hasattr(t, "clean_prompt") else t.prompt_text
        if not prompt:
            continue

        template = prompt.lower().strip()

        # Replace specific nouns with {noun}
        nouns_sorted = sorted(t.nouns, key=len, reverse=True)  # longer first
        for noun in nouns_sorted:
            if len(noun) >= 2:
                template = re.sub(
                    r'\b' + re.escape(noun.lower()) + r'\b',
                    '{noun}', template, count=1
                )

        # Replace specific verbs with {verb}
        verbs_sorted = sorted(t.verbs, key=len, reverse=True)
        for verb in verbs_sorted:
            if len(verb) >= 2:
                template = re.sub(
                    r'\b' + re.escape(verb.lower()) + r'\b',
                    '{verb}', template, count=1
                )

        # Replace specific adjectives with {adj}
        adjs_sorted = sorted(t.adjectives, key=len, reverse=True)
        for adj in adjs_sorted:
            if len(adj) >= 2:
                template = re.sub(
                    r'\b' + re.escape(adj.lower()) + r'\b',
                    '{adj}', template, count=1
                )

        # Normalize multiple spaces
        template = re.sub(r'\s+', ' ', template).strip()

        if template and len(template) > 5:
            pattern_ctr[template] += 1

    # Filter by min_count and sort
    total = sum(pattern_ctr.values())
    results = []
    for pattern, count in pattern_ctr.most_common():
        if count >= min_count:
            results.append({
                "pattern": pattern,
                "count": count,
                "pct": round(count / total * 100, 2) if total else 0,
            })
    return results


# ===================================================================
# Part E: Seed Example Selection
# ===================================================================

def _select_seed_examples(
    dim_name: str,
    joint_results: List[JointAnalysisResult],
    text_map: Dict[str, TextAnalysisResult],
    image_map: Dict[str, ImageAnalysisResult],
    manifest_map: Dict[str, ManifestItem],
    max_seeds: int = 10,
) -> List[SeedExample]:
    """Select top-N seed examples for a dimension."""
    candidates = []

    for jr in joint_results:
        dim_eval = getattr(jr, f"dim_{dim_name}")
        if not dim_eval.evaluable:
            continue

        tr = text_map.get(jr.sample_id)
        ir = image_map.get(jr.sample_id)
        mi = manifest_map.get(jr.sample_id)
        if not tr or not ir or not mi:
            continue

        # Scoring: higher = better seed
        score = 0.0
        # Subject overlap
        score += jr.subject_overlap_ratio * 3.0
        # Slot completeness
        slot_count = _count_dim_slots(tr, dim_name)
        score += min(slot_count, 3) * 1.0
        # Prompt length (prefer 10-80 chars)
        plen = len(mi.clean_prompt_text)
        if 10 <= plen <= 80:
            score += 2.0
        elif plen > 80:
            score += 1.0

        candidates.append((score, jr.sample_id, tr, ir, mi))

    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    seeds = []
    for score, sid, tr, ir, mi in candidates[:max_seeds]:
        text_slots = _get_dim_text_slots(tr, dim_name)
        image_subj_summary = [
            {
                "name": s.name,
                "attributes": s.attributes.model_dump() if s.attributes else {},
                "pose_action": s.current_pose_action,
                "is_animate": s.is_animate,
            }
            for s in ir.subjects[:5]
        ]
        bg_summary = ir.background.model_dump() if ir.background else {}
        cam_summary = ir.camera_baseline.model_dump() if ir.camera_baseline else {}

        seeds.append(SeedExample(
            sample_id=sid,
            original_prompt=mi.prompt_text,
            clean_prompt=mi.clean_prompt_text,
            dimension=dim_name,
            text_slots=text_slots,
            image_subjects=image_subj_summary,
            image_background=bg_summary,
            image_camera=cam_summary,
            selection_reason=f"score={score:.2f}, overlap={jr.subject_overlap_ratio}",
            confidence=min(score / 8.0, 1.0),
        ))

    return seeds


def _count_dim_slots(tr: TextAnalysisResult, dim_name: str) -> int:
    """Count non-empty slots for a dimension."""
    if dim_name == "attribute_binding":
        return len(tr.attribute_change_slots)
    elif dim_name == "action_binding":
        return len(tr.action_slots)
    elif dim_name == "motion_binding":
        return len(tr.motion_slots)
    elif dim_name == "spatial_relation":
        return len(tr.spatial_relation_slots)
    elif dim_name == "scene_dynamics":
        return len(tr.background_change_slots)
    elif dim_name == "camera_transformation":
        return len(tr.camera_movement_slots)
    return 0


def _get_dim_text_slots(tr: TextAnalysisResult, dim_name: str) -> dict:
    """Extract the relevant text slots for a given dimension as a dict."""
    if dim_name == "attribute_binding":
        return {"attribute_change_slots": [s.model_dump() for s in tr.attribute_change_slots]}
    elif dim_name == "action_binding":
        return {"action_slots": [s.model_dump() for s in tr.action_slots]}
    elif dim_name == "motion_binding":
        return {"motion_slots": [s.model_dump() for s in tr.motion_slots]}
    elif dim_name == "spatial_relation":
        return {"spatial_relation_slots": [s.model_dump() for s in tr.spatial_relation_slots]}
    elif dim_name == "scene_dynamics":
        return {"background_change_slots": [s.model_dump() for s in tr.background_change_slots]}
    elif dim_name == "camera_transformation":
        return {"camera_movement_slots": [s.model_dump() for s in tr.camera_movement_slots]}
    return {}


# ===================================================================
# Part F: Global Distribution Extraction
# ===================================================================

def _extract_global_distributions(
    text_results: List[TextAnalysisResult],
    top_n: int = 30,
) -> List[ConceptDistribution]:
    """Extract global concept distributions across all analyzed samples."""
    total = len(text_results)
    dists = []

    # Intent distribution
    intent_ctr = Counter(t.primary_intent for t in text_results)
    dists.append(ConceptDistribution(
        category="primary_intent", entries=_counter_to_entries(intent_ctr, top_n), total_samples=total))

    # Global nouns
    noun_ctr = Counter()
    verb_ctr = Counter()
    adj_ctr = Counter()
    for t in text_results:
        noun_ctr.update(n.lower() for n in t.nouns)
        verb_ctr.update(v.lower() for v in t.verbs)
        adj_ctr.update(a.lower() for a in t.adjectives)

    dists.append(ConceptDistribution(
        category="noun", entries=_counter_to_entries(noun_ctr, top_n), total_samples=total))
    dists.append(ConceptDistribution(
        category="verb", entries=_counter_to_entries(verb_ctr, top_n), total_samples=total))
    dists.append(ConceptDistribution(
        category="adjective", entries=_counter_to_entries(adj_ctr, top_n), total_samples=total))

    # Dimension involvement rates
    dim_ctr = Counter()
    for t in text_results:
        if t.involves_attribute_change:
            dim_ctr["attribute_binding"] += 1
        if t.involves_action:
            dim_ctr["action_binding"] += 1
        if t.involves_directed_motion:
            dim_ctr["motion_binding"] += 1
        if t.involves_spatial_relation_change:
            dim_ctr["spatial_relation"] += 1
        if t.involves_background_change:
            dim_ctr["scene_dynamics"] += 1
        if t.involves_camera_movement:
            dim_ctr["camera_transformation"] += 1
    dists.append(ConceptDistribution(
        category="dimension_involvement", entries=_counter_to_entries(dim_ctr, 10), total_samples=total))

    return dists


# ===================================================================
# Part G: Pika Parameter Distribution
# ===================================================================

def _extract_pika_distributions(
    manifest_items: List[ManifestItem],
) -> Tuple[List[dict], List[dict]]:
    """Extract Pika camera and motion parameter distributions."""
    camera_ctr = Counter()
    motion_ctr = Counter()
    total = len(manifest_items)

    for m in manifest_items:
        if m.pika_camera:
            camera_ctr[m.pika_camera.lower()] += 1
        if m.pika_motion is not None:
            motion_ctr[m.pika_motion] += 1

    def _to_list(ctr: Counter) -> List[dict]:
        return [
            {"value": str(k), "count": v, "pct": round(v / total * 100, 2)}
            for k, v in ctr.most_common()
        ]

    return _to_list(camera_ctr), _to_list(motion_ctr)


# ===================================================================
# Part H: Co-occurrence Matrix
# ===================================================================

def _compute_cooccurrence(results: List[JointAnalysisResult]) -> dict:
    """Compute 6x6 dimension co-occurrence dictionary."""
    n = len(results)
    if n == 0:
        return {}

    matrix = {}
    for d1 in DIMENSIONS:
        for d2 in DIMENSIONS:
            if d1 <= d2:
                matrix[f"{d1}×{d2}"] = 0

    for r in results:
        active = []
        for dim_name in DIMENSIONS:
            dim_eval = getattr(r, f"dim_{dim_name}")
            if dim_eval.text_requests:
                active.append(dim_name)

        for i, d1 in enumerate(active):
            for d2 in active[i + 1:]:
                key = f"{min(d1, d2)}×{max(d1, d2)}"
                if key in matrix:
                    matrix[key] += 1

    # Convert to percentage
    return {k: round(v / n * 100, 2) for k, v in matrix.items() if v > 0}


# ===================================================================
# Main Entry
# ===================================================================

def joint_analyze(config: dict) -> None:
    """
    Main entry: combine text + image analysis into dimension-level evaluability
    and extract deep priors for downstream LLM prompt generation.
    """
    output_base = Path(config["paths"]["output_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    text_path = str(output_base / "text_analysis" / "text_parse.jsonl")
    image_path = str(output_base / "image_analysis" / "image_parse.jsonl")
    manifest_path = str(manifest_dir / "manifest_clean.jsonl")
    joint_dir = str(output_base / "joint_analysis")
    seed_dir = str(Path(joint_dir) / "seed_examples")
    ensure_dir(joint_dir)
    ensure_dir(seed_dir)

    # Prior package config
    pp_cfg = config.get("prior_package", {})
    seed_per_dim = pp_cfg.get("seed_examples_per_dim", 10)
    top_n = pp_cfg.get("top_n_concepts", 30)
    template_min = pp_cfg.get("template_min_count", 2)

    # ------ Load data ------
    text_results = read_jsonl(text_path, TextAnalysisResult)
    image_results = read_jsonl(image_path, ImageAnalysisResult)
    manifest_items = read_jsonl(manifest_path, ManifestItem)

    if not text_results:
        logger.error("No text analysis results found. Run step3 first.")
        return
    if not image_results:
        logger.error("No image analysis results found. Run step2 first.")
        return

    text_map = {r.sample_id: r for r in text_results if r.parse_success}
    image_map = {r.sample_id: r for r in image_results if r.parse_success}
    manifest_map = {m.sample_id: m for m in manifest_items}

    common_ids = set(text_map.keys()) & set(image_map.keys())
    logger.info(
        f"Text: {len(text_map)}, Image: {len(image_map)}, "
        f"Common (inner join): {len(common_ids)}"
    )
    if not common_ids:
        logger.error("No common sample IDs between text and image results.")
        return

    # ====== A. Evaluability assessment ======
    logger.info("Phase 1/5: Evaluability assessment...")
    joint_results = []
    for sid in sorted(common_ids):
        text_r = text_map[sid]
        image_r = image_map[sid]
        dims = _evaluate_dimension(text_r, image_r)
        text_subs, image_subs, overlap, ratio = _compute_subject_overlap(text_r, image_r)

        result = JointAnalysisResult(
            sample_id=sid,
            dim_attribute_binding=dims["attribute_binding"],
            dim_motion_binding=dims["motion_binding"],
            dim_spatial_relation=dims["spatial_relation"],
            dim_action_binding=dims["action_binding"],
            dim_scene_dynamics=dims["scene_dynamics"],
            dim_camera_transformation=dims["camera_transformation"],
            text_target_subjects=text_subs,
            image_detected_subjects=image_subs,
            subject_overlap=overlap,
            subject_overlap_ratio=ratio,
        )
        joint_results.append(result)

    # Write joint analysis JSONL
    joint_path = str(Path(joint_dir) / "joint_analysis.jsonl")
    write_jsonl(joint_path, joint_results)

    # Dimension coverage CSV
    total = len(joint_results)
    coverage_rows = []
    for dim_name in DIMENSIONS:
        text_req_count = sum(1 for r in joint_results if getattr(r, f"dim_{dim_name}").text_requests)
        img_sup_count = sum(1 for r in joint_results if getattr(r, f"dim_{dim_name}").image_supports)
        eval_count = sum(1 for r in joint_results if getattr(r, f"dim_{dim_name}").evaluable)
        coverage_rows.append({
            "dimension": dim_name,
            "text_request_count": text_req_count,
            "text_request_pct": round(text_req_count / total * 100, 2) if total else 0,
            "image_support_count": img_sup_count,
            "image_support_pct": round(img_sup_count / total * 100, 2) if total else 0,
            "evaluable_count": eval_count,
            "evaluable_pct": round(eval_count / total * 100, 2) if total else 0,
        })
    write_csv(str(Path(joint_dir) / "dimension_coverage.csv"), coverage_rows)

    # ====== B. Per-dimension prior extraction ======
    logger.info("Phase 2/5: Per-dimension concept distribution extraction...")
    dim_priors = []
    for dim_name in DIMENSIONS:
        # Collect evaluable sample IDs for this dimension
        eval_sids = [
            r.sample_id for r in joint_results
            if getattr(r, f"dim_{dim_name}").evaluable
        ]
        eval_texts = [text_map[sid] for sid in eval_sids if sid in text_map]
        eval_images = [image_map[sid] for sid in eval_sids if sid in image_map]
        eval_count = len(eval_sids)
        eval_pct = round(eval_count / total * 100, 2) if total else 0.0

        # B1: Concept distributions
        concept_dists = _extract_concept_distributions(dim_name, eval_texts, top_n)

        # B2: Visual composition prior
        visual_prior = _extract_visual_prior(eval_images, top_n)

        # B3: Structural templates
        templates = _extract_structural_templates(eval_texts, template_min)

        # B4: Seed examples
        seeds = _select_seed_examples(
            dim_name, joint_results, text_map, image_map, manifest_map, seed_per_dim
        )

        dp = DimensionPrior(
            dimension=dim_name,
            display_name=DIM_DISPLAY_NAMES[dim_name],
            sample_count=eval_count,
            coverage_pct=eval_pct,
            concept_distributions=concept_dists,
            visual_prior=visual_prior,
            structural_templates=templates[:50],  # cap at 50
            seed_examples=seeds,
            constraints=DIM_CONSTRAINTS.get(dim_name, {}),
        )
        dim_priors.append(dp)

        # Write seed examples per dimension
        if seeds:
            write_jsonl(str(Path(seed_dir) / f"seed_{dim_name}.jsonl"), seeds)

        logger.info(
            f"  {DIM_DISPLAY_NAMES[dim_name]}: "
            f"{eval_count} evaluable, {len(concept_dists)} distributions, "
            f"{len(templates)} templates, {len(seeds)} seeds"
        )

    # Write dimension priors
    write_jsonl(str(Path(joint_dir) / "dimension_priors.jsonl"), dim_priors)

    # ====== C. Global distributions ======
    logger.info("Phase 3/5: Global concept distributions...")
    all_text_success = [t for t in text_results if t.parse_success]
    global_dists = _extract_global_distributions(all_text_success, top_n)
    write_jsonl(str(Path(joint_dir) / "global_distributions.jsonl"), global_dists)

    # ====== D. Global visual prior ======
    logger.info("Phase 4/5: Global visual composition prior...")
    all_image_success = [i for i in image_results if i.parse_success]
    global_visual = _extract_visual_prior(all_image_success, top_n)
    with open(str(Path(joint_dir) / "global_visual_prior.json"), "w", encoding="utf-8") as f:
        json.dump(global_visual.model_dump(), f, ensure_ascii=False, indent=2)

    # ====== E. Pika distributions ======
    logger.info("Phase 5/5: Pika parameter distributions...")
    pika_cam_dist, pika_motion_dist = _extract_pika_distributions(manifest_items)
    with open(str(Path(joint_dir) / "pika_distributions.json"), "w", encoding="utf-8") as f:
        json.dump({
            "pika_camera_distribution": pika_cam_dist,
            "pika_motion_distribution": pika_motion_dist,
        }, f, ensure_ascii=False, indent=2)

    # ====== F. Co-occurrence matrix ======
    cooccurrence = _compute_cooccurrence(joint_results)
    with open(str(Path(joint_dir) / "dimension_cooccurrence.json"), "w", encoding="utf-8") as f:
        json.dump(cooccurrence, f, ensure_ascii=False, indent=2)

    # Also write mixed_intent_matrix.csv for backward compat
    _write_mixed_intent_csv(joint_results, joint_dir)

    # ====== Summary log ======
    logger.info("=" * 60)
    logger.info("Joint Analysis + Prior Extraction Summary")
    logger.info("=" * 60)
    logger.info(f"Total joint samples: {total}")
    for row in coverage_rows:
        logger.info(
            f"  {row['dimension']:25s}: "
            f"text_req={row['text_request_pct']:5.1f}%  "
            f"img_sup={row['image_support_pct']:5.1f}%  "
            f"evaluable={row['evaluable_pct']:5.1f}% ({row['evaluable_count']})"
        )
    logger.info(f"Output: {joint_dir}")


def _write_mixed_intent_csv(results: List[JointAnalysisResult], joint_dir: str) -> None:
    """Write 6x6 mixed intent co-occurrence matrix CSV."""
    n = len(results)
    if n == 0:
        return

    matrix = {}
    for d1 in DIMENSIONS:
        matrix[d1] = {}
        for d2 in DIMENSIONS:
            matrix[d1][d2] = 0

    for r in results:
        active = []
        for dim_name in DIMENSIONS:
            dim_eval = getattr(r, f"dim_{dim_name}")
            if dim_eval.text_requests:
                active.append(dim_name)
        for i, d1 in enumerate(active):
            for d2 in active[i + 1:]:
                matrix[d1][d2] += 1
                matrix[d2][d1] += 1

    df = pd.DataFrame(matrix)
    if n > 0:
        df = (df / n * 100).round(2)
    df.to_csv(str(Path(joint_dir) / "mixed_intent_matrix.csv"), encoding="utf-8-sig")
    logger.info("Wrote mixed intent co-occurrence matrix")
