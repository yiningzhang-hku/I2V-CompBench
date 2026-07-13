"""
Quality-experiment Pydantic v2 schemas for I2V-CompBench.

Covers §6.0–§6.7: source-asset lineage, quality candidates,
target repair, prompt/image variants, human annotation, selection,
and difficulty features.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from i2vcompbench.schemas.phase2 import PreserveItem, SubjectRef, TargetRelation

# ============================================================
# §6.0 — Source Asset & Lineage
# ============================================================


class SourceAssetRecord(BaseModel):
    """Upstream asset provenance record."""

    upstream_asset_id: str
    source_sample_id: str
    asset_role: Literal["source_first_frame", "reference_asset"]
    upstream_path: str  # POSIX relative path
    canonical_upstream_sha256: str
    width: int
    height: int
    source_manifest_path: str
    source_manifest_sha256: str
    verification_source: Literal["phase1_manifest", "manual_migration"]
    verified_by: str | None = None
    verified_at: str | None = None


class AssetLineageRecord(BaseModel):
    """Tracks transform lineage from source to derived asset."""

    question_id: str
    role: str
    source_type: Literal["tip_derived_reference"]
    source_sample_id: str
    upstream_asset_id: str
    canonical_upstream_sha256: str
    observed_upstream_sha256: str
    transform_name: str
    transform_params: dict[str, Any]
    transform_spec_sha256: str
    derived_path: str
    expected_derived_sha256: str
    code_version: str
    verification_source: Literal[
        "construction_run", "deterministic_migration", "manual_migration"
    ]
    status: Literal[
        "verified", "needs_manual_review", "source_mismatch", "derived_mismatch"
    ]


# ============================================================
# §6.1 — Quality Candidate
# ============================================================


class QualityCandidate(BaseModel):
    """A candidate question entering the quality experiment pipeline."""

    question_id: str
    dimension: Literal[
        "attribute_binding",
        "action_binding",
        "motion_binding",
        "background_dynamics",
        "view_transformation",
    ]
    difficulty_old: Literal["easy", "medium", "hard"]
    semantic_rarity: Literal["common", "rare"]
    prompt: str
    first_frame_path: str
    source_manifest_hash: str
    source_sample_id: str | None = None
    upstream_asset_id: str | None = None
    canonical_upstream_sha256: str | None = None
    lineage_observed_upstream_sha256: str | None = None
    lineage_expected_derived_sha256: str | None = None
    resolved_derived_sha256: str | None = None
    asset_lineage_record_sha256: str | None = None
    asset_lineage_manifest_sha256: str | None = None
    resolution_source: Literal[
        "manifest_path", "asset_lineage", "legacy_candidate", "unresolved"
    ] = "unresolved"
    asset_binding_status: Literal[
        "verified",
        "needs_manual_review",
        "missing_source_hash",
        "missing_lineage",
        "source_hash_mismatch",
        "derived_hash_mismatch",
        "ambiguous",
        "missing",
    ] = "missing"
    source_plan_hash: str | None = None
    quadrant: Literal[
        "A_control", "B_lexical_probe", "C_subject_probe", "D_compound_probe"
    ] | None = None


# ============================================================
# §6.2 — Target Repair
# ============================================================


class TargetRepairResult(BaseModel):
    """Result of target-relation repair/verification."""

    question_id: str
    target_subjects: list[SubjectRef]
    target_relation: TargetRelation
    preservation_set: list[PreserveItem]
    reviewed_dimension: str
    dimension_consistent: bool
    confidence: float
    repair_source: Literal["phase1_rebuild", "vlm_migration", "manual"]
    status: Literal[
        "pass", "needs_manual_review", "rejected", "api_failed", "invalid_response"
    ]
    raw_response_path: str | None = None


# ============================================================
# §6.3 — Prompt Variant
# ============================================================


class PromptVariantResult(BaseModel):
    """Result of a prompt rewriting experiment."""

    question_id: str
    method: Literal[
        "A0", "A1", "A2", "A3", "A4", "A5", "A6",
        "B0", "B1", "B2", "B3",
    ]
    source_layer: Literal["P1-Source-Off", "P1-Source-On"] = "P1-Source-Off"
    generator_id: str | None = None
    generator_version: str | None = None
    decoding_config_hash: str | None = None
    parent_method: str | None = None
    prompt_before: str
    prompt_after: str
    word_count: int
    rare_modifier_hits: list[str]
    structural_issues: list[str]
    semantic_similarity: float | None = None
    semantic_preservation_pass: bool | None = None
    dimension_consistent: bool | None = None
    rare_modifier_rate: float | None = None
    mean_zipf_score: float | None = None
    clip_sim_delta: float | None = None
    perplexity_mean: float | None = None
    api_calls: int = 0
    status: Literal[
        "pass", "failed", "needs_manual_review",
        "unsupported_optional", "missing_dependency",
    ]


# ============================================================
# §6.4 — Image Variant
# ============================================================


class ImageVariantResult(BaseModel):
    """Result of an image enhancement/transform experiment."""

    question_id: str
    experiment: Literal["clarity", "aspect"]
    method: str
    source_path: str
    output_path: str
    upstream_asset_id: str
    canonical_upstream_sha256: str
    input_asset_sha256: str
    input_stage: Literal["canonical_source", "clarity_output", "aspect_output"]
    output_sha256: str
    asset_binding_verified: bool
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    metrics: dict[str, float | None]
    status: Literal[
        "pass", "failed", "missing_dependency",
        "missing_weight", "unsupported_optional",
    ]


# ============================================================
# §6.5 — Human Annotation (Discriminated Union)
# ============================================================

RubricScore = Literal[0.0, 0.5, 1.0]


class AnnotationBase(BaseModel):
    """Common fields shared across all annotation types."""

    annotation_id: str
    question_id: str
    experiment: str
    method_code: str  # blinded code
    annotator_id: str
    task_version: str
    created_at: str
    comment: str = ""


class TargetRepairAnnotation(AnnotationBase):
    experiment: Literal["target_repair"]
    subject_correct: bool
    relation_correct: bool
    preservation_correct: bool
    dimension_correct: bool


class PromptAnnotation(AnnotationBase):
    experiment: Literal["prompt", "fallback"]
    target_consistent: bool
    dimension_correct: bool
    prompt_naturalness: int  # 1..5
    overall_usable: bool


class ImageAnnotation(AnnotationBase):
    experiment: Literal["clarity", "aspect"]
    clarity_score: int | None = None
    subject_complete: bool
    scene_reference_preserved: bool
    geometry_distortion: int  # 1..5
    identity_changed: bool | None = None
    artifact_present: bool
    overall_usable: bool


class DifficultyAnnotation(AnnotationBase):
    experiment: Literal["difficulty"]
    difficulty_label: Literal["easy", "medium", "hard"]
    target_complexity: RubricScore
    subject_localization: RubricScore
    background_interference: RubricScore
    temporal_complexity: RubricScore
    initial_state_visibility: RubricScore
    target_observability: RubricScore
    alternative_explanation_risk: RubricScore
    evaluator_support_gap: RubricScore


class FinalReviewAnnotation(AnnotationBase):
    experiment: Literal["final_review"]
    asset_correct: bool
    dimension_correct: bool
    subject_correct: bool
    transform_complete: bool
    preservation_valid: bool
    target_consistent: bool
    final_accept: bool
    rejection_reasons: list[
        Literal[
            "asset_mismatch",
            "wrong_dimension",
            "subject_incorrect",
            "target_relation_incorrect",
            "preservation_invalid",
            "prompt_invalid",
            "image_artifact",
            "subject_lost",
            "scene_reference_lost",
            "other",
        ]
    ]


# Discriminated Union
HumanAnnotation = Annotated[
    TargetRepairAnnotation
    | PromptAnnotation
    | ImageAnnotation
    | DifficultyAnnotation
    | FinalReviewAnnotation,
    Field(discriminator="experiment"),
]


class AdjudicationRecord(BaseModel):
    """Multi-annotator disagreement resolution record."""

    adjudication_id: str
    question_id: str
    experiment: str
    source_annotation_ids: list[str]
    adjudicator_id: str
    resolved_payload: dict[str, Any]
    rationale: str
    created_at: str


# ============================================================
# §6.6 — Selection Decision
# ============================================================


class SelectionDecision(BaseModel):
    """Final eligibility and selection record for a question."""

    question_id: str
    dimension: str
    eligible: bool
    blocking_reasons: list[str]
    prompt_method: str | None
    clarity_method: str | None
    aspect_method: str | None
    native_first_frame_path: str | None
    native_first_frame_sha256: str | None
    inference_companion_path: str | None
    inference_companion_sha256: str | None
    subject_tier: str | None
    difficulty_new: str | None
    quality_rank_components: dict[str, float | int | None]
    quality_rank_score: float | None
    selection_order_key: str
    selected: bool = False
    bucket_assignments: dict[str, str] = Field(default_factory=dict)


# ============================================================
# §6.7 — Difficulty Feature
# ============================================================


class DifficultyFeatureRecord(BaseModel):
    """Multi-dimensional difficulty scoring features."""

    question_id: str
    target_complexity: float
    subject_localization: float
    background_interference: float
    temporal_complexity: float
    initial_state_visibility: float
    target_observability: float
    alternative_explanation_risk: float
    evaluator_support_gap: float
    d_task: float
    d_judge: float
    evidence: dict[str, list[str]]
    feature_sources: dict[str, Literal["rule", "vlm", "manual"]]
    feature_confidence: dict[str, float]
    status: Literal["pass", "needs_manual_review", "rejected", "missing_evidence"]
    predicted_label: Literal["easy", "medium", "hard"] | None = None
    predicted_probabilities: dict[str, float] | None = None
    prediction_uncertainty: float | None = None
    calibration_model_hash: str | None = None
