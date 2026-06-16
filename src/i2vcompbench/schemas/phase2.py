"""
Phase 2 Pydantic schemas.

严格对齐 Phase2_Benchmark数据集合成.md §4.3 / §4.4 / §4.7 / §7.2 与 Phase 3 §2.4 schema：
- BenchmarkSample 顶层 17 字段全部扁平化（禁止 metadata 嵌套）
- 字段名 `prompt`（不是 i2v_prompt）
- source_type 枚举 ⊂ {tip_i2v_real, tip_i2v_synthetic_first_frame, external_real, external_synthetic}
- contrastive_role 枚举 ⊂ {original, baseline_static_copy, baseline_random_motion,
                          baseline_global_filter, baseline_camera_pan_cheat,
                          baseline_subject_swap}
- evaluator_tools / expected_failure_modes 强枚举（_TOOLS_BY_DIM / _FAILURES_BY_DIM）
- 调试字段统一收纳到 _audit 子节点（phase3_manifest.jsonl 必须剔除）
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# ============================================================
# 全局枚举
# ============================================================

DIMENSIONS_V2: List[str] = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "spatial_composition",
    "background_dynamics",
    "view_transformation",
    "interaction_reasoning",
]

DIM_SHORT: Dict[str, str] = {
    "attribute_binding": "attr",
    "action_binding": "act",
    "motion_binding": "motion",
    "spatial_composition": "spatial",
    "background_dynamics": "bg",
    "view_transformation": "view",
    "interaction_reasoning": "inter",
}

InputMode = Literal["single_image", "multi_image"]

# Phase 3 顶层 source_type 强枚举（4 项）
SourceType = Literal[
    "tip_i2v_real",
    "tip_i2v_synthetic_first_frame",
    "external_real",
    "external_synthetic",
]

# 资产层 source_type（input_assets_manifest / 内部回退使用，不暴露到 BenchmarkSample 顶层）
AssetSourceType = Literal[
    "tip_derived_reference",
    "t2i_generated",
    "external",
]

QCStatus = Literal["pass", "fail", "needs_manual_review"]

# Phase 2 §4.7 step 5：6 项 contrastive_role 枚举
ContrastiveRole = Literal[
    "original",
    "baseline_static_copy",
    "baseline_random_motion",
    "baseline_global_filter",
    "baseline_camera_pan_cheat",
    "baseline_subject_swap",
]

# Phase 3 §A 工具枚举（9 项）
EvaluatorTool = Literal[
    "grounding",
    "depth",
    "dot_motion",
    "optical_flow",
    "vlm_existence",
    "vlm_attribute",
    "vlm_relation",
    "dinov2",
    "clip",
]

# Phase 3 §2.5 FAILURE_MODES（15 项，词汇以 Phase 3 §2.5 为准）
FAILURE_MODES: List[str] = [
    "static_copy",
    "global_filter",
    "camera_pan_cheat",
    "object_missing",
    "wrong_attribute",
    "wrong_direction",
    "wrong_relation",
    "wrong_camera",
    "identity_lost",
    "non_target_drift",
    "background_drift",
    "artifact_severe",
    "timing_wrong",
    "identity_unbound",
    "tool_uncertain",
]


# ============================================================
# 衔接强枚举映射表（Phase2 §A / §C 附录权威表）
# ============================================================

# §A.1 dimension -> required evaluator_tools
_TOOLS_BY_DIM: Dict[str, List[str]] = {
    "attribute_binding": ["grounding", "vlm_attribute", "vlm_existence"],
    "action_binding": ["grounding", "vlm_existence", "optical_flow"],
    "motion_binding": ["grounding", "dot_motion", "optical_flow"],
    "spatial_composition": ["grounding", "depth", "vlm_relation"],
    "background_dynamics": ["grounding", "optical_flow", "vlm_existence"],
    "view_transformation": ["depth", "optical_flow", "vlm_existence"],
    "interaction_reasoning": ["grounding", "vlm_relation", "optical_flow"],
}

# multi_image 加挂工具
_MULTI_IMAGE_EXTRA_TOOLS: List[str] = ["dinov2", "clip", "grounding"]

# §A.2 dimension -> default expected_failure_modes（取值必 ⊂ Phase 3 §2.5 15 项）
_FAILURES_BY_DIM: Dict[str, List[str]] = {
    "attribute_binding": ["wrong_attribute", "object_missing", "identity_lost"],
    "action_binding": ["static_copy", "object_missing", "timing_wrong"],
    "motion_binding": ["wrong_direction", "static_copy", "camera_pan_cheat"],
    "spatial_composition": ["wrong_relation", "object_missing", "identity_lost"],
    "background_dynamics": ["background_drift", "non_target_drift", "global_filter"],
    "view_transformation": ["wrong_camera", "camera_pan_cheat", "identity_lost"],
    "interaction_reasoning": ["wrong_relation", "timing_wrong", "object_missing"],
}

# §C `_LEGACY_SOURCE_MAP`：Phase 1 source_type -> Phase 3 source_type
_LEGACY_SOURCE_MAP: Dict[str, str] = {
    "observed_single_image": "tip_i2v_real",
    "derived_single_image": "tip_i2v_synthetic_first_frame",
    "derived_multi_reference": "tip_i2v_synthetic_first_frame",
    "external_real": "external_real",
    "external_synthetic": "external_synthetic",
}


def map_legacy_source_type(legacy: Optional[str]) -> str:
    """把 Phase 1 旧词汇映射到 Phase 3 顶层 source_type 强枚举值。

    未知/缺省值默认按 tip_i2v_synthetic_first_frame 落地（最常见的 derived 情形）。
    """
    if not legacy:
        return "tip_i2v_synthetic_first_frame"
    if legacy in _LEGACY_SOURCE_MAP:
        return _LEGACY_SOURCE_MAP[legacy]
    # 已经是 Phase 3 词汇时原样透传
    if legacy in {"tip_i2v_real", "tip_i2v_synthetic_first_frame", "external_real", "external_synthetic"}:
        return legacy
    return "tip_i2v_synthetic_first_frame"


def required_tools_for(dimension: str, input_mode: str) -> List[str]:
    """按 §A.1 强枚举返回 required evaluator_tools；multi_image 时追加 dinov2/clip/grounding。"""
    base = list(_TOOLS_BY_DIM.get(dimension, []))
    if input_mode == "multi_image":
        for t in _MULTI_IMAGE_EXTRA_TOOLS:
            if t not in base:
                base.append(t)
    return base


def default_failure_modes_for(dimension: str) -> List[str]:
    """按 §A.2 强枚举返回默认 expected_failure_modes。"""
    return list(_FAILURES_BY_DIM.get(dimension, []))


# ============================================================
# QuestionPlan 子结构
# ============================================================

class RequiredImageSpec(BaseModel):
    """input_plan.required_images 中的一项。"""

    role: str  # e.g. first_frame / target_subject / attribute_reference / scene_reference
    description: str = ""
    source_preference: List[AssetSourceType] = Field(
        default_factory=lambda: ["tip_derived_reference", "t2i_generated"]
    )


class InputPlan(BaseModel):
    required_images: List[RequiredImageSpec] = Field(default_factory=list)


class SubjectRef(BaseModel):
    """target_subjects[i]：Phase 3 evaluator 通过 id 引用主体。"""

    id: str  # s1 / s2 / ...
    description: str
    noun: Optional[str] = None
    ref_image_idx: Optional[int] = None  # multi_image 时必填，与 ref_images/{qid}_ref{k}.png 中的 k 一致


class TargetRelation(BaseModel):
    """target_relation：与 Phase 3 §2.6 对齐：{type, value, subj, obj}。

    type: 关系类别（spatial / interaction / temporal / ...）
    value: 具体关系值（on / left_of / hand_to / ...）
    subj/obj: 引用 SubjectRef.id（如 s1, s2）
    """

    type: str = ""
    value: str = ""
    subj: Optional[str] = None
    obj: Optional[str] = None


class PreserveItem(BaseModel):
    scope: str  # target_identity / background / camera / s2 等
    constraint: str  # preserve / stable / appearance_and_position 等


class TargetPlan(BaseModel):
    """要求模型在视频中执行的变换。"""

    target_subjects: List[SubjectRef] = Field(default_factory=list)
    target_relation: Optional[TargetRelation] = None
    operation: str = "transform"
    attribute_source: Optional[str] = None  # 如 ref:attribute_reference
    expected_final_state: str = ""


class DimensionIsolationPlan(BaseModel):
    """Phase 2 在 question 层独立保存一份，便于 finalize 与 verify 直接读。

    camera_constraint 取值（§6.3）：
        - static_or_minor：镜头静止或仅有微弱自然抖动
        - forbidden：禁止任何主动运镜（view_transformation 以外维度默认）
        - required_active：必须包含明确运镜（仅 view_transformation 维度）
    """

    forbidden_words: List[str] = Field(default_factory=list)
    camera_constraint: Literal["static_or_minor", "forbidden", "required_active"] = "forbidden"


class QuestionPlan(BaseModel):
    """Phase2 §4.3 的核心 schema。"""

    question_id: str
    recipe_id: str
    dimension: str  # ⊂ DIMENSIONS_V2
    input_mode: InputMode
    subtype: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    semantic_rarity: Literal["common", "rare"] = "common"
    contrastive_pair_id: Optional[str] = None
    contrastive_role: ContrastiveRole = "original"

    input_plan: InputPlan = Field(default_factory=InputPlan)
    target_plan: TargetPlan = Field(default_factory=TargetPlan)
    preserve_plan: List[PreserveItem] = Field(default_factory=list)
    dimension_isolation: DimensionIsolationPlan = Field(
        default_factory=DimensionIsolationPlan
    )

    # §4.3 step 7/8：强枚举落地，禁止自由文本
    evaluator_tools: List[str] = Field(default_factory=list)
    expected_failure_modes: List[str] = Field(default_factory=list)

    prompt_draft: str = ""


# ============================================================
# InputAsset
# ============================================================

class AssetQualityLite(BaseModel):
    """Phase 2 输入资产的轻量质量字段（区别于 Phase 1 ReferenceAsset.quality）。"""

    identity_visibility: Literal["high", "medium", "low", "unknown"] = "unknown"
    crop_leakage_risk: Literal["low", "medium", "high", "unknown"] = "unknown"
    resolution_ok: bool = True
    notes: str = ""


class InputAssetItem(BaseModel):
    """input_assets_manifest.jsonl 中的一项。"""

    asset_id: str
    role: str  # first_frame / target_subject / attribute_reference / scene_reference 等
    path: str
    source_type: AssetSourceType
    source_ref_id: Optional[str] = None  # Phase 1 sample_id / asset_id；T2I 时为 None
    ref_image_idx: Optional[int] = None  # multi_image 参考的下标，与 SubjectRef.ref_image_idx 对齐
    quality: AssetQualityLite = Field(default_factory=AssetQualityLite)


class InputAssetManifest(BaseModel):
    question_id: str
    assets: List[InputAssetItem] = Field(default_factory=list)


# ============================================================
# QC
# ============================================================

class QCCheck(BaseModel):
    name: str
    answer: bool
    confidence: float = 0.0
    rationale: str = ""


class QCReport(BaseModel):
    """qc_reports/<question_id>.json"""

    question_id: str
    qc_status: QCStatus = "fail"
    checks: List[QCCheck] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    retry_count: int = 0
    notes: str = ""


# ============================================================
# Finalized prompt（字段名 prompt，不是 i2v_prompt）
# ============================================================

class FinalPromptEntry(BaseModel):
    question_id: str
    prompt: str
    length_words: int = 0
    forbidden_hits: List[str] = Field(default_factory=list)
    polish_attempts: int = 1
    used_fallback: bool = False
    vlm_caption: str = ""
    failed_check: Optional[str] = None  # 如 missing_active_verb / forbidden_hit / out_of_range


# ============================================================
# BenchmarkSample（Phase 2 终交付物）—— Phase 3 §2.4 顶层扁平 17 字段
# ============================================================

class MultiReferenceQuality(BaseModel):
    crop_leakage_risk: Literal["low", "medium", "high", "unknown"] = "unknown"
    scene_leakage_risk: Literal["low", "medium", "high", "unknown"] = "unknown"
    identity_visibility: Literal["high", "medium", "low", "unknown"] = "unknown"
    scale_compatibility: float = 0.0


class SourceTrace(BaseModel):
    """`_audit.source_trace`：调试与可审计性，禁止进入 phase3_manifest。"""

    recipe_id: str
    legacy_source_type: str = ""  # 来自 Phase 1 CandidateRecipe.source_type
    phase1_sample_ids: List[str] = Field(default_factory=list)
    phase1_asset_ids: List[str] = Field(default_factory=list)


class QCSummary(BaseModel):
    status: QCStatus = "pass"
    risk_flags: List[str] = Field(default_factory=list)


class BenchmarkAudit(BaseModel):
    """`_audit` 子节点：调试与可审计性字段汇总；phase3_manifest.jsonl 必须剔除。"""

    source_trace: SourceTrace
    qc: QCSummary = Field(default_factory=QCSummary)
    multi_reference_quality: Optional[MultiReferenceQuality] = None


class BenchmarkSample(BaseModel):
    """samples/{dimension}.jsonl + phase3_manifest.jsonl 的行级 schema。

    顶层严格 17 字段（与 Phase 3 §2.4 / Phase 2 §7.2 一致），禁止 metadata 嵌套。
    """

    # ---- 顶层 17 字段 ----
    question_id: str
    dimension: str
    input_mode: InputMode
    first_frame_path: str
    input_image_paths: List[str] = Field(default_factory=list)
    prompt: str
    target_subjects: List[SubjectRef] = Field(default_factory=list)
    target_relation: Optional[TargetRelation] = None
    preservation_set: List[PreserveItem] = Field(default_factory=list)
    contrastive_pair_id: Optional[str] = None
    contrastive_role: ContrastiveRole = "original"
    evaluator_tools: List[str] = Field(default_factory=list)
    expected_failure_modes: List[str] = Field(default_factory=list)
    subtype: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    semantic_rarity: Literal["common", "rare"] = "common"
    source_type: SourceType = "tip_i2v_synthetic_first_frame"

    # ---- 调试与可审计性（_audit 子节点；phase3_manifest 必须剔除） ----
    audit: Optional[BenchmarkAudit] = Field(default=None, alias="_audit")

    model_config = {"populate_by_name": True}


# ============================================================
# Quota schema
# ============================================================

class QuotaBucket(BaseModel):
    bucket_id: str
    dimension: str
    input_mode_or_subtype: str  # single_image / multi_image / type_a_absolute_single ...
    difficulty: Literal["easy", "medium", "hard"]
    rarity: Literal["common", "rare"]
    target_count: int
    contrastive_pair_required: bool = False


class QuotaPlan(BaseModel):
    mode: Literal["pilot", "full"]
    num_per_dimension: int
    buckets: List[QuotaBucket] = Field(default_factory=list)


# ============================================================
# Sampled recipe wrapper（采样后的中间态）
# ============================================================

class SampledRecipe(BaseModel):
    """Phase 1 CandidateRecipe + Phase 2 配额上下文，便于后续模块独立读取。"""

    bucket_id: str
    dimension: str
    input_mode: InputMode
    subtype: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    semantic_rarity: Literal["common", "rare"] = "common"
    contrastive_pair_id: Optional[str] = None
    contrastive_role: ContrastiveRole = "original"
    recipe: Dict[str, Any] = Field(default_factory=dict)  # 原始 CandidateRecipe.dict()
