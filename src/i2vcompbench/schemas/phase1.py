"""
Phase 1 扩展 Pydantic schema —— 在不破坏既有 schema.py 的前提下新增以下数据结构：

1. 第 7 维 interaction_reasoning 槽位：InteractionSlot
2. 维度路由与防泄漏：TransformOntology / DimensionIsolation
3. 实例级几何：InstanceGeometry（bbox/mask/segmentation_quality）
4. 对齐产物：AlignedSample
5. 可评估性诊断：EvaluatorFeasibility
6. 资产银行：ReferenceAsset / AssetQuality / Provenance
7. 候选 recipe：CandidateRecipe / ContrastiveSpec / PreserveConstraint

7 维度统一命名（与合作者版总览 / Phase1_先验数据准备.md 保持一致）：
    attribute_binding / action_binding / motion_binding / spatial_composition /
    background_dynamics / view_transformation / interaction_reasoning
"""

from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

# ============================================================
# 维度常量（Phase 1 全程使用）
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

# 旧 → 新 维度命名映射（用于把 step4 的 6 维结果迁移到 7 维语义）
LEGACY_TO_V2_DIM: Dict[str, str] = {
    "dim_attribute_binding": "attribute_binding",
    "dim_action_binding": "action_binding",
    "dim_motion_binding": "motion_binding",
    "dim_spatial_relation": "spatial_composition",
    "dim_scene_dynamics": "background_dynamics",
    "dim_camera_transformation": "view_transformation",
}


# ============================================================
# 第 7 维：interaction_reasoning
# ============================================================

class InteractionSlot(BaseModel):
    """
    交互推理槽位 —— 描述多主体协同 / 因果链。
    interaction_type:
        - cooperative_action：协同动作（A 抱起 B / A 递给 B 物品）
        - causal_chain：因果链（A 推 B → B 倒）
        - mutual_motion：相互运动（A 与 B 相向 / 相背）
        - contact_event：接触事件（A 触碰 B）
    """
    interaction_type: Literal[
        "cooperative_action",
        "causal_chain",
        "mutual_motion",
        "contact_event",
    ]
    agent_subject: str           # 发起者
    patient_subject: str         # 接收者 / 共同参与者
    relation_phrase: str = ""    # 原文片段（如 "hands the cup to"）
    expected_outcome: str = ""   # 期望结果（如 "cup ends up with patient"）


# ============================================================
# 变换本体 + 维度路由 + 防泄漏
# ============================================================

class TransformOntology(BaseModel):
    """
    把 prompt 涉及的变化映射到 7 维空间，支持多维并存。
    primary_dimension：唯一主维度（用于决定 recipe.target_dimension）。
    candidate_dimensions：所有命中的维度（用于审计与共现统计）。
    """
    primary_dimension: Optional[str] = None
    candidate_dimensions: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    routing_reason: str = ""


class DimensionIsolation(BaseModel):
    """
    防止其他维度污染主维度的硬约束。
    例如：当 primary_dimension = attribute_binding 时，
    forbidden_dimensions 应包含 motion_binding / view_transformation 等，
    用于在 Phase 2 生成时禁止往 prompt 中写入这些维度的指令。
    """
    primary_dimension: str
    forbidden_dimensions: List[str] = Field(default_factory=list)
    leakage_risk_notes: str = ""


# ============================================================
# 实例级几何（P0 mock，未来可替换为 SAM / Grounding-DINO）
# ============================================================

class InstanceGeometry(BaseModel):
    """
    实例几何信息。P0 阶段由 mock_geometry 从 position_in_frame（9 宫格）映射出 bbox。
    bbox 采用归一化坐标 [x0, y0, x1, y1] ∈ [0, 1]。
    mask_path 在 P0 留空（不做真实分割）。
    """
    instance_id: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    mask_path: Optional[str] = None
    segmentation_quality: Literal[
        "high",          # SAM 输出 + 人工校验
        "medium",        # SAM 输出 / VLM bbox + 简单 refine
        "low",           # 仅 9 宫格 mock
        "unavailable",   # 无几何信息
    ] = "low"
    tracking_feasibility: Literal["easy", "medium", "hard", "infeasible"] = "medium"


# ============================================================
# 对齐产物
# ============================================================

class EvaluatorFeasibility(BaseModel):
    """
    某一维度在该样本上的可评测性诊断。
    tool_status 与三阶段总览 §5.4 对齐：valid / low_confidence / tool_uncertain / invalid_input
    """
    dimension: str
    feasible: bool = False
    tool_status: Literal[
        "valid", "low_confidence", "tool_uncertain", "invalid_input"
    ] = "tool_uncertain"
    reasons: List[str] = Field(default_factory=list)
    required_tools: List[str] = Field(default_factory=list)


class AlignedSample(BaseModel):
    """
    Phase 1 Step 4（align_instances）的输出。
    把图像主体 ↔ 文本目标主体对齐到同一 instance_id 空间。
    """
    sample_id: str
    aligned_subjects: List[Dict] = Field(default_factory=list)
    # 每项：{"instance_id": "...", "image_subject_id": "subj_0",
    #        "text_subject_name": "...", "alignment_confidence": 0.0,
    #        "geometry": InstanceGeometry.dict()}
    unmatched_text_subjects: List[str] = Field(default_factory=list)
    unmatched_image_subjects: List[str] = Field(default_factory=list)
    overlap_ratio: float = 0.0
    evaluator_feasibility: List[EvaluatorFeasibility] = Field(default_factory=list)
    dimension_isolation: Optional[DimensionIsolation] = None


# ============================================================
# Reference Bank
# ============================================================

class Provenance(BaseModel):
    """资产来源 —— 审计必填，用于追溯每张图的产生路径。"""
    source_sample_id: str          # 来源 manifest sample_id
    source_image_path: str         # 原始图像路径
    extraction_method: Literal[
        "original",                 # 原图直接复用
        "mock_crop_9grid",          # P0 9 宫格 mock 裁剪
        "bbox_crop",                # 真实 bbox 裁剪
        "sam_mask_crop",            # SAM 分割后裁剪
        "inpaint_remove_subject",   # 主体抹除 inpainting
    ]
    extraction_params: Dict = Field(default_factory=dict)  # bbox / 9grid cell 等参数
    created_by: str = "phase1"
    created_at: str = ""


class AssetQuality(BaseModel):
    """资产质量字段 —— 用于 Phase 2 筛选。"""
    resolution: Optional[Tuple[int, int]] = None
    aspect_ratio: Optional[float] = None
    is_clean_background: bool = False
    has_visible_subject: bool = True
    quality_score: float = 0.0  # [0, 1]
    quality_flags: List[str] = Field(default_factory=list)


class ReferenceAsset(BaseModel):
    """
    Reference Bank 五类资产之一：
      subject / attribute / object / scene_reference_original / scene_reference_inpainted
    usable_for：该 asset 自身适用的维度（与三阶段总览 §3.3 Step 5 修订一致）。
    """
    asset_id: str
    asset_type: Literal[
        "subject",
        "attribute",
        "object",
        "scene_reference_original",
        "scene_reference_inpainted",
    ]
    asset_path: str
    semantic_tags: List[str] = Field(default_factory=list)
    usable_for: List[str] = Field(default_factory=list)  # 子集 ⊂ DIMENSIONS_V2
    quality: AssetQuality = Field(default_factory=AssetQuality)
    provenance: Provenance


# ============================================================
# Candidate Recipe
# ============================================================

class PreserveConstraint(BaseModel):
    """需要保持不变的语义元素。"""
    target: str                 # 主体名 / 属性名 / 区域名
    aspect: Literal[
        "identity", "attribute_color", "attribute_material",
        "spatial_position", "scene_context", "camera_framing",
    ]
    note: str = ""


class ContrastiveSpec(BaseModel):
    """对比组规格（baseline 系列由 Phase 2 生成时使用）。

    5 类 baseline（§5.6.7）：
      - static_copy / random_motion / global_filter / camera_pan_cheat：无条件默认
      - subject_swap：仅 spatial_composition / interaction_reasoning 条件追加
    """
    contrast_type: Literal[
        "static_copy", "random_motion", "global_filter", "camera_pan_cheat",
        "subject_swap",
    ]
    description: str = ""


class CandidateRecipe(BaseModel):
    """
    Phase 1 Step 7（build_recipes）输出。
    一个 recipe = 一条候选评测题的“半成品配方”，由 Phase 2 决定是否落地为最终样本。
    """
    recipe_id: str
    source_sample_id: str
    target_dimension: str  # ⊂ DIMENSIONS_V2
    input_mode: Literal["single_image", "multi_image"]
    source_type: Literal[
        "observed_single_image",
        "derived_single_image",
        "derived_multi_reference",
        "external_multi_reference",
    ]
    reference_asset_ids: List[str] = Field(default_factory=list)
    base_prompt_draft: str = ""
    preserve_constraints: List[PreserveConstraint] = Field(default_factory=list)
    contrastive_spec: List[ContrastiveSpec] = Field(default_factory=list)
    dimension_isolation: Optional[DimensionIsolation] = None
    expected_difficulty: Literal["easy", "medium", "hard"] = "medium"
    # §5.6.7 要求的额外字段（Phase 2 配额匹配依赖）
    subtype: str = ""
    semantic_rarity: Literal["common", "rare"] = "common"
    quality_flags: List[str] = Field(default_factory=list)
    contrastive_pair_id: Optional[str] = None
    contrastive_role: Literal[
        "original",
        "baseline_static_copy",
        "baseline_random_motion",
        "baseline_global_filter",
        "baseline_camera_pan_cheat",
        "baseline_subject_swap",
    ] = "original"
    notes: str = ""
