"""
Pydantic v2 schema definitions for TIP-I2V data analysis pipeline.
All data structures used across pipeline steps are defined here.
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ============================================================
# Manifest
# ============================================================

class ManifestItem(BaseModel):
    sample_id: str
    prompt_text: str              # 原始 prompt（含 Pika 参数）
    clean_prompt_text: str = ""   # 清洗后的 prompt（去除 Pika 参数）
    image_path: str
    video_path: Optional[str] = None
    subject_field: Optional[str] = None
    pika_camera: Optional[str] = None     # 从 prompt 中提取的 Pika 运镜参数
    pika_motion: Optional[int] = None     # 从 prompt 中提取的 motion 级别
    status: str  # "ok" / "missing_image" / "empty_prompt" / "bad_format"
    error_detail: Optional[str] = None


# ============================================================
# Image Analysis — Nested Models
# ============================================================

class SubjectAttributes(BaseModel):
    color: List[str] = Field(default_factory=list)
    size: Optional[str] = None
    material_texture: List[str] = Field(default_factory=list)
    state: List[str] = Field(default_factory=list)
    wearing: List[str] = Field(default_factory=list)


class SubjectInfo(BaseModel):
    id: str
    name: str
    instance_description: str = ""
    count: int = 1
    attributes: SubjectAttributes = Field(default_factory=SubjectAttributes)
    current_pose_action: str = ""
    position_in_frame: str = ""
    is_animate: bool = False
    # ---- Phase 1 扩展：实例几何（P0 由 mock_geometry 从 position_in_frame 反推 bbox） ----
    instance_id: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # 归一化 [x0,y0,x1,y1]
    mask_path: Optional[str] = None
    segmentation_quality: str = "low"          # high / medium / low / unavailable
    tracking_feasibility: str = "medium"       # easy / medium / hard / infeasible


class SubjectRelation(BaseModel):
    subject_a: str
    predicate: str
    subject_b: str


class BackgroundElement(BaseModel):
    name: str
    type: str  # "rigid" / "potentially_dynamic"
    current_state: str = ""
    region: str = ""


class BackgroundInfo(BaseModel):
    elements: List[BackgroundElement] = Field(default_factory=list)
    lighting: str = ""
    weather: str = ""
    time_of_day: str = ""
    foreground_background_separability: str = ""  # high / medium / low
    rigid_background_ratio: str = ""  # high / medium / low


class CameraBaseline(BaseModel):
    shot_type: str = ""
    framing: str = ""
    camera_angle: str = ""
    estimated_depth: str = ""
    has_rigid_reference_structure: bool = False
    scene_depth_complexity: str = ""  # simple / medium / complex


class ImageAnalysisResult(BaseModel):
    sample_id: str
    subjects: List[SubjectInfo] = Field(default_factory=list)
    subject_count: int = 0
    has_multiple_subjects: bool = False
    has_same_category_instances: bool = False
    subjects_clearly_distinguishable: bool = True
    subject_relations: List[SubjectRelation] = Field(default_factory=list)
    background: BackgroundInfo = Field(default_factory=BackgroundInfo)
    camera_baseline: CameraBaseline = Field(default_factory=CameraBaseline)
    raw_vlm_output: str = ""
    parse_success: bool = True
    parse_error: Optional[str] = None
    # ---- Phase 1 扩展：参考银行抽取潜力 ----
    reference_potential: Dict = Field(default_factory=dict)
    # 形如 {"subject": True, "attribute": True, "object": False,
    #       "scene_reference_original": True, "scene_reference_inpainted": False,
    #       "reason": "..."}


# ============================================================
# Text Analysis — Nested Models
# ============================================================

class AttributeChangeSlot(BaseModel):
    target_subject: str
    attribute_type: str  # color / texture / state / wearing / size
    from_value: Optional[str] = None
    to_value: str


class ActionSlot(BaseModel):
    target_subject: str
    action_verb: str
    action_detail: str = ""


class MotionSlot(BaseModel):
    target_subject: str
    direction: str  # leftward / rightward / upward / downward / toward_camera / away_from_camera / forward / backward
    motion_phrase: str = ""


class SpatialRelationSlot(BaseModel):
    subject_a: str
    target_predicate: str
    subject_b: str


class BackgroundChangeSlot(BaseModel):
    target_region: str
    change_type: str  # weather / lighting / state / texture / phenomenon
    from_state: Optional[str] = None
    to_state: str


class CameraMovementSlot(BaseModel):
    command: str  # zoom_in / zoom_out / pan_left / pan_right / tilt_up / tilt_down / push_in / pull_out / orbit / tracking / static_camera / closer_view / wider_view
    target_subject: Optional[str] = None
    speed_modifier: Optional[str] = None
    framing_constraint: Optional[str] = None


class TextAnalysisResult(BaseModel):
    sample_id: str
    prompt_text: str
    primary_intent: str  # subject_focused / background_focused / camera_focused / mixed / ambiguous
    subject_sub_intent: Optional[str] = None
    background_sub_intent: Optional[str] = None
    camera_sub_intent: Optional[str] = None
    involves_attribute_change: bool = False
    involves_action: bool = False
    involves_directed_motion: bool = False
    involves_spatial_relation_change: bool = False
    involves_background_change: bool = False
    involves_camera_movement: bool = False
    attribute_change_slots: List[AttributeChangeSlot] = Field(default_factory=list)
    action_slots: List[ActionSlot] = Field(default_factory=list)
    motion_slots: List[MotionSlot] = Field(default_factory=list)
    spatial_relation_slots: List[SpatialRelationSlot] = Field(default_factory=list)
    background_change_slots: List[BackgroundChangeSlot] = Field(default_factory=list)
    camera_movement_slots: List[CameraMovementSlot] = Field(default_factory=list)
    nouns: List[str] = Field(default_factory=list)
    verbs: List[str] = Field(default_factory=list)
    adjectives: List[str] = Field(default_factory=list)
    raw_llm_output: str = ""
    parse_success: bool = True
    parse_error: Optional[str] = None
    # ---- Phase 1 扩展：第 7 维 interaction + 维度路由 + 防泄漏 ----
    involves_interaction: bool = False
    interaction_slots: List[Dict] = Field(default_factory=list)
    # InteractionSlot 字段：interaction_type / agent_subject / patient_subject /
    # relation_phrase / expected_outcome
    primary_dimension: Optional[str] = None       # ⊂ DIMENSIONS_V2
    candidate_dimensions: List[str] = Field(default_factory=list)
    forbidden_dimension_leakage: List[str] = Field(default_factory=list)
    transform_ontology_confidence: float = 0.0
    routing_reason: str = ""


# ============================================================
# Joint Analysis
# ============================================================

class DimensionEvaluability(BaseModel):
    text_requests: bool = False
    image_supports: bool = False
    evaluable: bool = False
    reason: str = ""


class JointAnalysisResult(BaseModel):
    sample_id: str
    dim_attribute_binding: DimensionEvaluability = Field(default_factory=DimensionEvaluability)
    dim_motion_binding: DimensionEvaluability = Field(default_factory=DimensionEvaluability)
    dim_spatial_relation: DimensionEvaluability = Field(default_factory=DimensionEvaluability)
    dim_action_binding: DimensionEvaluability = Field(default_factory=DimensionEvaluability)
    dim_scene_dynamics: DimensionEvaluability = Field(default_factory=DimensionEvaluability)
    dim_camera_transformation: DimensionEvaluability = Field(default_factory=DimensionEvaluability)
    # ---- Phase 1 扩展：第 7 维 ----
    dim_interaction_reasoning: DimensionEvaluability = Field(default_factory=DimensionEvaluability)
    text_target_subjects: List[str] = Field(default_factory=list)
    image_detected_subjects: List[str] = Field(default_factory=list)
    subject_overlap: List[str] = Field(default_factory=list)
    subject_overlap_ratio: float = 0.0


# ============================================================
# Candidate Pool (legacy, kept for backward compatibility)
# ============================================================

class CandidatePoolItem(BaseModel):
    sample_id: str
    dimension: str
    selected_reason: str = ""
    text_evidence: str = ""
    image_evidence: str = ""
    confidence: float = 0.0


# ============================================================
# Prior Package — 先验包数据模型
# ============================================================

class ConceptDistribution(BaseModel):
    """某个概念类型的频率分布（如主体类型、动词类型等）"""
    category: str                # e.g. "subject_category", "action_verb"
    entries: List[dict] = Field(default_factory=list)  # [{"name": "person", "count": 50, "pct": 23.5}, ...]
    total_samples: int = 0


class VisualCompositionPrior(BaseModel):
    """视觉构成先验 — 描述某维度适合的首帧图像特征分布"""
    subject_count_distribution: List[dict] = Field(default_factory=list)
    scene_type_distribution: List[dict] = Field(default_factory=list)     # lighting + weather + time
    shot_type_distribution: List[dict] = Field(default_factory=list)
    camera_angle_distribution: List[dict] = Field(default_factory=list)
    background_separability_distribution: List[dict] = Field(default_factory=list)
    typical_subject_categories: List[dict] = Field(default_factory=list)


class SeedExample(BaseModel):
    """种子样例 — 完整标注的高质量示例，供 LLM few-shot"""
    sample_id: str
    original_prompt: str
    clean_prompt: str
    dimension: str
    text_slots: dict = Field(default_factory=dict)           # 该维度的语义槽位
    image_subjects: List[dict] = Field(default_factory=list)  # 图像主体摘要
    image_background: dict = Field(default_factory=dict)      # 图像背景摘要
    image_camera: dict = Field(default_factory=dict)          # 图像运镜基线
    selection_reason: str = ""
    confidence: float = 0.0


class DimensionPrior(BaseModel):
    """单个维度的完整先验包"""
    dimension: str
    display_name: str
    sample_count: int = 0              # 该维度可评估样本数
    coverage_pct: float = 0.0          # 覆盖率
    # 概念分布
    concept_distributions: List[ConceptDistribution] = Field(default_factory=list)
    # 视觉构成先验
    visual_prior: VisualCompositionPrior = Field(default_factory=VisualCompositionPrior)
    # 句式模板
    structural_templates: List[dict] = Field(default_factory=list)  # [{"pattern": "...", "count": N}, ...]
    # 种子样例
    seed_examples: List[SeedExample] = Field(default_factory=list)
    # 维度约束规则
    constraints: dict = Field(default_factory=dict)


class PriorPackage(BaseModel):
    """完整的 Prior Package — 所有维度的先验集合"""
    dataset_name: str = "TIP-I2V"
    split: str = ""
    total_samples: int = 0
    clean_samples: int = 0
    analyzed_samples: int = 0
    # 全局概念分布
    global_distributions: List[ConceptDistribution] = Field(default_factory=list)
    # 全局视觉构成先验
    global_visual_prior: VisualCompositionPrior = Field(default_factory=VisualCompositionPrior)
    # Pika 参数分布（运镜维度的真实用户偏好）
    pika_camera_distribution: List[dict] = Field(default_factory=list)
    pika_motion_distribution: List[dict] = Field(default_factory=list)
    # 6 维度的独立先验
    dimension_priors: List[DimensionPrior] = Field(default_factory=list)
    # 维度共现矩阵
    dimension_cooccurrence: dict = Field(default_factory=dict)
