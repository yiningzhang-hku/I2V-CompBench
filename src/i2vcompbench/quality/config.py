"""Quality experiments configuration loading and validation.

Loads configs/quality_experiments.yaml and provides:
- Pydantic v2 models for type-safe access
- Post-load validation: ratio sums, path existence, method enums, numeric ranges
- CLI override merging
- Run ID generation when null
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# Section models
# ============================================================

class RunConfig(BaseModel):
    """Run-level configuration."""
    run_id: Optional[str] = None
    seed: int = 20260712
    output_root: str = "data/benchmark_dataset/quality_experiments"
    overwrite: bool = False


class InputConfig(BaseModel):
    """Input paths and formal dimension list."""
    benchmark_root: str = "data/benchmark_dataset"
    phase1_bundle_dir: Optional[str] = None
    source_asset_manifest: Optional[str] = None
    asset_lineage_manifest: Optional[str] = None
    allow_legacy_qid_search: bool = False
    manifest: str = "phase3_manifest.jsonl"
    question_plans: str = "question_plans.jsonl"
    final_prompts: str = "prompts/final_prompts.jsonl"
    formal_dimensions: List[str] = Field(default_factory=lambda: [
        "attribute_binding",
        "action_binding",
        "motion_binding",
        "background_dynamics",
        "view_transformation",
    ])

    @field_validator("formal_dimensions")
    @classmethod
    def validate_dimensions(cls, v: List[str]) -> List[str]:
        allowed = {
            "attribute_binding", "action_binding", "motion_binding",
            "background_dynamics", "view_transformation",
        }
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(
                f"formal_dimensions contains invalid entries: {invalid}. "
                f"Allowed: {sorted(allowed)}"
            )
        return v


class SplitConfig(BaseModel):
    """Development/validation split configuration."""
    development_per_dimension: int = 50
    validation_per_dimension: int = 50
    stratify_by: List[str] = Field(default_factory=lambda: ["difficulty_old", "semantic_rarity"])


class TargetRepairConfig(BaseModel):
    """Target repair configuration."""
    prefer_phase1_rebuild: bool = True
    allow_vlm_migration: bool = True
    min_confidence: float = 0.80
    manual_review_below: float = 0.90

    @model_validator(mode="after")
    def check_confidence_order(self) -> "TargetRepairConfig":
        if self.min_confidence > self.manual_review_below:
            raise ValueError(
                f"min_confidence ({self.min_confidence}) must be <= "
                f"manual_review_below ({self.manual_review_below})"
            )
        return self


class PromptConfig(BaseModel):
    """Prompt quality experiment configuration."""
    min_words: int = 8
    max_words: int = 25
    zipf_threshold: float = 3.5
    do_not_replace_pos: List[str] = Field(default_factory=lambda: ["NOUN", "PROPN"])
    max_llm_retries: int = 3
    semantic_similarity_min: float = 0.95
    human_naturalness_min: float = 4.0
    optional_perplexity_threshold: Optional[float] = None
    fallback_development_size: int = 60
    fallback_validation_size: int = 57

    @model_validator(mode="after")
    def check_word_range(self) -> "PromptConfig":
        if self.min_words >= self.max_words:
            raise ValueError(
                f"min_words ({self.min_words}) must be < max_words ({self.max_words})"
            )
        return self


class ClarityConfig(BaseModel):
    """Image clarity experiment configuration."""
    sample_size: int = 120
    validation_sample_size: int = 120
    face_subset_size: int = 40
    target_long_edge: int = 854
    output_format: Literal["PNG", "JPEG", "WEBP"] = "PNG"
    output_color_space: str = "sRGB"
    methods: List[str] = Field(
        default_factory=lambda: ["lanczos", "unsharp", "realesrgan", "swinir"]
    )
    optional_methods: List[str] = Field(default_factory=lambda: [
        "realesrgan_gfpgan", "realesrgan_unsharp",
        "realesrgan_clahe", "realesrgan_unsharp_clahe",
    ])
    video_probe_size: int = 30
    dino_subject_noninferiority_margin: Optional[float] = None
    face_identity_noninferiority_margin: Optional[float] = None

    @field_validator("methods")
    @classmethod
    def validate_core_methods(cls, v: List[str]) -> List[str]:
        known = {
            "lanczos", "unsharp", "realesrgan", "swinir",
            "realesrgan_gfpgan", "realesrgan_unsharp",
            "realesrgan_clahe", "realesrgan_unsharp_clahe",
        }
        invalid = set(v) - known
        if invalid:
            raise ValueError(f"Unknown clarity methods: {invalid}. Known: {sorted(known)}")
        return v


class HybridConfig(BaseModel):
    """Hybrid aspect ratio adaptation sub-config."""
    minor_ratio_error: float = 0.04
    saliency_ratio_error: float = 0.20
    aggressive_saliency_ratio_error: float = 0.30


class AspectConfig(BaseModel):
    """Aspect ratio experiment configuration."""
    ratio_stage_sample_size: int = 60
    ratio_validation_sample_size: int = 60
    strategy_stage_sample_size: int = 120
    strategy_validation_sample_size: int = 120
    ratios: List[str] = Field(default_factory=lambda: ["4:3", "16:9"])
    common_pixel_budget: Optional[int] = None
    target_sizes: Dict[str, Optional[Any]] = Field(default_factory=lambda: {"4:3": None, "16:9": None})
    size_multiple: int = 16
    max_pixel_budget_relative_error: float = 0.02
    methods: List[str] = Field(default_factory=lambda: [
        "stretch", "center_crop", "letterbox",
        "blur_padding", "saliency_crop", "hybrid_conservative",
    ])
    optional_methods: List[str] = Field(default_factory=lambda: ["outpainting", "hybrid_aggressive"])
    subject_retention_noninferiority_margin: Optional[float] = None
    scene_retention_noninferiority_margin: Optional[float] = None
    hybrid: HybridConfig = Field(default_factory=HybridConfig)

    @field_validator("methods")
    @classmethod
    def validate_aspect_methods(cls, v: List[str]) -> List[str]:
        known = {
            "stretch", "center_crop", "letterbox", "blur_padding",
            "saliency_crop", "hybrid_conservative",
            "outpainting", "hybrid_aggressive",
        }
        invalid = set(v) - known
        if invalid:
            raise ValueError(f"Unknown aspect methods: {invalid}. Known: {sorted(known)}")
        return v


class SubjectTierConfig(BaseModel):
    """Subject tier distribution configuration."""
    ratios: Dict[str, float] = Field(default_factory=lambda: {
        "T1_common": 0.55,
        "T2_longtail": 0.28,
        "T3_finegrained": 0.12,
        "T4_rare_fictional": 0.05,
    })

    @field_validator("ratios")
    @classmethod
    def validate_ratios_sum(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(
                f"subject_tier.ratios must sum to 1.0, got {total:.4f}"
            )
        expected_keys = {"T1_common", "T2_longtail", "T3_finegrained", "T4_rare_fictional"}
        missing = expected_keys - set(v.keys())
        if missing:
            raise ValueError(f"subject_tier.ratios missing keys: {missing}")
        return v


class DifficultyConfig(BaseModel):
    """Difficulty labeling configuration."""
    task_weights: Dict[str, float] = Field(default_factory=lambda: {
        "target_complexity": 0.35,
        "subject_localization": 0.30,
        "background_interference": 0.20,
        "temporal_complexity": 0.15,
    })
    initial_thresholds: List[float] = Field(default_factory=lambda: [0.33, 0.66])
    report_semantic_rarity_separately: bool = True
    judgeability_review_threshold: float = 0.70
    feature_min_confidence: float = 0.80
    prediction_manual_review_below: float = 0.60
    allow_vlm_for_missing_features: bool = True

    @field_validator("task_weights")
    @classmethod
    def validate_weights_sum(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(
                f"difficulty.task_weights must sum to 1.0, got {total:.4f}"
            )
        return v

    @field_validator("initial_thresholds")
    @classmethod
    def validate_thresholds(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError("initial_thresholds must have exactly 2 values")
        if v[0] >= v[1]:
            raise ValueError(
                f"initial_thresholds must be ascending, got {v}"
            )
        if not all(0.0 < t < 1.0 for t in v):
            raise ValueError("initial_thresholds values must be in (0, 1)")
        return v


# ============================================================
# Top-level config
# ============================================================

class QualityConfig(BaseModel):
    """Top-level quality experiments configuration.

    Loaded from configs/quality_experiments.yaml and validated.
    """
    run: RunConfig = Field(default_factory=RunConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    target_repair: TargetRepairConfig = Field(default_factory=TargetRepairConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    clarity: ClarityConfig = Field(default_factory=ClarityConfig)
    aspect: AspectConfig = Field(default_factory=AspectConfig)
    subject_tier: SubjectTierConfig = Field(default_factory=SubjectTierConfig)
    difficulty: DifficultyConfig = Field(default_factory=DifficultyConfig)

    @model_validator(mode="after")
    def resolve_run_id(self) -> "QualityConfig":
        """Generate run_id if null."""
        if self.run.run_id is None:
            self.run.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self


# ============================================================
# Loading functions
# ============================================================

DEFAULT_CONFIG_PATH = "configs/quality_experiments.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_quality_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> QualityConfig:
    """Load and validate quality experiments configuration.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML config file. Defaults to DEFAULT_CONFIG_PATH relative
        to project_root.
    overrides : dict, optional
        CLI overrides to deep-merge on top of YAML values.
    project_root : Path, optional
        Project root for resolving relative paths. Defaults to cwd.

    Returns
    -------
    QualityConfig
        Fully validated configuration object.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    ValueError
        If validation fails (ratio sums, numeric ranges, method enums, etc.).
    """
    if project_root is None:
        project_root = Path.cwd()

    if config_path is None:
        config_path = str(project_root / DEFAULT_CONFIG_PATH)

    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / path

    if not path.exists():
        raise FileNotFoundError(f"Quality config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Apply CLI overrides
    if overrides:
        raw = _deep_merge(raw, overrides)

    # Parse and validate via Pydantic
    cfg = QualityConfig.model_validate(raw)

    # Post-validation: check input paths exist (relative to project_root)
    _validate_input_paths(cfg, project_root)

    logger.info(
        f"Loaded quality config from {path} | run_id={cfg.run.run_id} | "
        f"seed={cfg.run.seed} | dimensions={cfg.input.formal_dimensions}"
    )
    return cfg


def _validate_input_paths(cfg: QualityConfig, project_root: Path) -> None:
    """Validate that critical input paths exist.

    Checks benchmark_root, manifest, question_plans, final_prompts.
    phase1_bundle_dir is allowed to be null (reported as warning).
    """
    benchmark = project_root / cfg.input.benchmark_root
    if not benchmark.exists():
        raise FileNotFoundError(
            f"benchmark_root does not exist: {benchmark}"
        )

    # Check manifest file
    manifest_path = benchmark / cfg.input.manifest
    if not manifest_path.exists():
        logger.warning(f"Manifest file not found: {manifest_path}")

    # Check question_plans
    qp_path = benchmark / cfg.input.question_plans
    if not qp_path.exists():
        logger.warning(f"question_plans file not found: {qp_path}")

    # Check final_prompts
    fp_path = benchmark / cfg.input.final_prompts
    if not fp_path.exists():
        logger.warning(f"final_prompts file not found: {fp_path}")

    # Phase 1 bundle: warn if null
    if cfg.input.phase1_bundle_dir is None:
        logger.warning(
            "phase1_bundle_dir is null. Target repair with Phase 1 rebuild "
            "will not be available. Set allow_vlm_migration=true or provide bundle path."
        )
    else:
        bundle_path = Path(cfg.input.phase1_bundle_dir)
        if not bundle_path.is_absolute():
            bundle_path = project_root / bundle_path
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"phase1_bundle_dir does not exist: {bundle_path}"
            )

    # Asset manifests: warn if null (required before image experiments)
    if cfg.input.source_asset_manifest is None:
        logger.warning(
            "source_asset_manifest is null. Image experiments will be blocked "
            "until asset binding is established."
        )
    if cfg.input.asset_lineage_manifest is None:
        logger.warning(
            "asset_lineage_manifest is null. Image experiments will be blocked "
            "until lineage is established."
        )
