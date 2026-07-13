"""
I2V-CompBench Quality Experiment Package.

Provides schemas, path utilities, and hashing tools for the
quality assurance pipeline (§6.0–§6.7).
"""

from i2vcompbench.quality.config import (
    QualityConfig,
    load_quality_config,
)
from i2vcompbench.quality.hashing import (
    bytes_sha256,
    canonical_json_sha256,
    file_sha256,
    stable_order_key,
)
from i2vcompbench.quality.paths import (
    ensure_run_dirs,
    resolve_image_path,
    run_output_dir,
    to_posix,
)
from i2vcompbench.quality.schemas import (
    AdjudicationRecord,
    AnnotationBase,
    AssetLineageRecord,
    DifficultyAnnotation,
    DifficultyFeatureRecord,
    FinalReviewAnnotation,
    HumanAnnotation,
    ImageAnnotation,
    ImageVariantResult,
    PromptAnnotation,
    PromptVariantResult,
    QualityCandidate,
    SelectionDecision,
    SourceAssetRecord,
    TargetRepairAnnotation,
    TargetRepairResult,
)

__all__ = [
    # config
    "QualityConfig",
    "load_quality_config",
    # schemas
    "SourceAssetRecord",
    "AssetLineageRecord",
    "QualityCandidate",
    "TargetRepairResult",
    "PromptVariantResult",
    "ImageVariantResult",
    "AnnotationBase",
    "TargetRepairAnnotation",
    "PromptAnnotation",
    "ImageAnnotation",
    "DifficultyAnnotation",
    "FinalReviewAnnotation",
    "HumanAnnotation",
    "AdjudicationRecord",
    "SelectionDecision",
    "DifficultyFeatureRecord",
    # hashing
    "file_sha256",
    "bytes_sha256",
    "canonical_json_sha256",
    "stable_order_key",
    # paths
    "to_posix",
    "resolve_image_path",
    "run_output_dir",
    "ensure_run_dirs",
]
