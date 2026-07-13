"""Tests for i2vcompbench.quality.schemas module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pytest
from pydantic import ValidationError

from i2vcompbench.quality.schemas import (
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
from i2vcompbench.schemas.phase2 import PreserveItem, SubjectRef, TargetRelation


class TestSchemaInstantiation:
    """All schemas can be instantiated with valid data."""

    def test_source_asset_record(self):
        rec = SourceAssetRecord(
            upstream_asset_id="ua001",
            source_sample_id="ss001",
            asset_role="source_first_frame",
            upstream_path="images/test.png",
            canonical_upstream_sha256="abc123" * 10 + "ab",
            width=1920,
            height=1080,
            source_manifest_path="manifest.jsonl",
            source_manifest_sha256="def456" * 10 + "de",
            verification_source="phase1_manifest",
        )
        assert rec.upstream_asset_id == "ua001"
        assert rec.asset_role == "source_first_frame"

    def test_quality_candidate(self):
        qc = QualityCandidate(
            question_id="attr_single_0001",
            dimension="attribute_binding",
            difficulty_old="easy",
            semantic_rarity="common",
            prompt="The elephant lifts its trunk upward.",
            first_frame_path="data/first_frames/test.png",
            source_manifest_hash="abc123",
        )
        assert qc.question_id == "attr_single_0001"
        assert qc.dimension == "attribute_binding"

    def test_prompt_variant_result(self):
        pvr = PromptVariantResult(
            question_id="test001",
            method="A0",
            prompt_before="old prompt",
            prompt_after="new prompt",
            word_count=5,
            rare_modifier_hits=[],
            structural_issues=[],
            status="pass",
        )
        assert pvr.method == "A0"
        assert pvr.status == "pass"

    def test_image_variant_result(self):
        ivr = ImageVariantResult(
            question_id="test001",
            experiment="clarity",
            method="lanczos",
            source_path="source.png",
            output_path="output.png",
            upstream_asset_id="ua001",
            canonical_upstream_sha256="abc" * 20 + "ab",
            input_asset_sha256="def" * 20 + "de",
            input_stage="canonical_source",
            output_sha256="ghi" * 20 + "gh",
            asset_binding_verified=True,
            source_size=(480, 360),
            output_size=(854, 480),
            metrics={"psnr": 35.2},
            status="pass",
        )
        assert ivr.experiment == "clarity"

    def test_selection_decision(self):
        sd = SelectionDecision(
            question_id="test001",
            dimension="action_binding",
            eligible=True,
            blocking_reasons=[],
            prompt_method="A0",
            clarity_method="lanczos",
            aspect_method="center_crop",
            native_first_frame_path="path.png",
            native_first_frame_sha256="abc123",
            inference_companion_path="comp.png",
            inference_companion_sha256="def456",
            subject_tier="T1_common",
            difficulty_new="easy",
            quality_rank_components={"prompt_score": 0.9},
            quality_rank_score=0.85,
            selection_order_key="sha256key",
        )
        assert sd.eligible is True

    def test_difficulty_feature_record(self):
        dfr = DifficultyFeatureRecord(
            question_id="test001",
            target_complexity=0.5,
            subject_localization=0.6,
            background_interference=0.3,
            temporal_complexity=0.4,
            initial_state_visibility=0.7,
            target_observability=0.8,
            alternative_explanation_risk=0.2,
            evaluator_support_gap=0.1,
            d_task=0.55,
            d_judge=0.45,
            evidence={"target_complexity": ["verb is rare"]},
            feature_sources={"target_complexity": "rule"},
            feature_confidence={"target_complexity": 0.9},
            status="pass",
        )
        assert dfr.d_task == 0.55


class TestSchemaValidation:
    """Required fields and Literal constraints are enforced."""

    def test_quality_candidate_missing_required(self):
        with pytest.raises(ValidationError):
            QualityCandidate()  # type: ignore

    def test_quality_candidate_invalid_dimension(self):
        with pytest.raises(ValidationError):
            QualityCandidate(
                question_id="test",
                dimension="invalid_dim",  # not in Literal
                difficulty_old="easy",
                semantic_rarity="common",
                prompt="test",
                first_frame_path="test.png",
                source_manifest_hash="abc",
            )

    def test_quality_candidate_invalid_difficulty(self):
        with pytest.raises(ValidationError):
            QualityCandidate(
                question_id="test",
                dimension="attribute_binding",
                difficulty_old="impossible",  # not in Literal
                semantic_rarity="common",
                prompt="test",
                first_frame_path="test.png",
                source_manifest_hash="abc",
            )

    def test_prompt_variant_invalid_method(self):
        with pytest.raises(ValidationError):
            PromptVariantResult(
                question_id="test",
                method="Z99",  # not in Literal
                prompt_before="a",
                prompt_after="b",
                word_count=3,
                rare_modifier_hits=[],
                structural_issues=[],
                status="pass",
            )


class TestPhase2Imports:
    """SubjectRef, TargetRelation, PreserveItem from phase2 are correctly imported."""

    def test_subject_ref(self):
        sr = SubjectRef(id="s1", description="a red fox", noun="fox")
        assert sr.id == "s1"
        assert sr.noun == "fox"

    def test_target_relation(self):
        tr = TargetRelation(type="attribute", value="changes color", subj="s1")
        assert tr.type == "attribute"
        assert tr.value == "changes color"

    def test_preserve_item(self):
        pi = PreserveItem(scope="background", constraint="stable")
        assert pi.scope == "background"

    def test_target_repair_result_uses_phase2_types(self):
        trr = TargetRepairResult(
            question_id="test",
            target_subjects=[SubjectRef(id="s1", description="elephant", noun="elephant")],
            target_relation=TargetRelation(type="action", value="jumps"),
            preservation_set=[PreserveItem(scope="background", constraint="stable")],
            reviewed_dimension="action_binding",
            dimension_consistent=True,
            confidence=0.95,
            repair_source="phase1_rebuild",
            status="pass",
        )
        assert trr.target_subjects[0].noun == "elephant"


class TestHumanAnnotationDiscriminatedUnion:
    """HumanAnnotation discriminated union routes by 'experiment' field."""

    def _base_fields(self, experiment: str) -> dict:
        return {
            "annotation_id": "ann_001",
            "question_id": "q001",
            "experiment": experiment,
            "method_code": "M1",
            "annotator_id": "annotator_a",
            "task_version": "v1",
            "created_at": "2026-07-01T00:00:00",
        }

    def test_target_repair_annotation(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(HumanAnnotation)
        data = {
            **self._base_fields("target_repair"),
            "subject_correct": True,
            "relation_correct": True,
            "preservation_correct": True,
            "dimension_correct": True,
        }
        ann = adapter.validate_python(data)
        assert isinstance(ann, TargetRepairAnnotation)

    def test_prompt_annotation(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(HumanAnnotation)
        data = {
            **self._base_fields("prompt"),
            "target_consistent": True,
            "dimension_correct": True,
            "prompt_naturalness": 4,
            "overall_usable": True,
        }
        ann = adapter.validate_python(data)
        assert isinstance(ann, PromptAnnotation)

    def test_clarity_annotation(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(HumanAnnotation)
        data = {
            **self._base_fields("clarity"),
            "subject_complete": True,
            "scene_reference_preserved": True,
            "geometry_distortion": 1,
            "artifact_present": False,
            "overall_usable": True,
        }
        ann = adapter.validate_python(data)
        assert isinstance(ann, ImageAnnotation)

    def test_difficulty_annotation(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(HumanAnnotation)
        data = {
            **self._base_fields("difficulty"),
            "difficulty_label": "hard",
            "target_complexity": 1.0,
            "subject_localization": 0.5,
            "background_interference": 0.5,
            "temporal_complexity": 1.0,
            "initial_state_visibility": 0.5,
            "target_observability": 0.5,
            "alternative_explanation_risk": 1.0,
            "evaluator_support_gap": 0.5,
        }
        ann = adapter.validate_python(data)
        assert isinstance(ann, DifficultyAnnotation)

    def test_final_review_annotation(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(HumanAnnotation)
        data = {
            **self._base_fields("final_review"),
            "asset_correct": True,
            "dimension_correct": True,
            "subject_correct": True,
            "transform_complete": True,
            "preservation_valid": True,
            "target_consistent": True,
            "final_accept": True,
            "rejection_reasons": [],
        }
        ann = adapter.validate_python(data)
        assert isinstance(ann, FinalReviewAnnotation)
