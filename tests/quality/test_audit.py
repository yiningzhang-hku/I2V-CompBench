"""Tests for i2vcompbench.quality.audit module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import json

import pytest

from i2vcompbench.quality.audit import AuditContext, audit_candidate
from tests.quality.fixtures.sample_data import make_sample_candidate


def _make_context_with_assets(candidates: list[dict]) -> AuditContext:
    """Create an AuditContext pre-loaded with asset manifest for given candidates."""
    ctx = AuditContext()
    for c in candidates:
        qid = c["question_id"]
        # Simulate asset manifest entries keyed by source_sample_id
        source_id = c.get("source_sample_id", qid)
        ctx.asset_manifest[source_id] = {"question_id": qid}
    return ctx


def _audit_one(candidate: dict, config: dict = None, context: AuditContext = None,
               benchmark_root: Path = None) -> dict:
    """Convenience: audit a single candidate with sensible defaults."""
    if config is None:
        config = {"min_prompt_words": 5, "max_prompt_words": 30}
    if context is None:
        context = _make_context_with_assets([candidate])
    if benchmark_root is None:
        benchmark_root = Path(".")
    config["_benchmark_root"] = benchmark_root
    return audit_candidate(candidate, config, context)


class TestAuditBasicEligibility:
    """Core audit eligibility checks."""

    def test_complete_candidate_eligible(self, fixture_benchmark_dir):
        """完整字段候选 → eligible=True, issues minimal (only image_not_found due to path)."""
        candidate = make_sample_candidate()
        # Adjust first_frame_path to point to actual fixture file
        candidate["first_frame_path"] = str(
            fixture_benchmark_dir / "first_frames" / "attr_single_0001.png"
        )
        ctx = _make_context_with_assets([candidate])
        config = {"min_prompt_words": 5, "max_prompt_words": 30}
        config["_benchmark_root"] = fixture_benchmark_dir
        result = audit_candidate(candidate, config, ctx)
        # Should have no blocking issues (image exists, all fields present)
        assert result["eligible"] is True
        assert result["blocking_issues"] == []

    def test_empty_prompt_blocked(self):
        """空prompt → includes empty_prompt."""
        candidate = make_sample_candidate(prompt="")
        result = _audit_one(candidate)
        assert "empty_prompt" in result["issues"]
        assert result["eligible"] is False

    def test_repeated_article_blocked(self):
        """'The the subject' → includes repeated_article."""
        candidate = make_sample_candidate(
            prompt="The the subject walks slowly across the grassy field today."
        )
        result = _audit_one(candidate)
        assert "repeated_article" in result["issues"]
        assert result["eligible"] is False

    def test_missing_target_noun(self):
        """noun=None → includes missing_target_noun."""
        candidate = make_sample_candidate(noun="elephant")
        # Manually set noun to None while keeping the subject in the list
        candidate["target_subjects"][0]["noun"] = None
        result = _audit_one(candidate)
        assert "missing_target_noun" in result["issues"]

    def test_generic_target_description(self):
        """description='the subject' → includes generic_target_description."""
        candidate = make_sample_candidate()
        candidate["target_subjects"][0]["description"] = "the subject"
        result = _audit_one(candidate)
        assert "generic_target_description" in result["issues"]

    def test_missing_target_relation(self):
        """target_relation=None → includes missing_target_relation."""
        candidate = make_sample_candidate()
        candidate["target_relation"] = None
        result = _audit_one(candidate)
        assert "missing_target_relation" in result["issues"]

    def test_missing_preservation_set(self):
        """preservation_set=[] → includes missing_preservation_set."""
        candidate = make_sample_candidate(has_preservation=False)
        result = _audit_one(candidate)
        assert "missing_preservation_set" in result["issues"]
        assert result["eligible"] is False

    def test_invalid_dimension(self):
        """非正式维度 → includes invalid_dimension."""
        candidate = make_sample_candidate(dimension="spatial_composition")
        result = _audit_one(candidate)
        assert "invalid_dimension" in result["issues"]
        assert result["eligible"] is False

    def test_windows_backslash_path_non_blocking(self):
        """windows_backslash_path is a warning, not blocking."""
        candidate = make_sample_candidate()
        candidate["first_frame_path"] = "data\\benchmark\\first_frames\\test.png"
        result = _audit_one(candidate)
        assert "windows_backslash_path" in result["issues"]
        # windows_backslash_path is in WARNING_ISSUES, so it alone doesn't block
        # (image_not_found will still block since path doesn't exist)
        assert "windows_backslash_path" not in result["blocking_issues"]


class TestAuditRegressionTests:
    """回归测试 — 确保已知缺陷模式被正确拒绝。"""

    def test_repeated_article_rejected(self):
        """'The the subject.' 被拒绝 (repeated_article)."""
        candidate = make_sample_candidate(
            prompt="The the subject moves forward across the open field quickly."
        )
        result = _audit_one(candidate)
        assert "repeated_article" in result["issues"]
        assert result["eligible"] is False

    def test_failed_fallback_not_eligible(self):
        """failed_check 标记的记录不应进入正式manifest。"""
        candidate = make_sample_candidate()
        ctx = _make_context_with_assets([candidate])
        ctx.final_prompts[candidate["question_id"]] = {
            "question_id": candidate["question_id"],
            "failed_check": "missing_active_verb",
        }
        config = {"min_prompt_words": 5, "max_prompt_words": 30, "_benchmark_root": Path(".")}
        result = audit_candidate(candidate, config, ctx)
        assert "has_failed_check" in result["issues"]
        assert result["eligible"] is False

    def test_missing_preservation_set_rejected(self):
        """缺少preservation_set的记录eligible=False。"""
        candidate = make_sample_candidate(has_preservation=False)
        result = _audit_one(candidate)
        assert "missing_preservation_set" in result["issues"]
        assert result["eligible"] is False
