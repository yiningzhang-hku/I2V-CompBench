"""Tests for i2vcompbench.quality.prompt_rules module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pytest

from i2vcompbench.quality.prompt_rules import (
    check_article_before_punctuation,
    check_change_verb,
    check_empty_prompt,
    check_repeated_articles,
    check_unresolved_placeholders,
    check_view_camera_cues,
    check_word_count,
)


class TestCheckEmptyPrompt:
    """check_empty_prompt detects empty/whitespace-only prompts."""

    def test_empty_string(self):
        assert check_empty_prompt("") == ["empty_prompt"]

    def test_whitespace_only(self):
        assert check_empty_prompt("   ") == ["empty_prompt"]

    def test_valid_text(self):
        assert check_empty_prompt("valid text") == []

    def test_none_like(self):
        assert check_empty_prompt("") == ["empty_prompt"]


class TestCheckUnresolvedPlaceholders:
    """check_unresolved_placeholders detects {xxx} patterns."""

    def test_with_placeholder(self):
        result = check_unresolved_placeholders("The {subject} walks")
        assert result == ["unresolved_placeholder"]

    def test_no_placeholder(self):
        result = check_unresolved_placeholders("The elephant walks")
        assert result == []

    def test_multiple_placeholders(self):
        result = check_unresolved_placeholders("{subject} does {action}")
        assert result == ["unresolved_placeholder"]


class TestCheckRepeatedArticles:
    """check_repeated_articles detects 'the the' / 'a a' / 'an an'."""

    def test_the_the(self):
        result = check_repeated_articles("The the elephant walks")
        assert result == ["repeated_article"]

    def test_a_a(self):
        result = check_repeated_articles("A a dog runs")
        assert result == ["repeated_article"]

    def test_no_repetition(self):
        result = check_repeated_articles("The elephant walks")
        assert result == []

    def test_case_insensitive(self):
        result = check_repeated_articles("the THE elephant walks")
        # Lowered tokens: "the", "the" → caught
        assert result == ["repeated_article"]

    def test_articles_not_adjacent(self):
        result = check_repeated_articles("The elephant and the dog play")
        assert result == []


class TestCheckArticleBeforePunctuation:
    """check_article_before_punctuation catches 'The,' patterns."""

    def test_the_comma(self):
        result = check_article_before_punctuation("The, elephant")
        assert result == ["article_before_punctuation"]

    def test_a_period(self):
        result = check_article_before_punctuation("A. dog runs")
        assert result == ["article_before_punctuation"]

    def test_normal_usage(self):
        result = check_article_before_punctuation("The elephant walks.")
        assert result == []


class TestCheckWordCount:
    """check_word_count validates min/max word thresholds."""

    def test_too_short(self):
        result = check_word_count("short", min_words=8)
        assert result == ["word_count_too_short"]

    def test_too_long(self):
        long_prompt = " ".join(["word"] * 30)
        result = check_word_count(long_prompt, max_words=25)
        assert result == ["word_count_too_long"]

    def test_within_range(self):
        prompt = "The elephant gently lifts its trunk upward into the sky"
        result = check_word_count(prompt, min_words=8, max_words=25)
        assert result == []

    def test_empty_handled_elsewhere(self):
        # Empty prompt returns [] (handled by check_empty_prompt)
        result = check_word_count("", min_words=8)
        assert result == []


class TestCheckViewCameraCues:
    """check_view_camera_cues validates camera/motion cues for view dimension."""

    def test_view_with_camera_cue(self):
        """view_transformation with camera cue → passes."""
        result = check_view_camera_cues(
            "Camera pans across the scene", "view_transformation"
        )
        assert result == []

    def test_view_with_zoom(self):
        """view_transformation with zoom → passes."""
        result = check_view_camera_cues(
            "The scene zooms in on the details", "view_transformation"
        )
        assert result == []

    def test_view_missing_camera_cue(self):
        """view_transformation without camera cue → missing_camera_cue."""
        result = check_view_camera_cues(
            "The sky turns blue", "view_transformation"
        )
        assert result == ["missing_camera_cue"]

    def test_non_view_dimension_skipped(self):
        """Non-view dimension → always passes (not checked)."""
        result = check_view_camera_cues(
            "The sky turns blue", "action_binding"
        )
        assert result == []

    def test_view_with_pan_keyword(self):
        """回归测试: View prompt含pan时不被错误拒绝。"""
        result = check_view_camera_cues(
            "The camera pan left reveals a mountain", "view_transformation"
        )
        assert result == []

    def test_view_with_orbit(self):
        """回归测试: orbit关键词不被错误拒绝。"""
        result = check_view_camera_cues(
            "The shot orbiting around the subject slowly", "view_transformation"
        )
        # "orbit" is in VIEW_CAMERA_CUES
        assert result == []


class TestCheckChangeVerb:
    """check_change_verb validates presence of action/change verbs."""

    def test_has_action_verb(self):
        result = check_change_verb("The elephant walks across the field")
        assert result == []

    def test_static_only(self):
        """Static verbs only (is/are/was) → missing_change_verb."""
        result = check_change_verb("The elephant is happy being content")
        assert result == ["missing_change_verb"]

    def test_empty_prompt(self):
        result = check_change_verb("")
        assert result == ["missing_change_verb"]

    def test_ing_form_detected(self):
        result = check_change_verb("The bird flying across the sky")
        assert result == []

    def test_ed_form_detected(self):
        result = check_change_verb("The ball bounced off the wall")
        assert result == []

    def test_stands_still_static_exclusion(self):
        """验证静态动词排除 — 'stands' has suffix but is not purely static."""
        # 'stands' matches the verb suffix pattern (-s), so it should pass
        result = check_change_verb("The elephant stands still being happy")
        assert result == []


class TestRegressionPromptRules:
    """回归测试: 确保已知边缘情况不会误判。"""

    def test_view_zoom_pan_not_rejected(self):
        """View prompt含zoom/pan时不被错误拒绝。"""
        prompts = [
            "The camera zoom in on the clockwork mechanism steadily and smoothly",
            "The shot pans right revealing the entire valley below",
            "The camera tilts up showing the tall building",
        ]
        for prompt in prompts:
            result = check_view_camera_cues(prompt, "view_transformation")
            assert result == [], f"Prompt incorrectly rejected: {prompt}"

    def test_repeated_article_various(self):
        """Various repeated article patterns."""
        assert check_repeated_articles("an an elephant") == ["repeated_article"]
        assert check_repeated_articles("AN AN elephant") == ["repeated_article"]
        assert check_repeated_articles("a big an elephant") == []
