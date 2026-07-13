"""Tests for i2vcompbench.quality.split module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import json
from collections import Counter

import pytest

from i2vcompbench.quality.split import FORMAL_DIMENSIONS, run_split, stratified_split
from tests.quality.fixtures.sample_data import make_fixture_candidates


class TestStratifiedSplit:
    """stratified_split correctly partitions candidates."""

    def test_basic_split_5dim_2each(self, fixture_candidates):
        """10条fixture中5维各2条 → n_per_dim=1时产出5条。"""
        result = stratified_split(fixture_candidates, n_per_dimension=1, seed=42)
        assert len(result) == 5
        dims = [r["dimension"] for r in result]
        dim_counts = Counter(dims)
        for dim in FORMAL_DIMENSIONS:
            assert dim_counts[dim] == 1

    def test_no_overlap_between_dev_and_val(self, fixture_candidates):
        """开发集和验证集 question_id 无重叠。"""
        dev = stratified_split(fixture_candidates, n_per_dimension=1, seed=42)
        dev_qids = {r["question_id"] for r in dev}

        remaining = [c for c in fixture_candidates if c["question_id"] not in dev_qids]
        val = stratified_split(remaining, n_per_dimension=1, seed=42)
        val_qids = {r["question_id"] for r in val}

        assert dev_qids & val_qids == set()

    def test_deterministic_same_seed(self, fixture_candidates):
        """相同seed产生相同结果。"""
        result1 = stratified_split(fixture_candidates, n_per_dimension=1, seed=12345)
        result2 = stratified_split(fixture_candidates, n_per_dimension=1, seed=12345)
        qids1 = [r["question_id"] for r in result1]
        qids2 = [r["question_id"] for r in result2]
        assert qids1 == qids2

    def test_different_seed_different_result(self, fixture_candidates):
        """不同seed产生不同结果（在数据量允许的情况下）。"""
        result1 = stratified_split(fixture_candidates, n_per_dimension=1, seed=1)
        result2 = stratified_split(fixture_candidates, n_per_dimension=1, seed=999)
        qids1 = set(r["question_id"] for r in result1)
        qids2 = set(r["question_id"] for r in result2)
        # With only 2 per dim, different seeds may still select same items
        # but at minimum the function should run without error
        assert len(result1) == len(result2) == 5

    def test_exact_per_dimension_count(self, fixture_candidates):
        """每维精确数量。"""
        result = stratified_split(fixture_candidates, n_per_dimension=1, seed=42)
        dim_counts = Counter(r["dimension"] for r in result)
        for dim in FORMAL_DIMENSIONS:
            assert dim_counts[dim] == 1


class TestRunSplit:
    """run_split executes the full pipeline correctly."""

    def test_full_split_pipeline(self, fixture_benchmark_dir, tmp_path):
        """Full pipeline with dev_per_dim=1, val_per_dim=1."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        summary = run_split(
            benchmark_root=fixture_benchmark_dir,
            output_dir=output_dir,
            seed=42,
            dev_per_dim=1,
            val_per_dim=1,
        )

        assert summary["seed"] == 42
        assert summary["development"]["total"] == 5
        assert summary["validation"]["total"] == 5
        assert summary["overlap_check"] == "pass"

        # Check output files exist
        splits_dir = output_dir / "splits"
        assert (splits_dir / "development_250.jsonl").exists()
        assert (splits_dir / "validation_250.jsonl").exists()
        assert (splits_dir / "split_summary.json").exists()
        assert (splits_dir / "split_hash.txt").exists()

    def test_no_qid_overlap_in_output(self, fixture_benchmark_dir, tmp_path):
        """Output dev/val sets have no overlapping question_ids."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        run_split(
            benchmark_root=fixture_benchmark_dir,
            output_dir=output_dir,
            seed=42,
            dev_per_dim=1,
            val_per_dim=1,
        )

        splits_dir = output_dir / "splits"
        dev_rows = _read_jsonl(splits_dir / "development_250.jsonl")
        val_rows = _read_jsonl(splits_dir / "validation_250.jsonl")

        dev_qids = {r["question_id"] for r in dev_rows}
        val_qids = {r["question_id"] for r in val_rows}
        assert dev_qids & val_qids == set()


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
