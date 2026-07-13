"""Tests for i2vcompbench.quality.paths module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import pytest

from i2vcompbench.quality.paths import (
    ensure_run_dirs,
    resolve_image_path,
    run_output_dir,
    to_posix,
)


class TestToPosix:
    """to_posix converts backslash paths to forward-slash."""

    def test_windows_backslash(self):
        result = to_posix("data\\benchmark\\first_frames\\test.png")
        assert result == "data/benchmark/first_frames/test.png"

    def test_already_posix_unchanged(self):
        result = to_posix("data/already/posix")
        assert result == "data/already/posix"

    def test_mixed_slashes(self):
        result = to_posix("data/mixed\\path\\file.png")
        assert result == "data/mixed/path/file.png"

    def test_empty_string(self):
        result = to_posix("")
        assert result == ""


class TestResolveImagePath:
    """resolve_image_path resolves relative paths against base directory."""

    def test_existing_file(self, tmp_path):
        img = tmp_path / "images" / "test.png"
        img.parent.mkdir(parents=True)
        img.write_bytes(b"\x89PNG")
        resolved = resolve_image_path(tmp_path, "images/test.png")
        assert resolved == img.resolve()
        assert resolved.exists()

    def test_nonexistent_file(self, tmp_path):
        resolved = resolve_image_path(tmp_path, "images/missing.png")
        # Should still resolve the path, just won't exist
        assert not resolved.exists()
        assert "missing.png" in str(resolved)

    def test_backslash_input(self, tmp_path):
        img = tmp_path / "sub" / "dir" / "file.png"
        img.parent.mkdir(parents=True)
        img.write_bytes(b"\x89PNG")
        resolved = resolve_image_path(tmp_path, "sub\\dir\\file.png")
        assert resolved == img.resolve()


class TestRunOutputDir:
    """run_output_dir generates correct path."""

    def test_basic(self):
        result = run_output_dir("output/root", "run_001")
        assert result == Path("output/root") / "run_001"

    def test_with_posix(self):
        result = run_output_dir("data/experiments", "20260712_120000")
        assert result.name == "20260712_120000"


class TestEnsureRunDirs:
    """ensure_run_dirs creates the standard subdirectory structure."""

    def test_creates_all_subdirs(self, tmp_path):
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        result = ensure_run_dirs(run_dir)

        expected_subdirs = {
            "audit", "splits", "prompt", "clarity", "aspect",
            "subject", "difficulty", "annotation", "selection", "lineage",
        }
        assert set(result.keys()) == expected_subdirs
        for name, path in result.items():
            assert path.exists()
            assert path.is_dir()
            assert path == run_dir / name

    def test_idempotent(self, tmp_path):
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        ensure_run_dirs(run_dir)
        # Call again should not raise
        result = ensure_run_dirs(run_dir)
        assert len(result) == 10

    def test_creates_parent_if_needed(self, tmp_path):
        run_dir = tmp_path / "nested" / "deep" / "run"
        # ensure_run_dirs uses parents=True, so it should work
        result = ensure_run_dirs(run_dir)
        assert all(p.exists() for p in result.values())
