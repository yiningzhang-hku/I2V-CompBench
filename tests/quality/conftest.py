"""Pytest fixtures for quality module tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import json
import tempfile

import pytest

from tests.quality.fixtures.sample_data import (
    make_fixture_candidates,
    make_input_asset_manifest_row,
)


@pytest.fixture
def tmp_output(tmp_path):
    """临时输出目录。"""
    return tmp_path


@pytest.fixture
def fixture_candidates():
    """10条fixture候选数据。"""
    return make_fixture_candidates()


@pytest.fixture
def fixture_benchmark_dir(tmp_path, fixture_candidates):
    """模拟 benchmark 目录结构，含 manifest 和 first_frames。

    目录结构:
        tmp_path/
            benchmark/
                phase3_manifest.jsonl
                input_assets_manifest.jsonl
                first_frames/
                    *.png  (1x1 pixel PNGs)
                prompts/
                    final_prompts.jsonl
    """
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()

    # Write phase3_manifest.jsonl
    manifest_path = benchmark / "phase3_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for candidate in fixture_candidates:
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    # Create first_frames/ directory with minimal PNG files
    first_frames = benchmark / "first_frames"
    first_frames.mkdir()
    _create_minimal_pngs(first_frames, fixture_candidates)

    # Write input_assets_manifest.jsonl
    assets_path = benchmark / "input_assets_manifest.jsonl"
    with assets_path.open("w", encoding="utf-8") as f:
        for candidate in fixture_candidates:
            qid = candidate["question_id"]
            row = make_input_asset_manifest_row(qid, source_ref_id=f"src_{qid}")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write prompts/final_prompts.jsonl
    prompts_dir = benchmark / "prompts"
    prompts_dir.mkdir()
    prompts_path = prompts_dir / "final_prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as f:
        for candidate in fixture_candidates:
            entry = {
                "question_id": candidate["question_id"],
                "prompt": candidate["prompt"],
                "length_words": len(candidate["prompt"].split()),
                "forbidden_hits": [],
                "polish_attempts": 1,
                "used_fallback": False,
                "failed_check": None,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return benchmark


def _create_minimal_pngs(directory: Path, candidates: list):
    """Create 1x1 pixel minimal PNG files for each candidate."""
    # Minimal valid 1x1 white PNG (67 bytes)
    # Generated without Pillow dependency for robustness
    import struct
    import zlib

    def _make_minimal_png() -> bytes:
        """Create a minimal 1x1 pixel white PNG."""
        # PNG signature
        sig = b"\x89PNG\r\n\x1a\n"
        # IHDR chunk: width=1, height=1, bit_depth=8, color_type=2 (RGB)
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        # IDAT chunk: raw image data (filter byte 0 + RGB white pixel)
        raw_data = b"\x00\xff\xff\xff"  # filter=none, white pixel
        compressed = zlib.compress(raw_data)
        idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
        idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
        # IEND chunk
        iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
        return sig + ihdr + idat + iend

    png_bytes = _make_minimal_png()
    for candidate in candidates:
        qid = candidate["question_id"]
        (directory / f"{qid}.png").write_bytes(png_bytes)
