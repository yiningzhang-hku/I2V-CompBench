"""
Build asset derivation lineage (§6.0).

Tracks the transform lineage from upstream source images to derived
first-frame assets used in the benchmark dataset. Since Phase 1 bundle
is currently unavailable, most records are marked as 'needs_manual_review'
or 'missing_lineage'.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

from i2vcompbench.quality.hashing import canonical_json_sha256, file_sha256
from i2vcompbench.quality.paths import to_posix
from i2vcompbench.quality.schemas import AssetLineageRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Known transform applied during Phase 2 construction
KNOWN_TRANSFORM = {
    "transform_name": "resize_long_edge",
    "transform_params": {
        "long_edge": 854,
        "method": "lanczos",
        "enlarge": True,
    },
}

_FIRST_FRAMES_REL = "data/benchmark_dataset/first_frames"
_INPUT_MANIFEST_REL = "data/benchmark_dataset/input_assets_manifest.jsonl"

_OUT_LINEAGE = "asset_lineage_manifest.jsonl"
_OUT_SUMMARY = "asset_lineage_summary.json"
_OUT_CONFLICTS = "asset_lineage_conflicts.jsonl"
_OUT_MIGRATION = "asset_lineage_migration_queue.jsonl"

# Placeholder for canonical upstream hash when Phase 1 data is unavailable
_PLACEHOLDER_HASH = "0" * 64

# Code version identifier
_CODE_VERSION = "phase2_construct_v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_source_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """
    Load source asset manifest into a lookup dict.

    Returns:
        Mapping from source_sample_id to the record dict.
    """
    manifest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return manifest
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = rec.get("source_sample_id", "")
                if sid:
                    manifest[sid] = rec
            except json.JSONDecodeError:
                continue
    return manifest


def _load_existing_qids(path: Path) -> set[str]:
    """Load already-processed question_ids from existing lineage manifest."""
    qids: set[str] = set()
    if not path.exists():
        return qids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "question_id" in rec:
                    qids.add(rec["question_id"])
            except json.JSONDecodeError:
                continue
    return qids


def _load_input_manifest_entries(
    path: Path,
) -> list[dict[str, Any]]:
    """Load input_assets_manifest.jsonl entries."""
    entries: list[dict[str, Any]] = []
    if not path.exists():
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def build_asset_lineage(
    benchmark_root: Path,
    output_dir: Path,
    asset_manifest_path: Path | None = None,
    config: dict | None = None,
    limit: int | None = None,
    resume: bool = False,
) -> dict:
    """
    构建资产派生谱系。

    Args:
        benchmark_root: 项目根目录.
        output_dir: 输出目录.
        asset_manifest_path: 已构建的 source_asset_manifest.jsonl 路径（可选）.
        config: 可选的配置覆盖 dict.
        limit: 最多处理的候选数（用于调试）.
        resume: 若为 True，跳过已存在于输出中的 question_id.

    Returns:
        summary dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    lineage_path = output_dir / _OUT_LINEAGE
    summary_path = output_dir / _OUT_SUMMARY
    conflicts_path = output_dir / _OUT_CONFLICTS
    migration_path = output_dir / _OUT_MIGRATION

    first_frames_dir = benchmark_root / _FIRST_FRAMES_REL

    # Load source asset manifest if available
    source_manifest: dict[str, dict[str, Any]] = {}
    if asset_manifest_path and asset_manifest_path.exists():
        source_manifest = _load_source_manifest(asset_manifest_path)
        logger.info(
            "Loaded source asset manifest: {} entries", len(source_manifest)
        )
    else:
        # Try default location in output_dir
        default_manifest = output_dir / "source_asset_manifest.jsonl"
        if default_manifest.exists():
            source_manifest = _load_source_manifest(default_manifest)
            logger.info(
                "Loaded source asset manifest from default location: {} entries",
                len(source_manifest),
            )
        else:
            logger.warning(
                "No source asset manifest available; "
                "all records will be marked as missing_lineage"
            )

    # Load input manifest to get candidate list with source_ref_id mappings
    input_manifest_path = benchmark_root / _INPUT_MANIFEST_REL
    if not input_manifest_path.exists():
        logger.error("Input manifest not found: {}", input_manifest_path)
        return {"error": f"Input manifest not found: {input_manifest_path}"}

    entries = _load_input_manifest_entries(input_manifest_path)

    # Resume support
    processed_qids: set[str] = set()
    if resume:
        processed_qids = _load_existing_qids(lineage_path)
        logger.info("Resume mode: {} question_ids already processed", len(processed_qids))

    # Compute transform spec hash (stable across runs)
    transform_spec_sha256 = canonical_json_sha256(KNOWN_TRANSFORM)

    # Tracking
    records: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    migration_queue: list[dict[str, Any]] = []
    missing_first_frame = 0

    status_distribution: dict[str, int] = {
        "verified": 0,
        "needs_manual_review": 0,
        "source_mismatch": 0,
        "derived_mismatch": 0,
    }

    total_candidates = 0

    for entry in entries:
        if limit is not None and total_candidates >= limit:
            break

        question_id: str = entry.get("question_id", "")
        assets: list[dict] = entry.get("assets", [])

        if not question_id:
            continue

        total_candidates += 1

        # Skip if already processed (resume)
        if question_id in processed_qids:
            continue

        # Extract source_ref_id from first asset (primary asset)
        source_ref_id: str | None = None
        role = "first_frame"
        asset_idx = 0
        for i, asset in enumerate(assets):
            if asset.get("role") == "first_frame":
                source_ref_id = asset.get("source_ref_id")
                role = asset.get("role", "first_frame")
                asset_idx = i
                break

        if not source_ref_id and assets:
            source_ref_id = assets[0].get("source_ref_id")
            role = assets[0].get("role", "first_frame")

        source_sample_id = source_ref_id or ""
        upstream_asset_id = (
            f"{source_ref_id}__{role}__{asset_idx:02d}"
            if source_ref_id
            else f"unknown__{question_id}"
        )

        # Locate first frame file
        first_frame_file = first_frames_dir / f"{question_id}.png"
        derived_path_posix = to_posix(
            f"data/benchmark_dataset/first_frames/{question_id}.png"
        )

        if not first_frame_file.exists():
            missing_first_frame += 1
            migration_queue.append({
                "question_id": question_id,
                "source_ref_id": source_ref_id,
                "derived_path": derived_path_posix,
                "reason": "first_frame_not_found",
                "details": f"First frame file missing: {first_frame_file}",
            })
            continue

        # Compute file SHA-256
        try:
            observed_sha256 = file_sha256(first_frame_file)
        except Exception as e:
            logger.warning("Error hashing {}: {}", first_frame_file, e)
            missing_first_frame += 1
            migration_queue.append({
                "question_id": question_id,
                "source_ref_id": source_ref_id,
                "derived_path": derived_path_posix,
                "reason": "hash_error",
                "details": f"Error computing hash: {e}",
            })
            continue

        # Determine status based on source manifest availability
        canonical_upstream_sha256 = _PLACEHOLDER_HASH
        status: str

        if source_manifest and source_sample_id in source_manifest:
            manifest_rec = source_manifest[source_sample_id]
            canonical_upstream_sha256 = manifest_rec.get(
                "canonical_upstream_sha256", _PLACEHOLDER_HASH
            )
            # Since Phase 1 bundle is unavailable, we cannot verify the
            # derivation chain end-to-end. Mark as needs_manual_review.
            status = "needs_manual_review"
        elif source_manifest:
            # Manifest exists but this source_sample_id is not in it
            status = "needs_manual_review"
        else:
            # No manifest at all
            status = "needs_manual_review"

        status_distribution[status] += 1

        # Build lineage record
        record = AssetLineageRecord(
            question_id=question_id,
            role=role,
            source_type="tip_derived_reference",
            source_sample_id=source_sample_id,
            upstream_asset_id=upstream_asset_id,
            canonical_upstream_sha256=canonical_upstream_sha256,
            observed_upstream_sha256=observed_sha256,
            transform_name=KNOWN_TRANSFORM["transform_name"],
            transform_params=KNOWN_TRANSFORM["transform_params"],
            transform_spec_sha256=transform_spec_sha256,
            derived_path=derived_path_posix,
            expected_derived_sha256=observed_sha256,  # best-effort: use observed
            code_version=_CODE_VERSION,
            verification_source="construction_run",
            status=status,
        )
        records.append(record.model_dump())

    # Write outputs
    write_mode = "a" if resume else "w"
    with open(lineage_path, write_mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(conflicts_path, "w", encoding="utf-8") as f:
        for c in conflicts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(migration_path, "w", encoding="utf-8") as f:
        for m in migration_queue:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Compute lineage manifest hash
    lineage_sha256 = ""
    if lineage_path.exists() and lineage_path.stat().st_size > 0:
        lineage_sha256 = file_sha256(lineage_path)

    # If resume, recount from full file
    lineage_records_created = len(records)
    if resume:
        count = 0
        with open(lineage_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        lineage_records_created = count

    # Summary
    summary = {
        "total_candidates": total_candidates,
        "lineage_records_created": lineage_records_created,
        "status_distribution": status_distribution,
        "missing_first_frame": missing_first_frame,
        "conflicts": len(conflicts),
        "lineage_manifest_sha256": lineage_sha256,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        "Asset lineage built: {} records, {} missing, {} conflicts",
        lineage_records_created,
        missing_first_frame,
        len(conflicts),
    )

    return summary
