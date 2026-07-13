"""
Build canonical upstream asset manifest (§6.0).

Scans input_assets_manifest.jsonl, computes SHA-256 and dimensions
for each referenced image file, and produces a structured
SourceAssetRecord manifest with conflict / migration-queue reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

from i2vcompbench.quality.hashing import canonical_json_sha256, file_sha256
from i2vcompbench.quality.paths import to_posix
from i2vcompbench.quality.schemas import SourceAssetRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INPUT_MANIFEST_REL = "data/benchmark_dataset/input_assets_manifest.jsonl"

_OUT_MANIFEST = "source_asset_manifest.jsonl"
_OUT_SUMMARY = "source_asset_manifest_summary.json"
_OUT_CONFLICTS = "source_asset_conflicts.jsonl"
_OUT_MIGRATION = "source_asset_migration_queue.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) of an image file without loading pixel data."""
    with Image.open(path) as img:
        return img.size  # (width, height)


def _load_existing_ids(path: Path) -> set[str]:
    """Load already-processed upstream_asset_ids from an existing manifest."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "upstream_asset_id" in rec:
                    ids.add(rec["upstream_asset_id"])
            except json.JSONDecodeError:
                continue
    return ids


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def build_asset_manifest(
    benchmark_root: Path,
    output_dir: Path,
    config: dict | None = None,
    limit: int | None = None,
    resume: bool = False,
) -> dict:
    """
    构建 canonical 上游资产清单。

    Args:
        benchmark_root: 项目根目录 (包含 data/benchmark_dataset/).
        output_dir: 输出目录.
        config: 可选的配置覆盖 dict.
        limit: 最多处理的记录数（用于调试）.
        resume: 若为 True，跳过 output 中已存在的 upstream_asset_id.

    Returns:
        summary dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / _OUT_MANIFEST
    summary_path = output_dir / _OUT_SUMMARY
    conflicts_path = output_dir / _OUT_CONFLICTS
    migration_path = output_dir / _OUT_MIGRATION

    # Input manifest
    input_manifest = benchmark_root / _INPUT_MANIFEST_REL
    if not input_manifest.exists():
        logger.error("Input manifest not found: {}", input_manifest)
        return {"error": f"Input manifest not found: {input_manifest}"}

    # Resume support
    processed_ids: set[str] = set()
    if resume:
        processed_ids = _load_existing_ids(manifest_path)
        logger.info("Resume mode: {} records already processed", len(processed_ids))

    # Tracking structures
    records: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    migration_queue: list[dict[str, Any]] = []
    missing_files = 0

    # Dedup: (source_sample_id, role) -> sha256
    seen_hashes: dict[tuple[str, str], str] = {}

    # Compute input manifest SHA-256 for provenance
    input_manifest_sha256 = file_sha256(input_manifest)
    input_manifest_posix = to_posix(
        str(input_manifest.relative_to(benchmark_root))
    )

    # Read and process
    total_scanned = 0
    with open(input_manifest, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            if limit is not None and total_scanned >= limit:
                break

            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line {}", line_no)
                continue

            question_id: str = entry.get("question_id", "")
            assets: list[dict] = entry.get("assets", [])

            for idx, asset in enumerate(assets):
                total_scanned += 1
                if limit is not None and total_scanned > limit:
                    break

                source_ref_id: str | None = asset.get("source_ref_id")
                role: str = asset.get("role", "unknown")
                declared_path: str = asset.get("path", "")

                # source_ref_id missing → migration queue
                if not source_ref_id:
                    migration_queue.append({
                        "question_id": question_id,
                        "source_ref_id": None,
                        "declared_path": to_posix(declared_path),
                        "reason": "source_id_missing",
                        "details": f"Asset at index {idx} has no source_ref_id",
                    })
                    continue

                # Build IDs
                source_sample_id = source_ref_id
                upstream_asset_id = f"{source_ref_id}__{role}__{idx:02d}"

                # Skip if already processed (resume)
                if upstream_asset_id in processed_ids:
                    continue

                # Resolve file path
                posix_path = to_posix(declared_path)
                abs_path = (benchmark_root / Path(posix_path)).resolve()

                # Map role to schema enum
                asset_role: str
                if role == "first_frame":
                    asset_role = "source_first_frame"
                else:
                    asset_role = "reference_asset"

                # Try to read file
                if not abs_path.exists():
                    missing_files += 1
                    migration_queue.append({
                        "question_id": question_id,
                        "source_ref_id": source_ref_id,
                        "declared_path": posix_path,
                        "reason": "file_not_found",
                        "details": f"File does not exist: {abs_path}",
                    })
                    continue

                # Compute hash and dimensions
                try:
                    sha256 = file_sha256(abs_path)
                    width, height = _image_dimensions(abs_path)
                except Exception as e:
                    logger.warning(
                        "Error reading {}: {}", abs_path, e
                    )
                    migration_queue.append({
                        "question_id": question_id,
                        "source_ref_id": source_ref_id,
                        "declared_path": posix_path,
                        "reason": "file_not_found",
                        "details": f"Error reading file: {e}",
                    })
                    missing_files += 1
                    continue

                # Conflict detection: same (source_sample_id, role) with different hash
                dedup_key = (source_sample_id, role)
                if dedup_key in seen_hashes:
                    if seen_hashes[dedup_key] != sha256:
                        conflicts.append({
                            "question_id": question_id,
                            "source_sample_id": source_sample_id,
                            "role": role,
                            "upstream_asset_id": upstream_asset_id,
                            "declared_path": posix_path,
                            "existing_hash": seen_hashes[dedup_key],
                            "new_hash": sha256,
                            "reason": "hash_conflict",
                            "details": (
                                f"Same source_sample_id+role but different SHA-256"
                            ),
                        })
                        migration_queue.append({
                            "question_id": question_id,
                            "source_ref_id": source_ref_id,
                            "declared_path": posix_path,
                            "reason": "hash_conflict",
                            "details": (
                                f"Conflict with existing hash for "
                                f"{source_sample_id}/{role}"
                            ),
                        })
                        continue
                else:
                    seen_hashes[dedup_key] = sha256

                # Build record
                record = SourceAssetRecord(
                    upstream_asset_id=upstream_asset_id,
                    source_sample_id=source_sample_id,
                    asset_role=asset_role,
                    upstream_path=posix_path,
                    canonical_upstream_sha256=sha256,
                    width=width,
                    height=height,
                    source_manifest_path=input_manifest_posix,
                    source_manifest_sha256=input_manifest_sha256,
                    verification_source="manual_migration",
                    verified_by=None,
                    verified_at=None,
                )
                records.append(record.model_dump())

            if limit is not None and total_scanned >= limit:
                break

    # Write outputs
    # 1. Manifest
    write_mode = "a" if resume else "w"
    with open(manifest_path, write_mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 2. Conflicts
    with open(conflicts_path, "w", encoding="utf-8") as f:
        for c in conflicts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 3. Migration queue
    with open(migration_path, "w", encoding="utf-8") as f:
        for m in migration_queue:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Compute manifest hash
    manifest_sha256 = ""
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        manifest_sha256 = file_sha256(manifest_path)

    # Count unique source_sample_ids
    unique_source_ids = set()
    if resume:
        # Re-read full manifest for accurate count
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        unique_source_ids.add(r.get("source_sample_id", ""))
                    except json.JSONDecodeError:
                        pass
        total_records = len(unique_source_ids)  # approximate
    else:
        unique_source_ids = {r["source_sample_id"] for r in records}
        total_records = len(records)

    # 4. Summary
    summary = {
        "total_assets_scanned": total_scanned,
        "manifest_records_created": total_records,
        "unique_source_sample_ids": len(unique_source_ids),
        "conflicts": len(conflicts),
        "migration_queue_size": len(migration_queue),
        "missing_files": missing_files,
        "manifest_sha256": manifest_sha256,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        "Asset manifest built: {} records, {} conflicts, {} migration items",
        total_records,
        len(conflicts),
        len(migration_queue),
    )

    return summary
