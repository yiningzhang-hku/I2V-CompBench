"""
Quality audit module for I2V-CompBench candidate pool.

Performs comprehensive quality checks on phase3_manifest candidates and
produces structured audit reports (JSONL rows, summary JSON, blocking examples).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from i2vcompbench.quality.paths import resolve_image_path, to_posix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_DIMENSIONS = (
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "background_dynamics",
    "view_transformation",
)

# Issues that are warnings only (non-blocking)
WARNING_ISSUES = frozenset({"windows_backslash_path", "duplicate_source", "missing_target_relation", "empty_target_change"})

# Dimension → expected target_relation.type mapping
DIMENSION_RELATION_TYPE_MAP = {
    "attribute_binding": "attribute",
    "action_binding": "action",
    "motion_binding": "motion",
    "background_dynamics": "background",
    "view_transformation": "view",
}

# Template boilerplate phrases to strip from target_relation.value
_BOILERPLATE_PATTERNS = re.compile(
    r"^(the\s+subject\s+)?(should|will|must|is\s+expected\s+to)\s+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AuditContext:
    """Holds shared state across candidate audits."""

    seen_qids: set = field(default_factory=set)
    source_id_to_qids: dict = field(default_factory=lambda: defaultdict(list))
    asset_manifest: dict = field(default_factory=dict)  # qid → row
    final_prompts: dict = field(default_factory=dict)  # qid → row


# ---------------------------------------------------------------------------
# IO Helpers
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Iterate over JSONL lines, yielding dicts."""
    if not path.exists():
        logger.warning(f"JSONL file not found: {path}")
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error(f"Invalid JSON at {path}:{line_no}: {exc}")
                continue
            if isinstance(obj, dict):
                yield obj


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def _check_prompt(prompt: str, config: dict) -> List[str]:
    """Check prompt quality, return list of issue codes."""
    issues: List[str] = []
    text = (prompt or "").strip()

    if not text:
        return ["empty_prompt"]

    # Unresolved slot placeholders like {subject}
    if re.search(r"\{[^{}]+\}", text):
        issues.append("unresolved_placeholder")

    # Repeated articles: "the the", "a a", "an an"
    if re.search(r"\b(the|a|an)\s+\1\b", text, flags=re.IGNORECASE):
        issues.append("repeated_article")

    # Article directly before punctuation
    if re.search(r"\b(the|a|an)\s*[,.;:!?]", text, flags=re.IGNORECASE):
        issues.append("article_before_punctuation")

    # Word count range
    min_words = config.get("min_prompt_words", 8)
    max_words = config.get("max_prompt_words", 25)
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if not (min_words <= len(words) <= max_words):
        issues.append("word_count_out_of_range")

    return issues


def _check_targets(row: Dict[str, Any]) -> List[str]:
    """Check target_subjects and target_relation fields."""
    issues: List[str] = []

    # target_subjects
    subjects = row.get("target_subjects") or []
    if not subjects:
        issues.append("missing_target_subjects")
    for subject in subjects:
        noun = subject.get("noun")
        if noun is None or str(noun).strip() == "":
            issues.append("missing_target_noun")
        desc = str(subject.get("description") or "").strip().lower()
        if desc in {"", "subject", "the subject"}:
            issues.append("generic_target_description")

    # target_relation
    relation = row.get("target_relation")
    if relation is None or not isinstance(relation, dict):
        issues.append("missing_target_relation")
    else:
        raw_value = str(relation.get("value") or "").strip()
        # Strip boilerplate, quotes, punctuation
        cleaned = _BOILERPLATE_PATTERNS.sub("", raw_value)
        cleaned = re.sub(r"[\"'""''.,;:!?]", "", cleaned).strip()
        if not cleaned:
            issues.append("empty_target_change")

    # preservation_set
    preservation = row.get("preservation_set")
    if not preservation:
        issues.append("missing_preservation_set")

    return issues


def _check_dimension_consistency(row: Dict[str, Any]) -> List[str]:
    """Check if target_relation.type matches dimension."""
    issues: List[str] = []
    dimension = row.get("dimension", "")
    relation = row.get("target_relation")

    if relation is None or not isinstance(relation, dict):
        # Already flagged by target checks
        return issues

    rel_type = str(relation.get("type") or "").strip().lower()
    expected_type = DIMENSION_RELATION_TYPE_MAP.get(dimension, "")

    if rel_type and expected_type and rel_type != expected_type:
        issues.append("dimension_type_mismatch")

    return issues


def _check_image(row: Dict[str, Any], benchmark_root: Path) -> List[str]:
    """Check first_frame_path existence and format."""
    issues: List[str] = []
    raw_path = str(row.get("first_frame_path") or "")

    if not raw_path:
        issues.append("image_not_found")
        return issues

    # Windows backslash warning
    if "\\" in raw_path:
        issues.append("windows_backslash_path")

    # Convert to POSIX and resolve
    posix_path = to_posix(raw_path)
    # Resolve relative to project root (parent of benchmark_root)
    project_root = benchmark_root.parent.parent
    resolved = project_root / posix_path

    if not resolved.exists():
        # Also try relative to benchmark_root
        resolved_alt = benchmark_root / Path(posix_path).name
        if not resolved_alt.exists():
            issues.append("image_not_found")

    return issues


def _check_source(
    row: Dict[str, Any], context: AuditContext
) -> List[str]:
    """Check source linkage to input_assets_manifest."""
    issues: List[str] = []
    qid = str(row.get("question_id") or "")

    # Use source_sample_id if present, fallback to question_id for manifest lookup
    source_id = str(row.get("source_sample_id") or qid)

    # Check if linkable to asset manifest
    if source_id not in context.asset_manifest:
        issues.append("missing_source_link")

    # Track source_id for duplicate detection (handled in post-processing)
    context.source_id_to_qids[source_id].append(qid)

    return issues


# ---------------------------------------------------------------------------
# Core audit function
# ---------------------------------------------------------------------------


def audit_candidate(row: dict, config: dict, context: AuditContext) -> dict:
    """Audit a single candidate row, return structured result with issues."""
    qid = str(row.get("question_id") or "")
    dimension = str(row.get("dimension") or "")
    issues: List[str] = []

    # 1. Dimension check
    if dimension not in VALID_DIMENSIONS:
        issues.append("invalid_dimension")

    # 2. Uniqueness check
    if qid in context.seen_qids:
        issues.append("duplicate_question_id")
    context.seen_qids.add(qid)

    # 3. Prompt check
    prompt = str(row.get("prompt") or "")
    final_prompt_row = context.final_prompts.get(qid, {})
    if final_prompt_row.get("failed_check"):
        issues.append("has_failed_check")
    issues.extend(_check_prompt(prompt, config))

    # 4. Target checks
    issues.extend(_check_targets(row))

    # 5. Dimension consistency
    issues.extend(_check_dimension_consistency(row))

    # 6. Image check
    benchmark_root = config.get("_benchmark_root", Path("."))
    issues.extend(_check_image(row, benchmark_root))

    # 7. Source check
    issues.extend(_check_source(row, context))

    # Deduplicate and sort
    issues = sorted(set(issues))

    # Classify blocking vs warning
    blocking_issues = [i for i in issues if i not in WARNING_ISSUES]

    return {
        "question_id": qid,
        "dimension": dimension,
        "issues": issues,
        "issue_count": len(issues),
        "blocking_issues": blocking_issues,
        "blocking": bool(blocking_issues),
        "eligible": not bool(blocking_issues),
    }


# ---------------------------------------------------------------------------
# Post-processing: detect duplicate_source across all candidates
# ---------------------------------------------------------------------------


def _apply_duplicate_source_warnings(
    results: List[dict], context: AuditContext
) -> None:
    """Add duplicate_source warning to results where source_id produced multiple candidates."""
    # Find source_ids that produced multiple candidates
    duplicate_sources = {
        sid for sid, qids in context.source_id_to_qids.items() if len(qids) > 1
    }
    if not duplicate_sources:
        return

    # Build qid → result index for fast lookup
    qid_to_idx = {r["question_id"]: i for i, r in enumerate(results)}

    for sid in duplicate_sources:
        for qid in context.source_id_to_qids[sid]:
            idx = qid_to_idx.get(qid)
            if idx is None:
                continue
            result = results[idx]
            if "duplicate_source" not in result["issues"]:
                result["issues"] = sorted(result["issues"] + ["duplicate_source"])
                result["issue_count"] = len(result["issues"])
                # duplicate_source is a warning, does not change blocking status


# ---------------------------------------------------------------------------
# Main audit pipeline
# ---------------------------------------------------------------------------


def run_audit(
    benchmark_root: Path,
    output_dir: Path,
    config: dict,
    limit: int | None = None,
    only_qid: str | None = None,
    resume: bool = False,
) -> dict:
    """Execute full audit pipeline.

    Args:
        benchmark_root: Path to data/benchmark_dataset/
        output_dir: Directory for audit output files
        config: Configuration dict (may contain min_prompt_words, max_prompt_words)
        limit: Max candidates to process (None = all)
        only_qid: Process only this question_id (for debugging)
        resume: If True, skip already-processed qids from existing output

    Returns:
        Summary dict
    """
    benchmark_root = Path(benchmark_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Inject benchmark_root into config for image resolution
    config = dict(config)
    config["_benchmark_root"] = benchmark_root

    logger.info(f"Starting audit | root={benchmark_root} | output={output_dir}")

    # Load data sources
    manifest_path = benchmark_root / "phase3_manifest.jsonl"
    assets_path = benchmark_root / "input_assets_manifest.jsonl"
    prompts_path = benchmark_root / "prompts" / "final_prompts.jsonl"

    candidates = list(_iter_jsonl(manifest_path))
    logger.info(f"Loaded {len(candidates)} candidates from phase3_manifest")

    # Build context
    context = AuditContext()

    # Asset manifest indexed by question_id
    for row in _iter_jsonl(assets_path):
        qid = row.get("question_id")
        if qid:
            context.asset_manifest[qid] = row
    logger.info(f"Loaded {len(context.asset_manifest)} asset manifest entries")

    # Final prompts indexed by question_id
    for row in _iter_jsonl(prompts_path):
        qid = row.get("question_id")
        if qid:
            context.final_prompts[qid] = row
    logger.info(f"Loaded {len(context.final_prompts)} final prompt entries")

    # Resume support: load already-processed qids
    existing_qids: set = set()
    rows_output_path = output_dir / "candidate_quality_rows.jsonl"
    if resume and rows_output_path.exists():
        for row in _iter_jsonl(rows_output_path):
            existing_qids.add(row.get("question_id"))
        logger.info(f"Resume: skipping {len(existing_qids)} already-processed rows")

    # Filter candidates
    if only_qid:
        candidates = [c for c in candidates if c.get("question_id") == only_qid]
        logger.info(f"Filtered to single qid={only_qid}: {len(candidates)} rows")

    if limit is not None:
        candidates = candidates[:limit]
        logger.info(f"Limited to {limit} candidates")

    # Audit each candidate
    results: List[dict] = []
    for row in candidates:
        qid = str(row.get("question_id") or "")
        if resume and qid in existing_qids:
            continue
        result = audit_candidate(row, config, context)
        results.append(result)

    # Post-processing: duplicate source detection
    _apply_duplicate_source_warnings(results, context)

    # Build summary
    dim_counts: Counter = Counter()
    issue_dist: Counter = Counter()
    blocking_issue_dist: Counter = Counter()
    eligible_count = 0
    blocked_count = 0

    for r in results:
        dim_counts[r["dimension"]] += 1
        for issue in r["issues"]:
            issue_dist[issue] += 1
        for issue in r["blocking_issues"]:
            blocking_issue_dist[issue] += 1
        if r["eligible"]:
            eligible_count += 1
        else:
            blocked_count += 1

    summary = {
        "total_candidates": len(results),
        "by_dimension": dict(sorted(dim_counts.items())),
        "eligible_count": eligible_count,
        "blocked_count": blocked_count,
        "issue_distribution": dict(issue_dist.most_common()),
        "blocking_issue_distribution": dict(blocking_issue_dist.most_common()),
    }

    # Collect blocking examples (top 3 per blocking issue type)
    blocking_examples: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        for issue in r["blocking_issues"]:
            if len(blocking_examples[issue]) < 3:
                blocking_examples[issue].append({
                    "question_id": r["question_id"],
                    "dimension": r["dimension"],
                    "issue": issue,
                    "all_issues": r["issues"],
                })

    # Write outputs
    _write_jsonl(rows_output_path, results)
    _write_json(output_dir / "candidate_quality_summary.json", summary)

    # blocking_examples.jsonl — flatten
    examples_flat: List[dict] = []
    for issue_type, examples in sorted(blocking_examples.items()):
        for ex in examples:
            examples_flat.append(ex)
    _write_jsonl(output_dir / "blocking_examples.jsonl", examples_flat)

    logger.info(
        f"Audit complete: {len(results)} candidates | "
        f"eligible={eligible_count} | blocked={blocked_count}"
    )

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_audit_cli() -> int:
    """CLI entry point for audit module."""
    ap = argparse.ArgumentParser(
        description="Quality audit for I2V-CompBench candidate pool"
    )
    ap.add_argument(
        "--root",
        default="data/benchmark_dataset",
        help="Path to benchmark dataset root directory",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <root>/quality_experiments)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of candidates to audit",
    )
    ap.add_argument(
        "--only-qid",
        default=None,
        help="Audit only a single question_id (for debugging)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed candidates from existing output",
    )
    ap.add_argument(
        "--min-words",
        type=int,
        default=8,
        help="Minimum prompt word count (default: 8)",
    )
    ap.add_argument(
        "--max-words",
        type=int,
        default=25,
        help="Maximum prompt word count (default: 25)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output) if args.output else root / "quality_experiments"

    config = {
        "min_prompt_words": args.min_words,
        "max_prompt_words": args.max_words,
    }

    summary = run_audit(
        benchmark_root=root,
        output_dir=output_dir,
        config=config,
        limit=args.limit,
        only_qid=args.only_qid,
        resume=args.resume,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_audit_cli())
