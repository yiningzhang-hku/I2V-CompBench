"""Audit the five-dimension Phase 2 pool and prepare quality experiment splits.

This script intentionally uses only the Python standard library so it can run before
the full model environment is installed. It does not mutate the source dataset.

Outputs under data/benchmark_dataset/quality_experiments/:
  - candidate_quality_audit.json
  - candidate_quality_rows.jsonl
  - development_250.jsonl
  - validation_250.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DIMENSIONS = (
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "background_dynamics",
    "view_transformation",
)

KNOWN_RARE_MODIFIERS = {
    "aura", "celestial", "contemplation", "contemplative", "eerie",
    "ethereal", "grotesque", "iridescent", "luminescent", "malevolence",
    "menacing", "motifs", "mystical", "radiant", "sclera", "solemn",
    "spectral", "swirls", "translucent",
}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_issues(prompt: str, min_words: int = 8, max_words: int = 25) -> List[str]:
    issues: List[str] = []
    text = (prompt or "").strip()
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if not text:
        return ["empty_prompt"]
    if re.search(r"\{[^{}]+\}", text):
        issues.append("unresolved_placeholder")
    if re.search(r"\b(the|a|an)\s+\1\b", text, flags=re.IGNORECASE):
        issues.append("repeated_article")
    if re.search(r"\b(the|a|an)\s*[,.;:]", text, flags=re.IGNORECASE):
        issues.append("empty_slot")
    if not min_words <= len(words) <= max_words:
        issues.append("word_count_out_of_range")
    return issues


def target_issues(row: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    subjects = row.get("target_subjects") or []
    if not subjects:
        issues.append("missing_target_subjects")
    for subject in subjects:
        noun = str(subject.get("noun") or "").strip()
        desc = str(subject.get("description") or "").strip().lower()
        if not noun:
            issues.append("missing_target_noun")
        if desc in {"", "subject", "the subject"}:
            issues.append("generic_target_description")
    relation = row.get("target_relation") or {}
    if not str(relation.get("value") or "").strip():
        issues.append("missing_target_change")
    return sorted(set(issues))


def resolve_image(root: Path, row: Dict[str, Any]) -> Tuple[str, bool]:
    qid = str(row.get("question_id") or "")
    dim = str(row.get("dimension") or "")
    raw = str(row.get("first_frame_path") or "").replace("\\", "/")
    candidates = []
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else root.parent.parent / p)
        candidates.append(root / p.name)
    candidates.extend(
        [
            root / "first_frames" / f"{qid}.png",
            root / "by_dimension" / dim / qid / "first_frame.png",
            root / "by_dimension_v1_409" / dim / qid / "first_frame.png",
        ]
    )
    for path in candidates:
        if path.exists():
            return str(path), True
    return raw, False


def allocate_proportionally(groups: Dict[Tuple[str, str], List[Dict[str, Any]]], n: int) -> Dict[Tuple[str, str], int]:
    total = sum(len(v) for v in groups.values())
    if total < n:
        raise ValueError(f"Need {n} rows but only {total} are available")
    raw = {k: n * len(v) / total for k, v in groups.items()}
    out = {k: min(len(groups[k]), int(value)) for k, value in raw.items()}
    remaining = n - sum(out.values())
    order = sorted(groups, key=lambda k: (raw[k] - int(raw[k]), len(groups[k])), reverse=True)
    while remaining:
        progressed = False
        for key in order:
            if out[key] < len(groups[key]):
                out[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("Unable to allocate split")
    return out


def stratified_take(rows: List[Dict[str, Any]], n: int, rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("difficulty")), str(row.get("semantic_rarity")))].append(row)
    for values in groups.values():
        rng.shuffle(values)
    allocation = allocate_proportionally(groups, n)
    selected: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    for key, values in groups.items():
        k = allocation.get(key, 0)
        selected.extend(values[:k])
        remaining.extend(values[k:])
    rng.shuffle(selected)
    return selected, remaining


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/benchmark_dataset")
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--per-dimension", type=int, default=50)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    output = root / "quality_experiments"
    manifests = list(iter_jsonl(root / "phase3_manifest.jsonl"))
    finals = {
        row["question_id"]: row
        for row in iter_jsonl(root / "prompts" / "final_prompts.jsonl")
    }

    audit_rows: List[Dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    dim_counts: Counter[str] = Counter()
    rare_counts: Counter[str] = Counter()
    source_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in manifests:
        dim = str(row.get("dimension") or "")
        if dim not in DIMENSIONS:
            continue
        qid = str(row.get("question_id") or "")
        final = finals.get(qid, {})
        prompt = str(row.get("prompt") or "")
        blockers = target_issues(row) + prompt_issues(prompt)
        if final.get("failed_check"):
            blockers.append("final_failed_check")
        image_path, image_exists = resolve_image(root, row)
        if not image_exists:
            blockers.append("missing_first_frame")
        tokens = {w.lower() for w in re.findall(r"[A-Za-z]+", prompt)}
        rare_hits = sorted(tokens & KNOWN_RARE_MODIFIERS)
        for hit in rare_hits:
            rare_counts[hit] += 1
        blockers = sorted(set(blockers))
        issue_counts.update(blockers)
        dim_counts[dim] += 1
        audit = {
            "question_id": qid,
            "dimension": dim,
            "difficulty": row.get("difficulty"),
            "semantic_rarity": row.get("semantic_rarity"),
            "used_fallback": bool(final.get("used_fallback")),
            "rare_modifier_hits": rare_hits,
            "resolved_image_path": image_path,
            "image_exists": image_exists,
            "blocking_issues": blockers,
            "eligible_without_model_repair": not blockers,
        }
        audit_rows.append(audit)
        source_rows[dim].append(row)

    rng = random.Random(args.seed)
    development: List[Dict[str, Any]] = []
    validation: List[Dict[str, Any]] = []
    for dim in DIMENSIONS:
        dev, remain = stratified_take(source_rows[dim], args.per_dimension, rng)
        val, _ = stratified_take(remain, args.per_dimension, rng)
        development.extend(dev)
        validation.extend(val)

    report = {
        "source_manifest": str(root / "phase3_manifest.jsonl"),
        "seed": args.seed,
        "dimensions": list(DIMENSIONS),
        "candidate_count": len(audit_rows),
        "candidate_count_by_dimension": dict(sorted(dim_counts.items())),
        "eligible_without_model_repair": sum(
            bool(r["eligible_without_model_repair"]) for r in audit_rows
        ),
        "prompt_and_file_gate_pass_by_dimension": dict(sorted(Counter(
            r["dimension"]
            for r in audit_rows
            if not (
                set(r["blocking_issues"])
                - {"missing_target_noun", "generic_target_description", "missing_target_change"}
            )
        ).items())),
        "blocking_issue_counts": dict(issue_counts.most_common()),
        "known_rare_modifier_counts": dict(rare_counts.most_common()),
        "development_count": len(development),
        "validation_count": len(validation),
        "notes": [
            "Development/validation splits intentionally include flawed rows for quality-method comparison.",
            "No official 1500-row selection is produced until target repair and quality gates pass.",
        ],
    }
    write_json(output / "candidate_quality_audit.json", report)
    write_jsonl(output / "candidate_quality_rows.jsonl", audit_rows)
    write_jsonl(output / "development_250.jsonl", development)
    write_jsonl(output / "validation_250.jsonl", validation)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
