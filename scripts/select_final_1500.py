"""
Final 1500 Benchmark Selection Script
======================================
从 eligible 候选池中筛选出1500条最终benchmark（每维度300条）。

筛选逻辑：
1. 资格过滤：audit eligible + prompt rules非阻塞
2. 质量排序：按清晰度（Laplacian方差）降序
3. 配额分配：最大余数法按difficulty和semantic_rarity分层
4. 确定性：SHA256排序保证可复现

输出：
- data/benchmark_dataset/final_benchmark_1500.jsonl
- data/benchmark_dataset/final_1500/statistics.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 20260712
TARGET_PER_DIMENSION = 300
FORMAL_DIMENSIONS = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "background_dynamics",
    "view_transformation",
]

# Target ratios (soft goals - best-effort allocation)
DIFFICULTY_RATIO = {"easy": 0.40, "medium": 0.35, "hard": 0.25}
RARITY_RATIO = {"common": 0.55, "rare": 0.45}

# Prompt issues that are truly blocking (not rare_modifier)
PROMPT_BLOCKING_ISSUES = frozenset([
    "forbidden_word",
    "missing_change_verb",
    "word_count_too_long",
    "word_count_too_short",
    "empty_prompt",
    "unresolved_placeholder",
])

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = PROJECT_ROOT / "data" / "benchmark_dataset"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def stable_key(seed: int, question_id: str) -> str:
    """SHA256 deterministic sort key."""
    return hashlib.sha256(f"{seed}||{question_id}".encode("utf-8")).hexdigest()


def largest_remainder_allocation(total: int, proportions: Dict[str, float]) -> Dict[str, int]:
    """最大余数法分配整数名额."""
    if not proportions:
        return {}
    raw = {k: total * v for k, v in proportions.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remainders = {k: raw[k] - floors[k] for k in raw}
    distributed = sum(floors.values())
    extra = total - distributed
    sorted_keys = sorted(remainders, key=lambda k: (-remainders[k], k))
    for i in range(extra):
        floors[sorted_keys[i]] += 1
    return floors


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    rows = []
    if not path.exists():
        print(f"  [WARN] File not found: {path}")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_audit_eligible_set() -> set:
    """Load eligible question_ids from audit results."""
    audit_path = BENCHMARK_ROOT / "quality_experiments" / "final_audit" / "candidate_quality_rows.jsonl"
    eligible_qids = set()
    for row in load_jsonl(audit_path):
        if row.get("eligible", False):
            eligible_qids.add(row["question_id"])
    print(f"  Audit: {len(eligible_qids)} eligible question_ids loaded")
    return eligible_qids


def load_prompt_blocking_set() -> set:
    """Load question_ids with blocking prompt issues."""
    # Check latest prompt rules results
    prompt_results_path = (
        BENCHMARK_ROOT / "quality_experiments" / "20260713_093828" / "prompt" / "prompt_rules_results.jsonl"
    )
    if not prompt_results_path.exists():
        # Try to find any prompt results
        qe_root = BENCHMARK_ROOT / "quality_experiments"
        for run_dir in sorted(qe_root.iterdir(), reverse=True):
            candidate = run_dir / "prompt" / "prompt_rules_results.jsonl"
            if candidate.exists():
                prompt_results_path = candidate
                break

    blocked_qids = set()
    if not prompt_results_path.exists():
        print("  [WARN] No prompt_rules_results found, skipping prompt filter")
        return blocked_qids

    for row in load_jsonl(prompt_results_path):
        issues = row.get("issues", [])
        for issue in issues:
            # Issue format: "issue_type:detail" or "issue_type"
            issue_type = issue.split(":")[0] if ":" in issue else issue
            if issue_type in PROMPT_BLOCKING_ISSUES:
                blocked_qids.add(row["question_id"])
                break

    print(f"  Prompt rules: {len(blocked_qids)} blocked by prompt issues")
    return blocked_qids


def load_clarity_scores() -> Dict[str, float]:
    """Load Laplacian clarity scores from enhance report."""
    clarity_path = BENCHMARK_ROOT / "quality_experiments" / "clarity_enhance_report.json"
    scores = {}
    if not clarity_path.exists():
        print("  [WARN] clarity_enhance_report.json not found")
        return scores

    with open(clarity_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    for detail in report.get("details", []):
        filename = detail.get("file", "")
        laplacian = detail.get("laplacian_after")
        if filename and laplacian is not None:
            # Map filename to question_id pattern
            # filename format: act_single_0001.png → question_id format in manifest
            scores[filename] = float(laplacian)

    print(f"  Clarity: {len(scores)} scores loaded (mean={sum(scores.values())/max(len(scores),1):.1f})")
    return scores


def match_clarity_score(row: Dict[str, Any], clarity_scores: Dict[str, float]) -> float:
    """Match a candidate row to its clarity score."""
    # Try original_frame_path filename
    orig_path = row.get("original_frame_path", "")
    if orig_path:
        filename = Path(orig_path).name
        if filename in clarity_scores:
            return clarity_scores[filename]

    # Try first_frame_path (without _16x9 suffix)
    ff_path = row.get("first_frame_path", "")
    if ff_path:
        filename = Path(ff_path).name
        if filename in clarity_scores:
            return clarity_scores[filename]
        # Try without _16x9
        base = filename.replace("_16x9", "")
        if base in clarity_scores:
            return clarity_scores[base]

    # Fallback: try question_id based pattern
    qid = row.get("question_id", "")
    # qid like attr_single_0001 → filename like attr_single_0001.png
    candidate_name = f"{qid}.png"
    if candidate_name in clarity_scores:
        return clarity_scores[candidate_name]

    return 0.0  # No score available


# ---------------------------------------------------------------------------
# Selection Logic
# ---------------------------------------------------------------------------


def select_from_pool(
    pool: List[Dict[str, Any]],
    n: int,
    clarity_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    """从维度候选池中选取n条，兼顾配额和质量.

    策略：
    1. 按difficulty分配配额（最大余数法，软目标）
    2. 在每个difficulty层内按semantic_rarity分配
    3. 每个最终层内按clarity降序+SHA256排序
    4. 如果某层不足，将缺额重新分配给其他层
    """
    if len(pool) <= n:
        # 池子不够大，全部选取
        return sorted(pool, key=lambda r: stable_key(SEED, r["question_id"]))

    # Attach clarity score and stable key to each row
    for row in pool:
        row["_clarity"] = match_clarity_score(row, clarity_scores)
        row["_stable_key"] = stable_key(SEED, row["question_id"])

    # Phase 1: Allocate by difficulty
    diff_groups = defaultdict(list)
    for row in pool:
        diff_groups[row.get("difficulty", "unknown")].append(row)

    # Compute proportional quotas based on target ratio, capped by available
    avail_diffs = {d: len(rows) for d, rows in diff_groups.items()}
    target_diff_quota = largest_remainder_allocation(n, DIFFICULTY_RATIO)

    # Adjust quotas based on availability
    final_diff_quota = {}
    shortfall = 0
    surplus_diffs = []
    for d in DIFFICULTY_RATIO:
        available = avail_diffs.get(d, 0)
        target = target_diff_quota.get(d, 0)
        if available < target:
            final_diff_quota[d] = available
            shortfall += target - available
        else:
            final_diff_quota[d] = target
            surplus_diffs.append(d)

    # Redistribute shortfall to surplus groups proportionally
    if shortfall > 0 and surplus_diffs:
        surplus_total = sum(avail_diffs[d] - final_diff_quota[d] for d in surplus_diffs)
        if surplus_total > 0:
            for d in surplus_diffs:
                extra_available = avail_diffs[d] - final_diff_quota[d]
                extra = min(int(shortfall * extra_available / surplus_total + 0.5), extra_available)
                final_diff_quota[d] += extra
                shortfall -= extra
            # Any remaining shortfall goes to first surplus with capacity
            for d in surplus_diffs:
                if shortfall <= 0:
                    break
                extra_available = avail_diffs[d] - final_diff_quota[d]
                give = min(shortfall, extra_available)
                final_diff_quota[d] += give
                shortfall -= give

    # Phase 2: Within each difficulty, select by rarity + quality
    selected = []
    for diff_level, quota in final_diff_quota.items():
        if quota <= 0:
            continue
        diff_pool = diff_groups.get(diff_level, [])
        if not diff_pool:
            continue

        # Sub-allocate by semantic_rarity
        rarity_groups = defaultdict(list)
        for row in diff_pool:
            rarity_groups[row.get("semantic_rarity", "unknown")].append(row)

        avail_rarities = {r: len(rows) for r, rows in rarity_groups.items()}
        target_rarity_quota = largest_remainder_allocation(quota, RARITY_RATIO)

        # Adjust rarity quotas
        final_rarity_quota = {}
        rarity_shortfall = 0
        rarity_surplus = []
        for r in RARITY_RATIO:
            available = avail_rarities.get(r, 0)
            target = target_rarity_quota.get(r, 0)
            if available < target:
                final_rarity_quota[r] = available
                rarity_shortfall += target - available
            else:
                final_rarity_quota[r] = target
                rarity_surplus.append(r)

        # Redistribute rarity shortfall
        if rarity_shortfall > 0 and rarity_surplus:
            for r in rarity_surplus:
                extra_available = avail_rarities.get(r, 0) - final_rarity_quota[r]
                give = min(rarity_shortfall, extra_available)
                final_rarity_quota[r] += give
                rarity_shortfall -= give
                if rarity_shortfall <= 0:
                    break

        # Select from each rarity group by clarity desc, then stable_key
        for rarity, rq in final_rarity_quota.items():
            if rq <= 0:
                continue
            candidates = rarity_groups.get(rarity, [])
            # Sort by clarity descending, then stable key for determinism
            candidates.sort(key=lambda r: (-r["_clarity"], r["_stable_key"]))
            selected.extend(candidates[:rq])

    # Verify we got enough (handle edge case where rounding lost some)
    if len(selected) < n:
        # Fill remaining from unselected pool by quality
        selected_qids = {r["question_id"] for r in selected}
        remaining = [r for r in pool if r["question_id"] not in selected_qids]
        remaining.sort(key=lambda r: (-r["_clarity"], r["_stable_key"]))
        needed = n - len(selected)
        selected.extend(remaining[:needed])

    # Truncate if somehow over (shouldn't happen)
    selected = selected[:n]

    # Clean temporary fields
    for row in pool:
        row.pop("_clarity", None)
        row.pop("_stable_key", None)

    return selected


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("  I2V-CompBench Final 1500 Selection")
    print("=" * 70)
    print()

    # Step 1: Load data
    print("[1/6] Loading data sources...")
    manifest = load_jsonl(BENCHMARK_ROOT / "phase3_manifest.jsonl")
    print(f"  Manifest: {len(manifest)} candidates")

    # Step 2: Load filters
    print("\n[2/6] Loading quality filters...")
    eligible_qids = load_audit_eligible_set()
    prompt_blocked_qids = load_prompt_blocking_set()
    clarity_scores = load_clarity_scores()

    # Step 3: Apply filters
    print("\n[3/6] Applying eligibility filters...")
    # Filter 1: audit eligible
    pool = [r for r in manifest if r["question_id"] in eligible_qids]
    print(f"  After audit filter: {len(pool)}")

    # Filter 2: prompt rules (remove blocking issues, keep rare_modifier as OK)
    pool = [r for r in pool if r["question_id"] not in prompt_blocked_qids]
    print(f"  After prompt filter: {len(pool)}")

    # Filter 3: must be in formal dimensions
    pool = [r for r in pool if r.get("dimension") in FORMAL_DIMENSIONS]
    print(f"  After dimension filter: {len(pool)}")

    # Report per-dimension availability
    dim_avail = Counter(r["dimension"] for r in pool)
    print("\n  Available per dimension:")
    for dim in FORMAL_DIMENSIONS:
        count = dim_avail.get(dim, 0)
        status = "OK" if count >= TARGET_PER_DIMENSION else "TIGHT"
        print(f"    {dim}: {count} [{status}]")

    # Step 4: Select per dimension
    print(f"\n[4/6] Selecting {TARGET_PER_DIMENSION} per dimension...")
    final_selected = []
    dim_stats = {}

    for dim in FORMAL_DIMENSIONS:
        dim_pool = [r for r in pool if r["dimension"] == dim]
        selected = select_from_pool(dim_pool, TARGET_PER_DIMENSION, clarity_scores)
        final_selected.extend(selected)

        # Stats
        diff_dist = dict(Counter(r["difficulty"] for r in selected))
        rarity_dist = dict(Counter(r["semantic_rarity"] for r in selected))
        dim_stats[dim] = {
            "pool_size": len(dim_pool),
            "selected": len(selected),
            "difficulty_distribution": diff_dist,
            "rarity_distribution": rarity_dist,
        }
        print(f"  {dim}: {len(selected)}/{TARGET_PER_DIMENSION} "
              f"(diff={diff_dist}, rarity={rarity_dist})")

    # Step 5: Validate
    print(f"\n[5/6] Validation...")
    total = len(final_selected)
    qids = [r["question_id"] for r in final_selected]
    unique_qids = set(qids)
    dup_count = total - len(unique_qids)

    print(f"  Total selected: {total}")
    print(f"  Unique question_ids: {len(unique_qids)}")
    print(f"  Duplicates: {dup_count}")

    dim_counts = Counter(r["dimension"] for r in final_selected)
    all_dims_300 = all(dim_counts.get(d, 0) == TARGET_PER_DIMENSION for d in FORMAL_DIMENSIONS)
    print(f"  All dimensions = {TARGET_PER_DIMENSION}: {all_dims_300}")

    if total != 1500:
        print(f"  [WARN] Total is {total}, not 1500!")
    if dup_count > 0:
        print(f"  [WARN] {dup_count} duplicate question_ids found!")

    # Step 6: Output
    print(f"\n[6/6] Writing output files...")

    # Sort final selection by dimension then question_id for reproducibility
    final_selected.sort(key=lambda r: (r["dimension"], r["question_id"]))

    # Write final_benchmark_1500.jsonl
    output_path = BENCHMARK_ROOT / "final_benchmark_1500.jsonl"
    write_jsonl(output_path, final_selected)
    print(f"  Written: {output_path}")

    # Generate statistics
    overall_diff = dict(Counter(r["difficulty"] for r in final_selected))
    overall_rarity = dict(Counter(r["semantic_rarity"] for r in final_selected))
    overall_subtype = dict(Counter(r.get("subtype", "unknown") for r in final_selected))

    statistics = {
        "seed": SEED,
        "total_selected": total,
        "target_per_dimension": TARGET_PER_DIMENSION,
        "dimensions": FORMAL_DIMENSIONS,
        "per_dimension": dim_stats,
        "overall_difficulty_distribution": overall_diff,
        "overall_rarity_distribution": overall_rarity,
        "overall_subtype_distribution": overall_subtype,
        "difficulty_target_ratio": DIFFICULTY_RATIO,
        "rarity_target_ratio": RARITY_RATIO,
        "validation": {
            "total_equals_1500": total == 1500,
            "all_dimensions_300": all_dims_300,
            "no_duplicates": dup_count == 0,
        },
        "deterministic_hash": hashlib.sha256(
            "\n".join(sorted(qids)).encode("utf-8")
        ).hexdigest(),
    }

    stats_path = BENCHMARK_ROOT / "final_1500" / "statistics.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    print(f"  Written: {stats_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  FINAL SELECTION SUMMARY")
    print("=" * 70)
    print(f"\n  Total: {total} candidates")
    print(f"  Deterministic hash: {statistics['deterministic_hash'][:16]}...")
    print(f"\n  {'Dimension':<25} {'Count':>6} {'Easy':>6} {'Med':>6} {'Hard':>6} {'Common':>8} {'Rare':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for dim in FORMAL_DIMENSIONS:
        ds = dim_stats[dim]
        dd = ds["difficulty_distribution"]
        rd = ds["rarity_distribution"]
        print(f"  {dim:<25} {ds['selected']:>6} "
              f"{dd.get('easy',0):>6} {dd.get('medium',0):>6} {dd.get('hard',0):>6} "
              f"{rd.get('common',0):>8} {rd.get('rare',0):>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    print(f"  {'TOTAL':<25} {total:>6} "
          f"{overall_diff.get('easy',0):>6} {overall_diff.get('medium',0):>6} "
          f"{overall_diff.get('hard',0):>6} "
          f"{overall_rarity.get('common',0):>8} {overall_rarity.get('rare',0):>6}")
    print(f"\n  Target difficulty ratio: easy={DIFFICULTY_RATIO['easy']:.0%} "
          f"medium={DIFFICULTY_RATIO['medium']:.0%} hard={DIFFICULTY_RATIO['hard']:.0%}")
    print(f"  Actual difficulty ratio: easy={overall_diff.get('easy',0)/max(total,1):.1%} "
          f"medium={overall_diff.get('medium',0)/max(total,1):.1%} "
          f"hard={overall_diff.get('hard',0)/max(total,1):.1%}")
    print(f"\n  Target rarity ratio: common={RARITY_RATIO['common']:.0%} rare={RARITY_RATIO['rare']:.0%}")
    print(f"  Actual rarity ratio: common={overall_rarity.get('common',0)/max(total,1):.1%} "
          f"rare={overall_rarity.get('rare',0)/max(total,1):.1%}")

    # Success/failure message
    print("\n" + "=" * 70)
    if total == 1500 and all_dims_300 and dup_count == 0:
        print("  SUCCESS: Final benchmark selection complete!")
    else:
        print("  WARNING: Selection has issues - check above for details")
    print("=" * 70)

    return 0 if (total == 1500 and all_dims_300 and dup_count == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
