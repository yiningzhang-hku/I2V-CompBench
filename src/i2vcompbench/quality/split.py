"""开发集/验证集确定性划分模块

Provides stratified, deterministic splitting of the I2V-CompBench candidate pool
into non-overlapping development and validation subsets. Uses SHA-256 keyed ordering
and largest-remainder proportional allocation to ensure reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORMAL_DIMENSIONS = (
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "background_dynamics",
    "view_transformation",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stable_key(seed: int, question_id: str) -> str:
    """SHA256(str(seed) + '||' + question_id) 作为确定性排序键."""
    return hashlib.sha256(f"{seed}||{question_id}".encode("utf-8")).hexdigest()


def _largest_remainder_allocation(
    total: int, proportions: dict[str, float]
) -> dict[str, int]:
    """按比例分配整数名额，余额按小数部分从大到小分配.

    Args:
        total: 需分配的总数
        proportions: 各层的比例 (值之和应为1.0)

    Returns:
        各层分配的整数名额
    """
    if not proportions:
        return {}

    # Compute raw quotas
    raw = {k: total * v for k, v in proportions.items()}
    # Floor allocations
    floors = {k: int(v) for k, v in raw.items()}
    # Remainders
    remainders = {k: raw[k] - floors[k] for k in raw}
    # How many extra seats to distribute
    distributed = sum(floors.values())
    extra = total - distributed

    # Sort by remainder descending, break ties by key for determinism
    sorted_keys = sorted(remainders, key=lambda k: (-remainders[k], k))
    for i in range(extra):
        floors[sorted_keys[i]] += 1

    return floors


def _compute_strata_key(row: dict, stratify_by: list[str]) -> tuple:
    """Extract stratum tuple from a candidate row."""
    return tuple(row.get(f, "unknown") for f in stratify_by)


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------


def stratified_split(
    candidates: list[dict],
    n_per_dimension: int,
    seed: int,
    stratify_by: list[str] | None = None,
) -> list[dict]:
    """从候选中按维度分层抽样n条.

    分层因子：(difficulty, semantic_rarity) 的笛卡尔积
    每层按候选池中的比例分配名额，使用最大余数法分配余额。
    层内使用 SHA256(seed || question_id) 排序确保确定性。

    Args:
        candidates: 候选列表（必须包含 dimension, difficulty, semantic_rarity 字段）
        n_per_dimension: 每维度抽取数
        seed: 随机种子
        stratify_by: 分层字段，默认 ["difficulty", "semantic_rarity"]

    Returns:
        抽取的候选列表
    """
    if stratify_by is None:
        stratify_by = ["difficulty", "semantic_rarity"]

    selected: list[dict] = []

    # Group candidates by dimension
    by_dim: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        dim = c.get("dimension", "")
        if dim in FORMAL_DIMENSIONS:
            by_dim[dim].append(c)

    for dim in sorted(FORMAL_DIMENSIONS):
        pool = by_dim.get(dim, [])
        if not pool:
            logger.warning(f"维度 {dim} 无候选，跳过")
            continue

        n_take = min(n_per_dimension, len(pool))

        # Group pool by stratum
        strata: dict[tuple, list[dict]] = defaultdict(list)
        for row in pool:
            key = _compute_strata_key(row, stratify_by)
            strata[key].append(row)

        # Compute proportions from pool distribution
        total_pool = len(pool)
        proportions = {
            str(k): len(v) / total_pool for k, v in strata.items()
        }

        # Allocate quota per stratum
        quotas = _largest_remainder_allocation(n_take, proportions)

        # Select from each stratum using deterministic ordering
        for stratum_key, rows in sorted(strata.items(), key=lambda x: str(x[0])):
            quota = quotas.get(str(stratum_key), 0)
            if quota <= 0:
                continue

            # Sort by stable hash key
            sorted_rows = sorted(
                rows, key=lambda r: _stable_key(seed, r["question_id"])
            )
            selected.extend(sorted_rows[:quota])

    return selected


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_split(
    benchmark_root: Path,
    output_dir: Path,
    config: dict | None = None,
    seed: int = 20260712,
    dev_per_dim: int = 50,
    val_per_dim: int = 50,
) -> dict:
    """执行完整划分流程.

    流程：
    1. 读取 phase3_manifest.jsonl
    2. 过滤仅保留5个正式维度
    3. 第一步抽取开发集（每维50条）
    4. 从剩余候选中抽取验证集（每维50条）
    5. 验证硬约束
    6. 输出文件

    硬约束检查：
    - 开发集和验证集 question_id 完全不重叠
    - 同一 source_sample_id 不得跨集合
    - 每维度精确数量
    - 总数：开发集250条、验证集250条

    输出文件（到 output_dir/splits/ 下）：
    - development_250.jsonl — 开发集完整行
    - validation_250.jsonl — 验证集完整行
    - split_summary.json — 划分统计
    - split_hash.txt — 两个集合的确定性hash

    Args:
        benchmark_root: benchmark数据集根目录
        output_dir: 输出目录
        config: 可选配置覆盖
        seed: 随机种子
        dev_per_dim: 每维度开发集数量
        val_per_dim: 每维度验证集数量

    Returns:
        summary dict
    """
    # Override from config if provided
    if config:
        seed = config.get("seed", seed)
        dev_per_dim = config.get("dev_per_dimension", dev_per_dim)
        val_per_dim = config.get("val_per_dimension", val_per_dim)

    manifest_path = benchmark_root / "phase3_manifest.jsonl"
    assets_path = benchmark_root / "input_assets_manifest.jsonl"

    logger.info(f"读取 manifest: {manifest_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Step 1: Load all candidates
    all_candidates: list[dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_candidates.append(json.loads(line))

    logger.info(f"加载 {len(all_candidates)} 条候选")

    # Step 2: Filter to formal dimensions only
    formal_candidates = [
        c for c in all_candidates if c.get("dimension") in FORMAL_DIMENSIONS
    ]
    logger.info(
        f"过滤后保留 {len(formal_candidates)} 条（5个正式维度）"
    )

    # Step 3: Extract development set
    dev_set = stratified_split(
        formal_candidates, n_per_dimension=dev_per_dim, seed=seed
    )
    dev_qids = {r["question_id"] for r in dev_set}
    logger.info(f"开发集抽取: {len(dev_set)} 条")

    # Step 4: Extract validation set from remaining
    remaining = [c for c in formal_candidates if c["question_id"] not in dev_qids]
    val_set = stratified_split(
        remaining, n_per_dimension=val_per_dim, seed=seed
    )
    val_qids = {r["question_id"] for r in val_set}
    logger.info(f"验证集抽取: {len(val_set)} 条")

    # Step 5: Hard constraint checks
    # 5a. No overlap in question_id
    overlap = dev_qids & val_qids
    if overlap:
        raise ValueError(
            f"开发集与验证集存在 {len(overlap)} 个重叠 question_id: "
            f"{sorted(overlap)[:5]}..."
        )
    overlap_check = "pass"

    # 5b. Source crossover check
    source_crossover_check = _check_source_crossover(
        dev_set, val_set, assets_path
    )

    # 5c. Per-dimension count check
    dev_dim_counts = Counter(r["dimension"] for r in dev_set)
    val_dim_counts = Counter(r["dimension"] for r in val_set)
    for dim in FORMAL_DIMENSIONS:
        if dev_dim_counts.get(dim, 0) != dev_per_dim:
            logger.warning(
                f"开发集 {dim} 数量不足: "
                f"{dev_dim_counts.get(dim, 0)}/{dev_per_dim}"
            )
        if val_dim_counts.get(dim, 0) != val_per_dim:
            logger.warning(
                f"验证集 {dim} 数量不足: "
                f"{val_dim_counts.get(dim, 0)}/{val_per_dim}"
            )

    # Step 6: Compute statistics
    dev_total = len(dev_set)
    val_total = len(val_set)

    dev_by_difficulty = dict(Counter(r.get("difficulty", "unknown") for r in dev_set))
    dev_by_rarity = dict(Counter(r.get("semantic_rarity", "unknown") for r in dev_set))
    val_by_difficulty = dict(Counter(r.get("difficulty", "unknown") for r in val_set))
    val_by_rarity = dict(Counter(r.get("semantic_rarity", "unknown") for r in val_set))

    # Compute hashes
    dev_hash = _compute_set_hash(dev_set)
    val_hash = _compute_set_hash(val_set)
    combined_hash = hashlib.sha256(
        (dev_hash + val_hash).encode("utf-8")
    ).hexdigest()

    summary = {
        "seed": seed,
        "dev_per_dimension": dev_per_dim,
        "val_per_dimension": val_per_dim,
        "total_candidates": len(all_candidates),
        "formal_dimensions": len(FORMAL_DIMENSIONS),
        "development": {
            "total": dev_total,
            "by_dimension": dict(sorted(dev_dim_counts.items())),
            "by_difficulty": dict(sorted(dev_by_difficulty.items())),
            "by_rarity": dict(sorted(dev_by_rarity.items())),
        },
        "validation": {
            "total": val_total,
            "by_dimension": dict(sorted(val_dim_counts.items())),
            "by_difficulty": dict(sorted(val_by_difficulty.items())),
            "by_rarity": dict(sorted(val_by_rarity.items())),
        },
        "overlap_check": overlap_check,
        "source_crossover_check": source_crossover_check,
        "split_hash": combined_hash,
    }

    # Step 7: Write output files
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(splits_dir / "development_250.jsonl", dev_set)
    _write_jsonl(splits_dir / "validation_250.jsonl", val_set)

    with open(splits_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    hash_content = (
        f"development_sha256={dev_hash}\n"
        f"validation_sha256={val_hash}\n"
        f"combined_sha256={combined_hash}\n"
    )
    with open(splits_dir / "split_hash.txt", "w", encoding="utf-8") as f:
        f.write(hash_content)

    logger.info(
        f"划分完成 → {splits_dir} | "
        f"dev={dev_total} val={val_total} hash={combined_hash[:16]}..."
    )

    return summary


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSONL (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compute_set_hash(rows: list[dict]) -> str:
    """Compute SHA-256 of sorted question_ids joined by newline."""
    qids = sorted(r["question_id"] for r in rows)
    payload = "\n".join(qids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check_source_crossover(
    dev_set: list[dict],
    val_set: list[dict],
    assets_path: Path,
) -> str:
    """检查 source_ref_id 跨集合情况.

    尝试从 input_assets_manifest.jsonl 加载 question_id → source_ref_id 映射。
    如果文件不存在或字段缺失，标记检查为 "skipped"。

    Returns:
        "pass" | "skipped" | "fail:<details>"
    """
    if not assets_path.exists():
        logger.info(
            f"input_assets_manifest.jsonl 不存在，跳过 source crossover 检查"
        )
        return "skipped"

    # Build qid → source_ref_id mapping
    qid_to_source: dict[str, str] = {}
    try:
        with open(assets_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                qid = record.get("question_id", "")
                assets = record.get("assets", [])
                for asset in assets:
                    src_ref = asset.get("source_ref_id")
                    if src_ref:
                        qid_to_source[qid] = src_ref
                        break  # Use first asset's source_ref_id
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"解析 input_assets_manifest.jsonl 异常: {e}")
        return "skipped"

    if not qid_to_source:
        logger.info("无 source_ref_id 数据，跳过 crossover 检查")
        return "skipped"

    # Collect source_ref_ids for each set
    dev_sources: set[str] = set()
    for r in dev_set:
        src = qid_to_source.get(r["question_id"])
        if src:
            dev_sources.add(src)

    val_sources: set[str] = set()
    for r in val_set:
        src = qid_to_source.get(r["question_id"])
        if src:
            val_sources.add(src)

    crossover = dev_sources & val_sources
    if crossover:
        logger.warning(
            f"发现 {len(crossover)} 个 source_ref_id 跨集合: "
            f"{sorted(crossover)[:5]}..."
        )
        return f"fail:{len(crossover)}_sources_crossed"

    return "pass"
