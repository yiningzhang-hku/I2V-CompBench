"""
Phase 2 · Step 1: build quota plan.

Input  : configs/phase2.yaml + (optional) phase1_bundle/prior_package.json
Output : data/benchmark_dataset/quota_plan.json

Each bucket is a 4-tuple:
    (dimension, input_mode_or_subtype, difficulty, rarity) -> target_count

Pilot mode: num_per_dimension=20  (total 7*20=140)
Full  mode: num_per_dimension=200 (total 7*200=1400)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from ..schemas.phase2 import DIMENSIONS_V2, QuotaBucket, QuotaPlan
from ..utils.io import benchmark_paths, load_config, write_json


# ============================================================
# Helpers
# ============================================================

def _round_split(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    """Largest-remainder rounding so the sum of rounded counts equals `total`."""
    if total <= 0 or not ratios:
        return {k: 0 for k in ratios}
    raw = {k: total * float(v) for k, v in ratios.items()}
    base = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(base.values())
    if remainder > 0:
        # distribute remainder to the keys with largest fractional parts
        fracs = sorted(((raw[k] - base[k], k) for k in raw), reverse=True)
        for _, k in fracs[:remainder]:
            base[k] += 1
    elif remainder < 0:
        fracs = sorted(((raw[k] - base[k], k) for k in raw))
        for _, k in fracs[: -remainder]:
            base[k] = max(0, base[k] - 1)
    return base


def _bucket_id(dim: str, mode: str, difficulty: str, rarity: str) -> str:
    return f"{dim}__{mode}__{difficulty}__{rarity}"


# ============================================================
# Core
# ============================================================

def build_quota(config: Dict[str, Any]) -> QuotaPlan:
    mode: str = config.get("mode", "pilot")
    quota_cfg: Dict[str, Any] = config["quota"]
    if mode == "pilot":
        num_per_dim = int(quota_cfg.get("num_per_dimension_pilot", 20))
    else:
        num_per_dim = int(quota_cfg.get("num_per_dimension_full", 200))

    rarity_ratio: Dict[str, float] = quota_cfg.get("rarity", {"common": 0.8, "rare": 0.2})
    difficulty_ratio: Dict[str, float] = quota_cfg.get(
        "difficulty", {"easy": 0.4, "medium": 0.4, "hard": 0.2}
    )
    input_mode_ratio: Dict[str, Dict[str, float]] = quota_cfg.get("input_mode_ratio", {})
    contrastive_dims = set(
        (quota_cfg.get("contrastive_pair") or {}).get("enabled_dimensions") or []
    )
    skip_dims = set(quota_cfg.get("skip_dimensions") or [])

    buckets: List[QuotaBucket] = []
    for dim in DIMENSIONS_V2:
        if dim in skip_dims:
            continue
        modes = input_mode_ratio.get(dim) or {"single_image": 1.0}
        # validate ratios sum near 1.0
        s = sum(modes.values())
        if s <= 0:
            logger.warning(f"input_mode_ratio for {dim} sums to {s}; skipping")
            continue
        # split per_dim into modes
        modes_norm = {k: v / s for k, v in modes.items()}
        mode_counts = _round_split(num_per_dim, modes_norm)
        for mode_key, mc in mode_counts.items():
            if mc <= 0:
                continue
            diff_counts = _round_split(mc, difficulty_ratio)
            for diff_key, dc in diff_counts.items():
                if dc <= 0:
                    continue
                rarity_counts = _round_split(dc, rarity_ratio)
                for rk, rc in rarity_counts.items():
                    if rc <= 0:
                        continue
                    buckets.append(
                        QuotaBucket(
                            bucket_id=_bucket_id(dim, mode_key, diff_key, rk),
                            dimension=dim,
                            input_mode_or_subtype=mode_key,
                            difficulty=diff_key,  # type: ignore[arg-type]
                            rarity=rk,  # type: ignore[arg-type]
                            target_count=rc,
                            contrastive_pair_required=dim in contrastive_dims,
                        )
                    )

    plan = QuotaPlan(
        mode=mode,  # type: ignore[arg-type]
        num_per_dimension=num_per_dim,
        buckets=buckets,
    )
    total = sum(b.target_count for b in buckets)
    logger.info(
        f"Quota built: mode={mode}, dims={len(DIMENSIONS_V2)}, buckets={len(buckets)}, total_target={total}"
    )
    return plan


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 build_quota")
    parser.add_argument("--config", required=True, help="Path to configs/phase2.yaml")
    parser.add_argument("--mode", choices=["pilot", "full"], default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode

    paths = benchmark_paths(cfg["output_dir"])
    plan = build_quota(cfg)
    write_json(paths["quota_plan"], plan.model_dump())
    logger.info(f"Wrote quota plan -> {paths['quota_plan']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
