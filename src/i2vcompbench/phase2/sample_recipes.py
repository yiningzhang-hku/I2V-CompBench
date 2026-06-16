"""
Phase 2 · Step 2: sample recipes from candidate_recipes.jsonl according to quota.

Input:
  - data/benchmark_dataset/quota_plan.json          (from build_quota)
  - <phase1_bundle_dir>/candidate_recipes.jsonl
  - <phase1_bundle_dir>/reference_bank/assets.jsonl (for multi_image feasibility)

Output:
  - data/benchmark_dataset/sampled_recipes.jsonl    (List[SampledRecipe])
  - data/benchmark_dataset/quota_unfilled_report.json

Hard rules:
  - Do NOT silently downgrade input_mode (multi -> single). Unfilled multi-buckets
    must be reported, never satisfied with single-image recipes.
  - quality_flags can be present but blocking flags filter the recipe out.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..schemas.phase2 import SampledRecipe
from ..utils.ids import stable_pair_id
from ..utils.io import (
    Phase1Bundle,
    benchmark_paths,
    iter_jsonl,
    load_config,
    read_json,
    write_json,
    write_jsonl,
)


# Subtype keys that imply multi_image even if quota stores them as input_mode_or_subtype.
_MULTI_IMAGE_SUBTYPES = {"type_c_multi_motion"}
_BLOCKING_QUALITY_FLAGS = {
    "low_alignment",
    "missing_inpainted_scene",
    "subject_not_visible",
    "evaluator_infeasible",
}

# Phase 1 contrastive_spec.contrast_type → Phase 2 contrastive_role 枚举映射
# 文档§4.7 step 5 / §8 要求 6 项 contrastive_role 枚举
CONTRAST_TYPE_TO_ROLE: Dict[str, str] = {
    "static_copy": "baseline_static_copy",
    "random_motion": "baseline_random_motion",
    "global_filter": "baseline_global_filter",
    "camera_pan_cheat": "baseline_camera_pan_cheat",
    "subject_swap_inverse": "baseline_subject_swap",
    "subject_swap": "baseline_subject_swap",
}


# ============================================================
# Filtering
# ============================================================

def _bucket_input_mode(input_mode_or_subtype: str) -> str:
    if input_mode_or_subtype in ("single_image", "multi_image"):
        return input_mode_or_subtype
    if input_mode_or_subtype in _MULTI_IMAGE_SUBTYPES:
        return "multi_image"
    # heuristic: any subtype with explicit "multi" word
    if "multi" in input_mode_or_subtype.lower():
        return "multi_image"
    return "single_image"


def _bucket_subtype(input_mode_or_subtype: str) -> str:
    if input_mode_or_subtype in ("single_image", "multi_image"):
        return ""
    return input_mode_or_subtype


def _recipe_input_mode(recipe: Dict[str, Any]) -> str:
    return str(recipe.get("input_mode") or "single_image")


def _recipe_subtype(recipe: Dict[str, Any]) -> str:
    # subtype field is not in CandidateRecipe schema; check `notes` / `meta` / `subtype`
    return str(recipe.get("subtype") or recipe.get("notes_subtype") or "")


def _recipe_difficulty(recipe: Dict[str, Any]) -> str:
    return str(recipe.get("expected_difficulty") or "medium")


def _recipe_rarity(recipe: Dict[str, Any]) -> str:
    """Phase 1 may not always carry semantic_rarity; fall back to 'common'."""
    return str(recipe.get("semantic_rarity") or "common")


def _recipe_blocking(recipe: Dict[str, Any]) -> bool:
    flags = set(recipe.get("quality_flags") or [])
    return bool(flags & _BLOCKING_QUALITY_FLAGS)


def _multi_image_assets_ok(
    recipe: Dict[str, Any],
    asset_index: Dict[str, Dict[str, Any]],
    min_quality_score: float,
    require_clean_bg: bool,
) -> bool:
    ref_ids: List[str] = list(recipe.get("reference_asset_ids") or [])
    if len(ref_ids) < 2:
        return False
    for rid in ref_ids:
        a = asset_index.get(rid)
        if not a:
            return False
        q = a.get("quality") or {}
        if float(q.get("quality_score") or 0.0) < min_quality_score:
            return False
        if require_clean_bg and not bool(q.get("is_clean_background", False)):
            # tolerate scene_reference_inpainted assets which are inherently clean
            if a.get("asset_type") not in ("scene_reference_inpainted",):
                return False
    return True


def _filter_recipes_for_bucket(
    recipes: List[Dict[str, Any]],
    dimension: str,
    bucket_input_mode: str,
    bucket_subtype: str,
    difficulty: str,
    rarity: str,
    asset_index: Dict[str, Dict[str, Any]],
    construct_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    min_q = float(construct_cfg.get("multi_reference_min_quality_score", 0.4))
    require_clean = bool(construct_cfg.get("reference_required_clean_background", True))
    for r in recipes:
        if r.get("target_dimension") != dimension:
            continue
        if _recipe_input_mode(r) != bucket_input_mode:
            continue
        if bucket_subtype and _recipe_subtype(r) and _recipe_subtype(r) != bucket_subtype:
            continue
        if _recipe_difficulty(r) != difficulty:
            continue
        if _recipe_rarity(r) != rarity:
            # rarity is best-effort: if recipe lacks the field, treat 'common' as wildcard match
            if not (rarity == "common" and not r.get("semantic_rarity")):
                continue
        if _recipe_blocking(r):
            continue
        if bucket_input_mode == "multi_image":
            if not _multi_image_assets_ok(r, asset_index, min_q, require_clean):
                continue
        out.append(r)
    return out


# ============================================================
# Contrastive pairing
# ============================================================

def _is_contrastive_recipe(r: Dict[str, Any]) -> bool:
    cs = r.get("contrastive_spec") or []
    return bool(cs)


def _pair_recipes(
    candidates: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]]:
    """
    Try to organize candidates into (original, baseline, baseline_role) triples
    using Phase 1 contrastive_spec metadata.

    · carrying contrastive_spec 的 recipe 被当作 baseline，联动到同 source_sample_id 下的
      原 recipe（作为 original）。
    · 提取出 contrastive_spec[0].contrast_type（如 subject_swap_inverse）转为 baseline_*
      贫枚举，返回给上层赋值到 contrastive_role。
    · 无 contrast 伴生的 recipe 返回 (recipe, None, None) 表示 single-original。
    """
    by_sid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in candidates:
        by_sid[str(r.get("source_sample_id") or r.get("recipe_id"))].append(r)

    out: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]] = []
    used: set = set()
    for sid, group in by_sid.items():
        contr = [r for r in group if _is_contrastive_recipe(r)]
        plain = [r for r in group if not _is_contrastive_recipe(r)]

        # 每个 contrastive recipe 是一个 baseline。original 优先从同 sid 的 plain 里取。
        for b in contr:
            cs = b.get("contrastive_spec") or []
            ctype = ""
            if cs and isinstance(cs, list) and isinstance(cs[0], dict):
                ctype = str(cs[0].get("contrast_type") or "")
            role = CONTRAST_TYPE_TO_ROLE.get(ctype)
            if role is None:
                # 未知 contrast_type——退化为 static_copy baseline
                role = "baseline_static_copy"
            origin: Optional[Dict[str, Any]] = None
            if plain:
                origin = plain.pop(0)
            out.append((origin if origin is not None else b, b, role) if origin is not None else (b, None, None))
            used.add(b["recipe_id"])
            if origin is not None:
                used.add(origin["recipe_id"])

        for r in plain:
            if r["recipe_id"] in used:
                continue
            out.append((r, None, None))
    return out


# ============================================================
# Core sampling
# ============================================================

def sample_recipes(
    config: Dict[str, Any],
    seed: int = 0,
) -> Tuple[List[SampledRecipe], Dict[str, Any]]:
    paths = benchmark_paths(config["output_dir"])
    quota_plan = read_json(paths["quota_plan"])
    if not quota_plan or "buckets" not in quota_plan:
        raise FileNotFoundError(
            f"quota_plan.json not found or invalid at {paths['quota_plan']}; run build_quota first"
        )

    bundle = Phase1Bundle(config["phase1_bundle_dir"])
    recipes_all = bundle.recipes
    asset_index = bundle.assets
    if not recipes_all:
        raise RuntimeError(
            f"No candidate_recipes.jsonl in {config['phase1_bundle_dir']}; cannot sample"
        )
    logger.info(f"Loaded {len(recipes_all)} candidate recipes for sampling")

    construct_cfg = config.get("construct", {})
    rng = random.Random(seed)

    sampled: List[SampledRecipe] = []
    unfilled: List[Dict[str, Any]] = []
    used_recipe_ids: set[str] = set()

    for bucket in quota_plan["buckets"]:
        dim = bucket["dimension"]
        bm = _bucket_input_mode(bucket["input_mode_or_subtype"])
        bsub = _bucket_subtype(bucket["input_mode_or_subtype"])
        difficulty = bucket["difficulty"]
        rarity = bucket["rarity"]
        target = int(bucket["target_count"])
        need_pair = bool(bucket.get("contrastive_pair_required"))

        candidates = _filter_recipes_for_bucket(
            recipes_all, dim, bm, bsub, difficulty, rarity, asset_index, construct_cfg
        )
        # exclude already used
        candidates = [r for r in candidates if r["recipe_id"] not in used_recipe_ids]
        rng.shuffle(candidates)

        picked: List[SampledRecipe] = []
        if need_pair:
            triples = _pair_recipes(candidates)
            for orig, baseline, base_role in triples:
                if len(picked) >= target:
                    break
                if baseline is not None and base_role is not None:
                    pair_id = stable_pair_id(orig["recipe_id"], baseline["recipe_id"])
                    picked.append(_wrap(orig, bucket, pair_id, "original"))
                    used_recipe_ids.add(orig["recipe_id"])
                    if len(picked) < target:
                        picked.append(_wrap(baseline, bucket, pair_id, base_role))
                        used_recipe_ids.add(baseline["recipe_id"])
                else:
                    picked.append(_wrap(orig, bucket, None, "original"))
                    used_recipe_ids.add(orig["recipe_id"])
        else:
            for r in candidates:
                if len(picked) >= target:
                    break
                picked.append(_wrap(r, bucket, None, "original"))
                used_recipe_ids.add(r["recipe_id"])

        if len(picked) < target:
            reason_buckets = _diagnose_unfilled(
                recipes_all, dim, bm, bsub, difficulty, rarity, asset_index, construct_cfg
            )
            unfilled.append(
                {
                    "bucket_id": bucket["bucket_id"],
                    "dimension": dim,
                    "input_mode_or_subtype": bucket["input_mode_or_subtype"],
                    "difficulty": difficulty,
                    "rarity": rarity,
                    "target_count": target,
                    "actual_count": len(picked),
                    "shortfall": target - len(picked),
                    "diagnosis": reason_buckets,
                }
            )
        sampled.extend(picked)

    logger.info(
        f"Sampled {len(sampled)} recipes; unfilled buckets={len(unfilled)}; pairs respected"
    )
    return sampled, {
        "unfilled_buckets": unfilled,
        "total_target": sum(int(b["target_count"]) for b in quota_plan["buckets"]),
        "total_sampled": len(sampled),
    }


def _wrap(
    recipe: Dict[str, Any],
    bucket: Dict[str, Any],
    pair_id: Optional[str],
    role: str,
) -> SampledRecipe:
    bm = _bucket_input_mode(bucket["input_mode_or_subtype"])
    return SampledRecipe(
        bucket_id=bucket["bucket_id"],
        dimension=bucket["dimension"],
        input_mode=bm,  # type: ignore[arg-type]
        subtype=_bucket_subtype(bucket["input_mode_or_subtype"]) or _recipe_subtype(recipe),
        difficulty=bucket["difficulty"],
        semantic_rarity=bucket["rarity"],
        contrastive_pair_id=pair_id,
        contrastive_role=role,  # type: ignore[arg-type]
        recipe=recipe,
    )


def _diagnose_unfilled(
    recipes: List[Dict[str, Any]],
    dimension: str,
    bucket_input_mode: str,
    bucket_subtype: str,
    difficulty: str,
    rarity: str,
    asset_index: Dict[str, Dict[str, Any]],
    construct_cfg: Dict[str, Any],
) -> Dict[str, int]:
    """Lightweight reason categorization."""
    reasons = {
        "no_candidate": 0,
        "no_reference_asset": 0,
        "quality_below_threshold": 0,
        "blocking_flag": 0,
    }
    min_q = float(construct_cfg.get("multi_reference_min_quality_score", 0.4))
    require_clean = bool(construct_cfg.get("reference_required_clean_background", True))
    for r in recipes:
        if r.get("target_dimension") != dimension:
            continue
        if _recipe_input_mode(r) != bucket_input_mode:
            continue
        if bucket_subtype and _recipe_subtype(r) and _recipe_subtype(r) != bucket_subtype:
            continue
        if _recipe_difficulty(r) != difficulty:
            continue
        if _recipe_blocking(r):
            reasons["blocking_flag"] += 1
            continue
        if bucket_input_mode == "multi_image":
            ref_ids = list(r.get("reference_asset_ids") or [])
            if len(ref_ids) < 2 or any(rid not in asset_index for rid in ref_ids):
                reasons["no_reference_asset"] += 1
                continue
            if not _multi_image_assets_ok(r, asset_index, min_q, require_clean):
                reasons["quality_below_threshold"] += 1
                continue
        reasons["no_candidate"] += 1  # passed all gates → would have been picked
    return reasons


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 sample_recipes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = benchmark_paths(cfg["output_dir"])
    sampled, report = sample_recipes(cfg, seed=args.seed)
    write_jsonl(paths["sampled_recipes"], [s.model_dump() for s in sampled])
    write_json(paths["quota_unfilled_report"], report)
    logger.info(
        f"Wrote {len(sampled)} sampled recipes -> {paths['sampled_recipes']}; "
        f"unfilled report -> {paths['quota_unfilled_report']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
