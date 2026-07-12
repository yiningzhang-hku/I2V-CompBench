"""
Phase 2 · Step 7: export BenchmarkSample + phase3_manifest + contrastive_pairs.

Joins:
  question_plans.jsonl
  input_assets_manifest.jsonl
  qc_reports/*.json     (only qc_status=pass enter the dataset)
  final_prompts.jsonl

Produces (Phase 2 §7.2 / Phase 3 §2.4 顶层扁平 17 字段)：
  data/benchmark_dataset/samples/<dimension>.jsonl   (7 files；含 _audit)
  data/benchmark_dataset/phase3_manifest.jsonl       (剔除 _audit)
  data/benchmark_dataset/contrastive_pairs.jsonl     (按 pair_id 聚合)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ..schemas.phase2 import (
    DIMENSIONS_V2,
    BenchmarkAudit,
    BenchmarkSample,
    MultiReferenceQuality,
    PreserveItem,
    QCSummary,
    SourceTrace,
    SubjectRef,
    TargetRelation,
    map_legacy_source_type,
)
from ..utils.io import (
    benchmark_paths,
    iter_jsonl,
    load_config,
    read_json,
    write_jsonl,
)


# ============================================================
# 索引与加载
# ============================================================

def _index_by_question_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["question_id"]: r for r in rows if "question_id" in r}


def _load_qc(qc_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not qc_dir.exists():
        return out
    for p in qc_dir.glob("*.json"):
        try:
            data = read_json(p)
            if "question_id" in data:
                out[data["question_id"]] = data
        except Exception as e:
            logger.warning(f"Skip bad qc report {p}: {e}")
    return out


# ============================================================
# 顶层扁平 17 字段拼装
# ============================================================

def _resolve_first_frame_path(manifest: Dict[str, Any]) -> str:
    for a in manifest.get("assets") or []:
        if a.get("role") == "first_frame":
            return str(a.get("path") or "")
    return ""


def _resolve_input_image_paths(
    manifest: Dict[str, Any],
    input_mode: str,
) -> List[str]:
    """multi_image 时收集所有 ref_images（按 ref_image_idx 升序）；single_image 时为空。"""
    if input_mode != "multi_image":
        return []
    refs: List[Dict[str, Any]] = []
    for a in manifest.get("assets") or []:
        if a.get("role") == "first_frame":
            continue
        refs.append(a)
    refs.sort(
        key=lambda a: (
            a.get("ref_image_idx") if a.get("ref_image_idx") is not None else 1_000_000
        )
    )
    return [str(a.get("path") or "") for a in refs]


def _resolve_target_subjects(plan: Dict[str, Any]) -> List[SubjectRef]:
    tp = plan.get("target_plan") or {}
    out: List[SubjectRef] = []
    for s in tp.get("target_subjects") or []:
        out.append(
            SubjectRef(
                id=str(s.get("id") or f"s{len(out) + 1}"),
                description=str(s.get("description") or ""),
                noun=s.get("noun"),
                ref_image_idx=s.get("ref_image_idx"),
            )
        )
    if not out:
        out = [SubjectRef(id="s1", description="the subject")]
    return out


def _resolve_target_relation(plan: Dict[str, Any]) -> Optional[TargetRelation]:
    tp = plan.get("target_plan") or {}
    rel = tp.get("target_relation")
    if not rel:
        return None
    return TargetRelation(
        type=str(rel.get("type") or ""),
        value=str(rel.get("value") or rel.get("relation") or ""),
        subj=rel.get("subj"),
        obj=rel.get("obj"),
    )


def _resolve_preservation_set(plan: Dict[str, Any]) -> List[PreserveItem]:
    out: List[PreserveItem] = []
    for p in plan.get("preserve_plan") or []:
        out.append(
            PreserveItem(
                scope=str(p.get("scope") or "background"),
                constraint=str(p.get("constraint") or "preserve"),
            )
        )
    return out


# ============================================================
# source_type 推断（§C `_LEGACY_SOURCE_MAP`）
# ============================================================

def _infer_legacy_source_type(
    manifest: Dict[str, Any],
    input_mode: str,
) -> str:
    """从 first_frame asset 的资产层 source_type 反推 Phase 1 旧词汇。

    映射规则（§C）：
      - external → external_real
      - t2i_generated → derived_single_image
      - tip_derived_reference + multi_image → derived_multi_reference
      - tip_derived_reference + single_image 且来自原始样本 → observed_single_image
      - tip_derived_reference + single_image 且为裁剪/合成 → derived_single_image
    """
    first_asset_st: Optional[str] = None
    first_asset_ref_id: Optional[str] = None
    has_external = False
    for a in manifest.get("assets") or []:
        if a.get("role") == "first_frame" and first_asset_st is None:
            first_asset_st = str(a.get("source_type") or "")
            first_asset_ref_id = a.get("source_ref_id")
        if str(a.get("source_type") or "") == "external":
            has_external = True

    if has_external:
        return "external_real"
    if first_asset_st == "t2i_generated":
        return "derived_single_image"
    if first_asset_st == "tip_derived_reference":
        if input_mode == "multi_image":
            return "derived_multi_reference"
        # single_image: source_ref_id 存在且非 asset_ 前缀，说明直接引用原图
        if first_asset_ref_id and not str(first_asset_ref_id).startswith("asset_"):
            return "observed_single_image"
        return "derived_single_image"
    return "derived_single_image"


def _resolve_source_type(
    plan: Dict[str, Any],
    manifest: Dict[str, Any],
    input_mode: str,
) -> tuple[str, str]:
    """返回 (phase3_source_type, legacy_source_type)。"""
    legacy = str(plan.get("source_type") or "").strip()
    if not legacy:
        legacy = _infer_legacy_source_type(manifest, input_mode)
    return map_legacy_source_type(legacy), legacy


# ============================================================
# _audit 子节点
# ============================================================

def _aggregate_multi_quality(manifest: Dict[str, Any]) -> MultiReferenceQuality:
    levels = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
    inv = {v: k for k, v in levels.items()}

    def _min(vals: List[str], default: str = "unknown") -> str:
        if not vals:
            return default
        score = min(levels.get(v, 0) for v in vals)
        return inv.get(score, default)

    def _max(vals: List[str], default: str = "unknown") -> str:
        if not vals:
            return default
        score = max(levels.get(v, 0) for v in vals)
        return inv.get(score, default)

    crops: List[str] = []
    scenes: List[str] = []
    ids_vis: List[str] = []
    for a in manifest.get("assets") or []:
        if a.get("role") == "first_frame":
            continue
        q = a.get("quality") or {}
        crops.append(str(q.get("crop_leakage_risk") or "unknown"))
        if "scene" in str(a.get("role") or ""):
            scenes.append(str(q.get("crop_leakage_risk") or "unknown"))
        ids_vis.append(str(q.get("identity_visibility") or "unknown"))

    return MultiReferenceQuality(
        crop_leakage_risk=_max(crops),  # worst-case
        scene_leakage_risk=_max(scenes) if scenes else "unknown",
        identity_visibility=_min(ids_vis),  # worst-case identity
        scale_compatibility=0.8,  # placeholder
    )


def _build_source_trace(
    plan: Dict[str, Any],
    manifest: Dict[str, Any],
    legacy_source_type: str,
) -> SourceTrace:
    sample_ids: List[str] = []
    asset_ids: List[str] = []
    for a in manifest.get("assets") or []:
        rid = a.get("source_ref_id")
        if not rid:
            continue
        if a.get("source_type") == "tip_derived_reference":
            sample_ids.append(str(rid))
            if str(rid).startswith("asset_") or "_s" in str(rid):
                asset_ids.append(str(rid))
    return SourceTrace(
        recipe_id=str(plan.get("recipe_id") or ""),
        legacy_source_type=legacy_source_type,
        phase1_sample_ids=list(dict.fromkeys(sample_ids)),
        phase1_asset_ids=list(dict.fromkeys(asset_ids)),
    )


# ============================================================
# 写出辅助
# ============================================================

def _strip_audit(row: Dict[str, Any]) -> Dict[str, Any]:
    """phase3_manifest 必须剔除 `_audit` 字段。"""
    out = {k: v for k, v in row.items() if k != "_audit" and k != "audit"}
    return out


def _build_contrastive_pairs(
    samples: List[BenchmarkSample],
) -> List[Dict[str, Any]]:
    """按 contrastive_pair_id 聚合：每对 ≥1 original + ≥1 baseline。"""
    by_pair: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        pid = s.contrastive_pair_id
        if not pid:
            continue
        rec = by_pair.setdefault(
            pid,
            {
                "pair_id": pid,
                "dimension": s.dimension,
                "original_qids": [],
                "baseline_qids": [],
            },
        )
        if s.contrastive_role == "original":
            rec["original_qids"].append(s.question_id)
        else:
            rec["baseline_qids"].append(
                {"qid": s.question_id, "role": s.contrastive_role}
            )

    rows: List[Dict[str, Any]] = []
    for pid, rec in by_pair.items():
        if not rec["original_qids"] or not rec["baseline_qids"]:
            logger.warning(
                f"contrastive_pair {pid} incomplete: "
                f"originals={len(rec['original_qids'])}, baselines={len(rec['baseline_qids'])}"
            )
        rows.append(rec)
    rows.sort(key=lambda r: r["pair_id"])
    return rows


# ============================================================
# Driver
# ============================================================

def export_dataset(config: Dict[str, Any]) -> List[BenchmarkSample]:
    paths = benchmark_paths(config["output_dir"])

    plans_path = paths["question_plans"]
    if not plans_path.exists():
        raise FileNotFoundError(f"question_plans.jsonl not found at {plans_path}")
    manifests_path = paths["input_assets_manifest"]
    finals_path = paths["final_prompts"]

    plans = list(iter_jsonl(plans_path))
    manifests = _index_by_question_id(
        list(iter_jsonl(manifests_path)) if manifests_path.exists() else []
    )
    finals = _index_by_question_id(
        list(iter_jsonl(finals_path)) if finals_path.exists() else []
    )
    qc = _load_qc(paths["qc_reports"])

    samples_by_dim: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_samples: List[BenchmarkSample] = []

    for plan in plans:
        qid = plan["question_id"]
        if qc.get(qid, {}).get("qc_status") != "pass":
            continue
        if qid not in manifests or qid not in finals:
            logger.info(f"[{qid}] missing manifest or final prompt; skipping")
            continue
        manifest = manifests[qid]
        final = finals[qid]
        # 有 failed_check 的条目仅用于审计，禁止进入正式 manifest。
        if final.get("failed_check"):
            logger.info(
                f"[{qid}] final prompt failed {final.get('failed_check')}; skipping"
            )
            continue
        prompt_text = final.get("prompt") or final.get("i2v_prompt")  # 兼容旧产物
        if not prompt_text:
            logger.info(f"[{qid}] empty prompt; skipping")
            continue

        input_mode = plan["input_mode"]
        first_frame_path = _resolve_first_frame_path(manifest)
        input_image_paths = _resolve_input_image_paths(manifest, input_mode)
        target_subjects = _resolve_target_subjects(plan)
        target_relation = _resolve_target_relation(plan)
        # Phase 3 不允许从 prompt 反推目标。泛化占位或空 noun 均视为
        # 结构化阻断项，必须先经过 target repair 再导出。
        if any(
            not (s.noun or "").strip()
            or (s.description or "").strip().lower() in {"", "the subject", "subject"}
            for s in target_subjects
        ):
            logger.info(f"[{qid}] unresolved target_subjects; skipping")
            continue
        if target_relation is None or not target_relation.value.strip():
            logger.info(f"[{qid}] unresolved target change; skipping")
            continue
        preservation_set = _resolve_preservation_set(plan)
        phase3_source_type, legacy_source_type = _resolve_source_type(
            plan, manifest, input_mode
        )

        # _audit 子节点
        audit = BenchmarkAudit(
            source_trace=_build_source_trace(plan, manifest, legacy_source_type),
            qc=QCSummary(
                status=qc[qid]["qc_status"],
                risk_flags=list(qc[qid].get("risk_flags") or []),
            ),
            multi_reference_quality=(
                _aggregate_multi_quality(manifest)
                if input_mode == "multi_image"
                else None
            ),
        )

        sample = BenchmarkSample(
            question_id=qid,
            dimension=plan["dimension"],
            input_mode=input_mode,
            first_frame_path=first_frame_path,
            input_image_paths=input_image_paths,
            prompt=str(prompt_text),
            target_subjects=target_subjects,
            target_relation=target_relation,
            preservation_set=preservation_set,
            contrastive_pair_id=plan.get("contrastive_pair_id"),
            contrastive_role=plan.get("contrastive_role", "original"),
            evaluator_tools=list(plan.get("evaluator_tools") or []),
            expected_failure_modes=list(plan.get("expected_failure_modes") or []),
            subtype=str(plan.get("subtype", "")),
            difficulty=plan.get("difficulty", "medium"),
            semantic_rarity=plan.get("semantic_rarity", "common"),
            source_type=phase3_source_type,  # type: ignore[arg-type]
            **{"_audit": audit},  # type: ignore[arg-type]
        )

        all_samples.append(sample)
        # samples/{dimension}.jsonl 保留 _audit（by_alias=True 输出 "_audit"）
        samples_by_dim[plan["dimension"]].append(
            sample.model_dump(by_alias=True, exclude_none=False)
        )

    # write per-dimension samples (含 _audit)
    for dim in DIMENSIONS_V2:
        rows = samples_by_dim.get(dim, [])
        path = paths["samples"] / f"{dim}.jsonl"
        write_jsonl(path, rows)
        logger.info(f"  samples/{dim}.jsonl: {len(rows)} rows")

    # write phase3_manifest（剔除 _audit）
    manifest_rows = [
        _strip_audit(s.model_dump(by_alias=True, exclude_none=False))
        for s in all_samples
    ]
    write_jsonl(paths["phase3_manifest"], manifest_rows)
    logger.info(f"phase3_manifest.jsonl rows = {len(all_samples)}")

    # write contrastive_pairs.jsonl
    pair_rows = _build_contrastive_pairs(all_samples)
    write_jsonl(paths["contrastive_pairs"], pair_rows)
    logger.info(f"contrastive_pairs.jsonl rows = {len(pair_rows)}")

    return all_samples


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 export_dataset")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    samples = export_dataset(cfg)
    logger.info(f"Exported {len(samples)} BenchmarkSamples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
