"""
Phase 2 · Step 4: construct input images for each QuestionPlan.

For every required_image in the plan we walk the `source_preference` chain:

    tip_derived_reference  -> reuse Phase 1 first-frame or reference_bank asset
    t2i_generated          -> call Phase2SiliconFlowClient.call_t2i
    external               -> reserved for production runs (skipped in pilot)

Outputs:
    data/benchmark_dataset/first_frames/<question_id>.png             (single first_frame)
    data/benchmark_dataset/ref_images/<question_id>_ref<k>.png         (multi-image refs)
    data/benchmark_dataset/input_assets_manifest.jsonl

· 路径命名严格对齐 Phase 2 §3.2 / §7.2：使用 first_frames/ 与 ref_images/。
· multi_image 参考图的下标 k 由 question_plan.target_subjects[i].ref_image_idx
  在 §4.3 step 6 预先赋值；construct_inputs 严格对齐。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image

from ..schemas.phase2 import (
    AssetQualityLite,
    InputAssetItem,
    InputAssetManifest,
)
from ..utils.api_client import Phase2SiliconFlowClient
from ..utils.ids import asset_id_for
from ..utils.image import (
    DEFAULT_LONG_EDGE,
    bytes_to_image,
    image_resolution_ok,
    open_image,
    resize_long_edge,
    save_image,
    to_16x9_720p,
)
from ..utils.io import (
    Phase1Bundle,
    benchmark_paths,
    iter_jsonl,
    load_config,
    write_jsonl,
)
from ..utils.templates import TemplateRegistry, render_template


# Role -> Phase 1 asset_type mapping
_ROLE_TO_ASSET_TYPE = {
    "target_subject": ("subject",),
    "reference_subject": ("subject",),
    "subject_reference": ("subject",),
    "attribute_reference": ("attribute",),
    "object_reference": ("object",),
    "scene_reference": ("scene_reference_inpainted", "scene_reference_original"),
    "scene_reference_inpainted": ("scene_reference_inpainted",),
    "scene_reference_original": ("scene_reference_original",),
}


# ============================================================
# Helpers
# ============================================================

def _resolve_ref_idx_from_plan(plan: Dict[str, Any], role: str, idx: int) -> int:
    """从 question_plan.target_plan.target_subjects 反查 ref_image_idx。

    如果 plan 里未明细指定（类似 attribute_reference / scene_reference），退化为
    该 role 在 required_images 里的顺序下标（减去 first_frame 以适配§4.4 step 6）。
    """
    target_subjects = (plan.get("target_plan") or {}).get("target_subjects") or []
    for ts in target_subjects:
        if ts.get("ref_image_idx") is None:
            continue
        if role in (ts.get("id") or "", ts.get("description") or ""):
            return int(ts["ref_image_idx"])
    required = (plan.get("input_plan") or {}).get("required_images") or []
    seen = 0
    for j, spec in enumerate(required):
        if (spec.get("role") or "") == "first_frame":
            continue
        if j == idx:
            return seen
        seen += 1
    return max(0, idx)


def _save_target_path(
    paths: Dict[str, Path],
    question_id: str,
    role: str,
    ref_image_idx: Optional[int] = None,
) -> Path:
    """§3.2 / §7.2：first_frame 走 first_frames/{qid}.png；参考图走 ref_images/{qid}_ref{k}.png。"""
    if role == "first_frame":
        return paths["first_frames"] / f"{question_id}.png"
    k = ref_image_idx if ref_image_idx is not None else 0
    return paths["ref_images"] / f"{question_id}_ref{k}.png"


def _quality_for_tip(image_path: Path) -> AssetQualityLite:
    try:
        img = open_image(image_path)
    except Exception:
        return AssetQualityLite(
            identity_visibility="unknown",
            crop_leakage_risk="unknown",
            resolution_ok=False,
        )
    return AssetQualityLite(
        identity_visibility="high",
        crop_leakage_risk="low",
        resolution_ok=image_resolution_ok(img),
    )


def _quality_for_t2i() -> AssetQualityLite:
    return AssetQualityLite(
        identity_visibility="medium",
        crop_leakage_risk="low",
        resolution_ok=True,
        notes="t2i_generated",
    )


def _resolve_asset_for_role(
    role: str,
    recipe: Dict[str, Any],
    asset_index: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    desired = _ROLE_TO_ASSET_TYPE.get(role, ())
    candidates: List[Dict[str, Any]] = []
    for rid in recipe.get("reference_asset_ids") or []:
        a = asset_index.get(rid)
        if not a:
            continue
        if not desired or a.get("asset_type") in desired:
            candidates.append(a)
    if candidates:
        # prefer highest quality_score
        candidates.sort(
            key=lambda x: float((x.get("quality") or {}).get("quality_score") or 0.0),
            reverse=True,
        )
        return candidates[0]
    # fallback: any asset for this recipe
    for rid in recipe.get("reference_asset_ids") or []:
        a = asset_index.get(rid)
        if a:
            return a
    return None


def _resolve_t2i_prompt(
    template_subtype_block: Dict[str, Any],
    slots: Dict[str, Any],
    role: str,
    role_description: str,
) -> str:
    pattern = str(template_subtype_block.get("t2i_prompt_pattern") or "")
    if not pattern:
        # generic fallback
        target = slots.get("target_subject", "subject")
        if role == "first_frame":
            return f"Photo of {target}, plain neutral background, realistic"
        return f"Isolated photo of {role_description or target}, plain neutral background"
    # substitute role_subject for multi-image patterns
    role_subject = role_description or slots.get(role, slots.get("target_subject", "subject"))
    return render_template(pattern, {**slots, "role_subject": role_subject, "role": role})


def _save_inference_companion(img: Image.Image, primary_dst: Path) -> Path:
    """Produce the strict 1280x720 16:9 companion alongside the native-ratio PNG.

    Path convention (Phase 2 §6.4 dual-track output):
        first_frame.png             -> first_frame_16x9.png
        ref0.png                    -> ref0_16x9.png
        spatial_0042.png            -> spatial_0042_16x9.png
    Evaluator (Phase 3) keeps reading the native PNG; only the I2V generator wrapper
    reads the *_16x9.png companion.
    """
    companion = primary_dst.with_name(primary_dst.stem + "_16x9" + primary_dst.suffix)
    img_169 = to_16x9_720p(img)
    save_image(img_169, companion, fmt="PNG")
    return companion


def _copy_or_save_tip_image(src: Path, dst: Path, long_edge: int) -> None:
    img = open_image(src)
    # enlarge=True: TIP-I2V 原图常为 224x126 等低分辨率，必须强制放大到 720P 长边（1280）
    img = resize_long_edge(img, long_edge=long_edge, enlarge=True)
    save_image(img, dst, fmt="PNG")
    # 双轨产物：额外产一份 1280x720 严格 16:9 推理伴生文件
    _save_inference_companion(img, dst)


def _save_t2i_bytes(data: bytes, dst: Path, long_edge: int) -> None:
    img = bytes_to_image(data)
    img = resize_long_edge(img, long_edge=long_edge, enlarge=True)
    save_image(img, dst, fmt="PNG")
    _save_inference_companion(img, dst)


# ============================================================
# Per-question construction
# ============================================================

def _construct_for_question(
    plan: Dict[str, Any],
    bundle: Phase1Bundle,
    template_subtype_block: Dict[str, Any],
    client: Optional[Phase2SiliconFlowClient],
    paths: Dict[str, Path],
    long_edge: int,
    enable_t2i: bool,
    slots: Dict[str, Any],
    sampled_recipe_lookup: Dict[str, Dict[str, Any]],
) -> Optional[InputAssetManifest]:
    question_id: str = plan["question_id"]
    recipe_id: str = plan["recipe_id"]
    recipe = sampled_recipe_lookup.get(recipe_id, {})
    if not recipe:
        logger.warning(f"[{question_id}] recipe_id {recipe_id} not found in sampled_recipes")
        return None

    required: List[Dict[str, Any]] = (plan.get("input_plan") or {}).get("required_images") or []
    if not required:
        logger.warning(f"[{question_id}] no required_images in plan; skip")
        return None

    assets: List[InputAssetItem] = []
    for idx, spec in enumerate(required):
        role = str(spec.get("role") or f"image_{idx}")
        description = str(spec.get("description") or "")
        sources: List[str] = list(spec.get("source_preference") or ["tip_derived_reference", "t2i_generated"])

        # §4.4 step 6：ref_images 下标与 target_subjects[i].ref_image_idx 严格对齐
        ref_idx: Optional[int] = (
            None if role == "first_frame" else _resolve_ref_idx_from_plan(plan, role, idx)
        )
        target_path = _save_target_path(paths, question_id, role, ref_idx)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        chosen: Optional[Dict[str, Any]] = None  # the actual InputAssetItem dict

        for src_kind in sources:
            if src_kind == "tip_derived_reference":
                if role == "first_frame" and plan["input_mode"] == "single_image":
                    sid = recipe.get("source_sample_id")
                    p = bundle.get_image_path(sid) if sid else None
                    if p and Path(p).exists():
                        try:
                            _copy_or_save_tip_image(Path(p), target_path, long_edge)
                            chosen = {
                                "source_type": "tip_derived_reference",
                                "source_ref_id": sid,
                                "quality": _quality_for_tip(target_path),
                            }
                            break
                        except Exception as e:
                            logger.warning(f"[{question_id}] tip first_frame copy failed: {e}")
                else:
                    asset = _resolve_asset_for_role(role, recipe, bundle.assets)
                    if asset:
                        ap = Path(asset.get("asset_path") or "")
                        if not ap.is_absolute():
                            ap = bundle.dir / ap
                        if ap.exists():
                            try:
                                _copy_or_save_tip_image(ap, target_path, long_edge)
                                chosen = {
                                    "source_type": "tip_derived_reference",
                                    "source_ref_id": asset.get("asset_id"),
                                    "quality": _quality_for_tip(target_path),
                                }
                                break
                            except Exception as e:
                                logger.warning(f"[{question_id}] reference_bank copy failed for {role}: {e}")
            elif src_kind == "t2i_generated":
                if not enable_t2i or client is None:
                    continue
                t2i_prompt = _resolve_t2i_prompt(template_subtype_block, slots, role, description)
                try:
                    imgs = client.call_t2i(t2i_prompt, n=1)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[{question_id}] T2I error: {e}")
                    imgs = []
                if imgs:
                    try:
                        _save_t2i_bytes(imgs[0], target_path, long_edge)
                        chosen = {
                            "source_type": "t2i_generated",
                            "source_ref_id": None,
                            "quality": _quality_for_t2i(),
                        }
                        break
                    except Exception as e:
                        logger.warning(f"[{question_id}] T2I save failed: {e}")
            elif src_kind == "external":
                # not enabled in pilot
                continue

        if chosen is None:
            logger.warning(
                f"[{question_id}] could not satisfy required_image role={role} from {sources}; skipping question"
            )
            return None

        assets.append(
            InputAssetItem(
                asset_id=asset_id_for(question_id, role, idx),
                role=role,
                path=str(target_path),
                source_type=chosen["source_type"],  # type: ignore[arg-type]
                source_ref_id=chosen.get("source_ref_id"),
                ref_image_idx=ref_idx,
                quality=chosen["quality"],
            )
        )

    return InputAssetManifest(question_id=question_id, assets=assets)


# ============================================================
# Driver
# ============================================================

def construct_inputs(config: Dict[str, Any]) -> List[InputAssetManifest]:
    paths = benchmark_paths(config["output_dir"])
    plans_path = paths["question_plans"]
    if not plans_path.exists():
        raise FileNotFoundError(
            f"question_plans.jsonl not found at {plans_path}; run build_question_plan first"
        )

    sampled_path = paths["sampled_recipes"]
    sampled_recipe_lookup: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(sampled_path):
        rid = (row.get("recipe") or {}).get("recipe_id")
        if rid:
            sampled_recipe_lookup[rid] = row.get("recipe") or {}

    bundle = Phase1Bundle(config["phase1_bundle_dir"])
    construct_cfg = config.get("construct", {})
    long_edge = int(construct_cfg.get("long_edge", DEFAULT_LONG_EDGE))
    enable_t2i = bool(construct_cfg.get("enable_t2i", True))
    client: Optional[Phase2SiliconFlowClient] = None
    if enable_t2i:
        try:
            client = Phase2SiliconFlowClient(config)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"T2I client init failed; T2I disabled: {e}")
            client = None

    registry = TemplateRegistry()
    out: List[InputAssetManifest] = []

    # build a slot snapshot from the plan (target/reference subjects)
    for plan in iter_jsonl(plans_path):
        dim = plan["dimension"]
        template = registry.get(dim)
        subtype_block = template.find_subtype(plan.get("subtype"), plan.get("input_mode"))
        target_subjects = (plan.get("target_plan") or {}).get("target_subjects") or []
        primary = target_subjects[0] if target_subjects else {}
        secondary = target_subjects[1] if len(target_subjects) >= 2 else {}
        slots = {
            "target_subject": primary.get("description") or "the subject",
            "reference_subject": secondary.get("description") or _extract_reference_subject(plan),
            "direction": "to the right",
            "target_relation": (((plan.get("target_plan") or {}).get("target_relation") or {}).get("value") or ((plan.get("target_plan") or {}).get("target_relation") or {}).get("relation") or "right_of"),
            "background": "neutral",
        }
        manifest = _construct_for_question(
            plan=plan,
            bundle=bundle,
            template_subtype_block=subtype_block,
            client=client,
            paths=paths,
            long_edge=long_edge,
            enable_t2i=enable_t2i,
            slots=slots,
            sampled_recipe_lookup=sampled_recipe_lookup,
        )
        if manifest is not None:
            out.append(manifest)

    logger.info(f"Constructed inputs for {len(out)} questions")
    return out


def _extract_reference_subject(plan: Dict[str, Any]) -> str:
    for it in (plan.get("input_plan") or {}).get("required_images") or []:
        if it.get("role") == "reference_subject":
            return it.get("description") or "the reference subject"
    return "the reference subject"


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 construct_inputs")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = benchmark_paths(cfg["output_dir"])
    manifests = construct_inputs(cfg)
    write_jsonl(paths["input_assets_manifest"], [m.model_dump() for m in manifests])
    logger.info(f"Wrote {len(manifests)} manifests -> {paths['input_assets_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
