"""
Phase 1 Step 5 \u2014 Reference Bank \u8d44\u4ea7\u62bd\u53d6\u3002

\u8f93\u51fa 5 \u7c7b\u8d44\u4ea7\uff08\u4e0e\u4e09\u9636\u6bb5\u603b\u89c8 \u00a73.3 Step 5 \u4e00\u81f4\uff09\uff1a
    subject / attribute / object / scene_reference_original / scene_reference_inpainted

P0 \u9636\u6bb5\u8003\u8651\uff1a
* \u4e0d\u505a\u771f\u5b9e\u5206\u5272 / inpainting\uff1bsubject / attribute / object \u5747\u8d70 mock_crop_9grid\u3002
* scene_reference_original = \u539f\u56fe\u590d\u7528\uff08\u9002\u7528\u4e8e view_transformation \u4e0e background_dynamics\u7684\u5165\u53c2\u53c2\u8003\uff09\u3002
* scene_reference_inpainted \u4e00\u5f8b\u8df3\u8fc7\uff0c\u4ec5\u5728 audit \u4e2d\u8bb0\u5f55 0%\u3002
* provenance \u5fc5\u586b\uff08source_sample_id / source_image_path / extraction_method / extraction_params\uff09\u3002
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from i2vcompbench.phase1.mock_geometry import crop_to_path, position_to_bbox
from i2vcompbench.utils.io_utils import ensure_dir, read_jsonl, write_jsonl

# asset \u9002\u7528\u7ef4\u5ea6\u9ed8\u8ba4\u8868
USABLE_FOR_DEFAULT: Dict[str, List[str]] = {
    "subject":                    ["attribute_binding", "action_binding", "motion_binding",
                                    "spatial_composition", "interaction_reasoning"],
    "attribute":                  ["attribute_binding"],
    "object":                     ["spatial_composition", "interaction_reasoning"],
    "scene_reference_original":   ["background_dynamics", "view_transformation"],
    "scene_reference_inpainted":  ["spatial_composition", "interaction_reasoning"],
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _semantic_tags(subject: dict) -> List[str]:
    tags = [subject.get("name", "")]
    attrs = subject.get("attributes") or {}
    for k in ("color", "material_texture", "state", "wearing"):
        for v in attrs.get(k) or []:
            tags.append(str(v))
    return [t for t in tags if t]


def _build_subject_asset(
    sample_id: str,
    src_image_path: str,
    subject: dict,
    asset_type: str,
    out_dir: Path,
) -> Optional[Dict]:
    """\u4e3a\u4e00\u4e2a\u4e3b\u4f53\u5236\u4f5c subject / attribute / object \u8d44\u4ea7\uff08mock crop\uff09\u3002"""
    bbox = subject.get("bbox") or list(position_to_bbox(subject.get("position_in_frame")))
    instance_id = subject.get("instance_id") or subject.get("id") or "subj"

    suffix = {
        "subject": "subj",
        "attribute": "attr",
        "object": "obj",
    }[asset_type]
    asset_id = f"{sample_id}_{instance_id}_{suffix}"
    asset_filename = f"{asset_id}.jpg"
    asset_path = out_dir / asset_type / asset_filename
    pixel_size = crop_to_path(src_image_path, tuple(bbox), str(asset_path))

    if pixel_size is None:
        # \u88c1\u526a\u5931\u8d25\uff08\u56fe\u4e0d\u5728 / \u672a\u5b89\u88c5 PIL\uff09\uff0c\u8df3\u8fc7\u8d44\u4ea7\u521b\u5efa
        logger.debug(f"crop \u5931\u8d25\uff0c\u8df3\u8fc7 {asset_id} \uff08src={src_image_path}\uff09")
        return None

    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    quality_score = max(0.0, min(1.0, 0.5 + min(bw, bh)))  # \u7c97\u4f30\uff1a\u8d8a\u5927\u8d28\u91cf\u8d8a\u9ad8

    return {
        "asset_id": asset_id,
        "asset_type": asset_type,
        "asset_path": str(asset_path),
        "semantic_tags": _semantic_tags(subject),
        "usable_for": list(USABLE_FOR_DEFAULT[asset_type]),
        "quality": {
            "resolution": list(pixel_size),
            "aspect_ratio": round(pixel_size[0] / pixel_size[1], 3) if pixel_size[1] else None,
            "is_clean_background": False,  # mock crop \u4e0d\u53bb\u80cc
            "has_visible_subject": True,
            "quality_score": round(quality_score, 3),
            "quality_flags": ["mock_crop_9grid"],
        },
        "provenance": {
            "source_sample_id": sample_id,
            "source_image_path": src_image_path,
            "extraction_method": "mock_crop_9grid",
            "extraction_params": {"bbox_norm": list(bbox), "pad_ratio": 0.05},
            "created_by": "phase1.reference_bank",
            "created_at": _now_iso(),
        },
    }


def _build_scene_original_asset(sample_id: str, src_image_path: str) -> Optional[Dict]:
    """scene_reference_original \u2014\u2014 \u539f\u56fe\u590d\u7528\uff0c\u4e0d\u88c1\u526a\u3002"""
    if not Path(src_image_path).exists():
        return None
    return {
        "asset_id": f"{sample_id}_scene_orig",
        "asset_type": "scene_reference_original",
        "asset_path": src_image_path,
        "semantic_tags": [],
        "usable_for": list(USABLE_FOR_DEFAULT["scene_reference_original"]),
        "quality": {
            "resolution": None,
            "aspect_ratio": None,
            "is_clean_background": False,
            "has_visible_subject": True,
            "quality_score": 0.8,
            "quality_flags": ["original_image"],
        },
        "provenance": {
            "source_sample_id": sample_id,
            "source_image_path": src_image_path,
            "extraction_method": "original",
            "extraction_params": {},
            "created_by": "phase1.reference_bank",
            "created_at": _now_iso(),
        },
    }


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def build_reference_bank(config: dict) -> None:
    output_base = Path(config["paths"]["output_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    image_path = str(output_base / "image_analysis" / "image_parse_v2.jsonl")
    manifest_path = str(manifest_dir / "manifest_clean.jsonl")

    image_rows = read_jsonl(image_path)
    manifest_rows = read_jsonl(manifest_path)
    if not image_rows or not manifest_rows:
        logger.error("\u7f3a\u5c11 image_parse_v2 \u6216 manifest_clean\uff0c\u65e0\u6cd5\u62bd\u53d6\u8d44\u4ea7\u3002")
        return
    image_map = {r.get("sample_id"): r for r in image_rows if r.get("parse_success", True)}
    manifest_map = {r.get("sample_id"): r for r in manifest_rows}

    asset_dir = output_base / "phase1" / "reference_bank"
    ensure_dir(str(asset_dir))
    out_path = str(output_base / "phase1" / "assets.jsonl")

    assets: List[Dict] = []
    for sid, image_row in image_map.items():
        manifest = manifest_map.get(sid)
        if not manifest:
            continue
        src_image_path = manifest.get("image_path") or ""
        ref_pot = image_row.get("reference_potential") or {}

        subjects = image_row.get("subjects", []) or []

        # subject / attribute / object \u8d44\u4ea7
        for s in subjects:
            if ref_pot.get("subject"):
                a = _build_subject_asset(sid, src_image_path, s, "subject", asset_dir)
                if a:
                    assets.append(a)
            if ref_pot.get("attribute"):
                a = _build_subject_asset(sid, src_image_path, s, "attribute", asset_dir)
                if a:
                    assets.append(a)
            if ref_pot.get("object") and not s.get("is_animate", False):
                a = _build_subject_asset(sid, src_image_path, s, "object", asset_dir)
                if a:
                    assets.append(a)

        # scene_reference_original
        if ref_pot.get("scene_reference_original"):
            a = _build_scene_original_asset(sid, src_image_path)
            if a:
                assets.append(a)
        # scene_reference_inpainted: P0 \u8df3\u8fc7

    write_jsonl(out_path, assets)
    logger.info(f"reference_bank \u8d44\u4ea7\u6570={len(assets)} \u2192 {out_path}")
