"""
Phase 2 · Step 9: package the final dataset into a per-dimension / per-question folder
layout for human browsing or distribution.

Layout produced (under {output_dir}/by_dimension/):
    by_dimension/
      attribute_binding/
        attr_single_0001/
          prompt.json
          first_frame.png        # (or ref_1.png / ref_2.png ... for multi_image)
        attr_single_0002/
          ...
      action_binding/
        ...
      spatial_composition/
        README.md                # explains why dimension is empty (when 0 questions)
      ...

Source of truth: data/benchmark_dataset/phase3_manifest.jsonl

prompt.json schema (compact, human-friendly):
{
  "question_id":       str,
  "dimension":         str,
  "subtype":           str | null,
  "input_mode":        "single_image" | "multi_image",
  "difficulty":        "easy" | "medium" | "hard",
  "rarity":            "common" | "rare",
  "prompt":            str,                # final I2V prompt (post-finalize)
  "input_files":       List[str],          # filenames in this folder, e.g. ["first_frame.png"]
  "evaluator_tools":   List[str],
  "expected_failure_modes": List[str],
  "preservation_set":  List[Dict],
  "target_subjects":   List[Dict],
  "target_relation":   Dict | null,
  "contrastive_pair_id":  str | null,
  "contrastive_role":  str | null
}
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from ..utils.io import benchmark_paths, iter_jsonl, load_config


# 七维度的固定顺序（保证空目录也按维度顺序创建）
ALL_DIMENSIONS: List[str] = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "spatial_composition",
    "background_dynamics",
    "view_transformation",
    "interaction_reasoning",
]


_COMPACT_KEYS = (
    "question_id",
    "dimension",
    "subtype",
    "input_mode",
    "difficulty",
    "semantic_rarity",
    "prompt",
    "evaluator_tools",
    "expected_failure_modes",
    "preservation_set",
    "target_subjects",
    "target_relation",
    "contrastive_pair_id",
    "contrastive_role",
)


def _compact_record(row: Dict[str, Any], input_files: List[str]) -> Dict[str, Any]:
    """从 phase3_manifest 一行抽出精简字段，附带 input_files。"""
    out: Dict[str, Any] = {}
    for k in _COMPACT_KEYS:
        if k in row:
            v = row[k]
            # 对外字段名统一：semantic_rarity -> rarity
            out["rarity" if k == "semantic_rarity" else k] = v
    out["input_files"] = input_files
    return out


def _copy_question_assets(
    qid: str,
    input_mode: str,
    first_frame_path: str | None,
    input_image_paths: List[str],
    src_root: Path,
    dst_dir: Path,
) -> List[str]:
    """复制首帧 / ref 图到题目子目录，返回写入的文件名列表。"""
    written: List[str] = []
    if input_mode == "single_image":
        if not first_frame_path:
            logger.warning(f"[package] {qid} single_image but no first_frame_path; skipping image copy")
            return written
        src = (src_root / first_frame_path) if not Path(first_frame_path).is_absolute() else Path(first_frame_path)
        if not src.exists():
            logger.warning(f"[package] {qid} first frame missing: {src}")
            return written
        dst = dst_dir / "first_frame.png"
        shutil.copyfile(src, dst)
        written.append("first_frame.png")
        return written

    # multi_image: 复制 input_image_paths 为 ref_1.png / ref_2.png / ...
    for idx, p in enumerate(input_image_paths or [], start=1):
        src = (src_root / p) if not Path(p).is_absolute() else Path(p)
        if not src.exists():
            logger.warning(f"[package] {qid} ref image missing: {src}")
            continue
        ext = src.suffix.lower() or ".png"
        name = f"ref_{idx}{ext}"
        shutil.copyfile(src, dst_dir / name)
        written.append(name)
    return written


def _write_empty_dim_readme(dim_dir: Path, dimension: str) -> None:
    """空维度目录写 README 说明原因。"""
    msg = (
        f"# {dimension}\n\n"
        f"This dimension contains **0 questions** in the current pilot batch.\n\n"
        f"Reason: Phase 1 step4 routing did not assign any priors to this dimension "
        f"(common cause: the action_binding dimension absorbed most candidate recipes).\n\n"
        f"To populate this dimension, re-run Phase 1 step4 with adjusted routing weights, "
        f"or expand the input prior pool with more samples that match this dimension's primary signature.\n"
    )
    (dim_dir / "README.md").write_text(msg, encoding="utf-8")


def package_by_dimension(cfg: Dict[str, Any]) -> Dict[str, int]:
    """主入口：按维度+按题目打包到 by_dimension/。"""
    paths = benchmark_paths(cfg["output_dir"])
    src_root = Path(cfg["output_dir"]).resolve().parent  # phase3_manifest 中的路径是相对于 i2v_compbench/ 的
    # 确认上面 src_root 的实际语义：phase3_manifest 中存的是 "data\\benchmark_dataset\\first_frames\\xxx.png"，
    # 这是相对 i2v_compbench/ 的路径。output_dir 就是 "data/benchmark_dataset"，所以拼接 base 应该是它的 parent.parent。
    # 简化：把 first_frame_path 直接当成相对于 i2v_compbench/ (即 cwd at runtime) 的路径
    src_root = Path.cwd()

    manifest_path: Path = paths["phase3_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"phase3_manifest not found: {manifest_path}; run --step export first")

    out_root: Path = paths["root"] / "by_dimension"
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) 预先创建 7 个维度目录
    for dim in ALL_DIMENSIONS:
        (out_root / dim).mkdir(parents=True, exist_ok=True)

    # 2) 遍历 manifest，按 dim+qid 写入
    counts: Dict[str, int] = {dim: 0 for dim in ALL_DIMENSIONS}
    n_total = 0
    for row in iter_jsonl(manifest_path):
        qid = row["question_id"]
        dim = row["dimension"]
        if dim not in counts:
            logger.warning(f"[package] {qid} has unknown dimension {dim!r}; skipped")
            continue
        q_dir = out_root / dim / qid
        q_dir.mkdir(parents=True, exist_ok=True)

        input_files = _copy_question_assets(
            qid=qid,
            input_mode=row.get("input_mode", "single_image"),
            first_frame_path=row.get("first_frame_path"),
            input_image_paths=row.get("input_image_paths") or [],
            src_root=src_root,
            dst_dir=q_dir,
        )

        compact = _compact_record(row, input_files)
        (q_dir / "prompt.json").write_text(
            json.dumps(compact, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        counts[dim] += 1
        n_total += 1

    # 3) 空维度补 README
    for dim, n in counts.items():
        if n == 0:
            _write_empty_dim_readme(out_root / dim, dim)

    # 4) 顶层 README.md（索引）
    lines = [
        "# Benchmark Dataset (by-dimension layout)",
        "",
        f"Total questions: **{n_total}**",
        "",
        "| Dimension | Questions |",
        "| --- | --- |",
    ]
    for dim in ALL_DIMENSIONS:
        lines.append(f"| {dim} | {counts[dim]} |")
    lines += [
        "",
        "## Per-question folder layout",
        "",
        "Each `<dimension>/<question_id>/` contains:",
        "",
        "- `prompt.json` — compact manifest (final I2V prompt + minimal evaluation metadata)",
        "- `first_frame.png` — first frame image (single_image mode)",
        "- `ref_1.png`, `ref_2.png`, ... — reference images (multi_image mode)",
        "",
        "Empty dimensions have a `README.md` explaining the gap.",
    ]
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(f"[package] wrote {n_total} questions across {sum(1 for v in counts.values() if v > 0)}/{len(ALL_DIMENSIONS)} dimensions -> {out_root}")
    return counts
