"""
将 Phase 2 by_dimension/ 输出转换为 T2V-CompBench 风格目录结构。

Output layout (under <out_dir>):
    prompts/<dim>.txt              # one final_prompt per line, ordered by new id 0001..N
    meta_data/<dim>.json           # full evaluation-ready metadata, list of dicts
    first_frames/<dim>/0001.png    # native first frame
    first_frames/<dim>/0001_16x9.png  # 16:9 normalized first frame

Each dimension's questions are sorted by original question_id (lexical) and
renumbered 0001..N (T2V-CompBench convention).

Usage:
    python scripts/convert_v2_to_t2vlike.py \
        --src data/benchmark_dataset/by_dimension \
        --out data/benchmark_dataset/v2_t2vlike
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


DIMENSIONS_WITH_DATA = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "background_dynamics",
    "view_transformation",
]


def _load_prompt_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def convert(src_root: Path, out_root: Path) -> Dict[str, int]:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "prompts").mkdir(exist_ok=True)
    (out_root / "meta_data").mkdir(exist_ok=True)
    (out_root / "first_frames").mkdir(exist_ok=True)

    counts: Dict[str, int] = {}

    for dim in DIMENSIONS_WITH_DATA:
        dim_dir = src_root / dim
        if not dim_dir.is_dir():
            print(f"[skip] dimension dir missing: {dim_dir}")
            counts[dim] = 0
            continue

        question_dirs = sorted(
            [d for d in dim_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
        )
        if not question_dirs:
            counts[dim] = 0
            continue

        prompts_lines: List[str] = []
        meta_records: List[Dict[str, Any]] = []
        ff_dim_dir = out_root / "first_frames" / dim
        ff_dim_dir.mkdir(parents=True, exist_ok=True)

        for new_idx, qdir in enumerate(question_dirs, start=1):
            new_id = f"{new_idx:04d}"
            pj_path = qdir / "prompt.json"
            if not pj_path.exists():
                print(f"[warn] missing prompt.json: {qdir}")
                continue
            pj = _load_prompt_json(pj_path)

            final_prompt = (
                pj.get("final_prompt")
                or pj.get("prompt")
                or pj.get("refined_prompt")
                or ""
            ).strip()
            prompts_lines.append(final_prompt)

            # ---- meta record (full version) ----
            meta = {
                "id": new_id,
                "original_question_id": pj.get("question_id") or qdir.name,
                "dimension": pj.get("dimension", dim),
                "prompt": final_prompt,
                "input_mode": pj.get("input_mode"),
                "difficulty": pj.get("difficulty"),
                "rarity": pj.get("rarity"),
                "source_sample_id": pj.get("source_sample_id"),
                "qc_status": pj.get("qc_status"),
            }
            # Preserve any dimension-specific evaluation anchors
            for k in (
                "primary_dimension",
                "attribute_target",
                "action_verb",
                "motion_type",
                "scene_dynamic",
                "camera_motion",
                "view_transformation_type",
                "evaluation_anchors",
                "contrastive_pair_id",
                "contrastive_role",
                "objects",
                "attributes",
            ):
                if k in pj:
                    meta[k] = pj[k]
            meta_records.append(meta)

            # ---- copy images ----
            ff_native = qdir / "first_frame.png"
            ff_169 = qdir / "first_frame_16x9.png"
            if ff_native.exists():
                shutil.copy2(ff_native, ff_dim_dir / f"{new_id}.png")
            if ff_169.exists():
                shutil.copy2(ff_169, ff_dim_dir / f"{new_id}_16x9.png")

        # ---- write prompts/<dim>.txt ----
        with (out_root / "prompts" / f"{dim}.txt").open(
            "w", encoding="utf-8", newline="\n"
        ) as f:
            f.write("\n".join(prompts_lines) + "\n")

        # ---- write meta_data/<dim>.json ----
        with (out_root / "meta_data" / f"{dim}.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(meta_records, f, ensure_ascii=False, indent=2)

        counts[dim] = len(meta_records)
        print(f"[ok] {dim}: {len(meta_records)} questions")

    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("data/benchmark_dataset/by_dimension"),
        help="Source by_dimension directory (V2 output).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/benchmark_dataset/v2_t2vlike"),
        help="Output directory in T2V-CompBench-like structure.",
    )
    args = ap.parse_args()

    counts = convert(args.src, args.out)
    total = sum(counts.values())
    print("\n=== Summary ===")
    for d, n in counts.items():
        print(f"  {d}: {n}")
    print(f"  TOTAL: {total}")


if __name__ == "__main__":
    main()
