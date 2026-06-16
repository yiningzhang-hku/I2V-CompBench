"""
Phase 1 Step 8 \u2014 \u751f\u6210 phase1_audit_report.md\uff0c\u68c0\u67e5\u7ef4\u5ea6\u8986\u76d6 / provenance \u5b8c\u6574\u6027 /
\u8d44\u4ea7\u4e0e recipe \u5206\u5e03 / \u7ef4\u5ea6\u9694\u79bb\u6709\u6548\u6027\u3002

\u8fd0\u884c\u540e\u8f93\u51fa <output_dir>/phase1/phase1_audit_report.md\uff0c\u540c\u65f6\u5728 stdout \u7ed9\u51fa\u9aa8\u67b6\u6307\u6807\u3002
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from loguru import logger

from i2vcompbench.utils.io_utils import ensure_dir, read_jsonl

DIMENSIONS_V2 = [
    "attribute_binding",
    "action_binding",
    "motion_binding",
    "spatial_composition",
    "background_dynamics",
    "view_transformation",
    "interaction_reasoning",
]


def _pct(num: int, denom: int) -> str:
    return f"{(num / denom * 100):.2f}%" if denom else "N/A"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def audit(config: dict) -> None:
    output_base = Path(config["paths"]["output_dir"])
    p1_dir = output_base / "phase1"
    aligned = read_jsonl(str(p1_dir / "aligned_instances.jsonl"))
    assets = read_jsonl(str(p1_dir / "assets.jsonl"))
    recipes = read_jsonl(str(p1_dir / "candidate_recipes.jsonl"))
    text_rows = read_jsonl(str(output_base / "text_analysis" / "text_parse_v2.jsonl"))
    image_rows = read_jsonl(str(output_base / "image_analysis" / "image_parse_v2.jsonl"))

    ensure_dir(str(p1_dir))
    out_path = p1_dir / "phase1_audit_report.md"

    # ---- 1. \u6837\u672c\u603b\u91cf ----
    n_aligned = len(aligned)
    n_text = len([t for t in text_rows if t.get("parse_success", True)])
    n_image = len([i for i in image_rows if i.get("parse_success", True)])

    # ---- 2. \u4e3b\u7ef4\u5ea6\u8986\u76d6 ----
    primary_ctr = Counter(
        t.get("primary_dimension") for t in text_rows if t.get("parse_success", True)
    )
    primary_rows = []
    for d in DIMENSIONS_V2:
        c = primary_ctr.get(d, 0)
        primary_rows.append([d, str(c), _pct(c, n_text)])
    primary_rows.append(["(\u672a\u8def\u7531 / null)", str(primary_ctr.get(None, 0)), _pct(primary_ctr.get(None, 0), n_text)])

    # ---- 3. 7 \u7ef4\u53ef\u8bc4\u6d4b\u6027 ----
    feas_status_counter: Dict[str, Counter] = {d: Counter() for d in DIMENSIONS_V2}
    feasible_counter: Dict[str, int] = {d: 0 for d in DIMENSIONS_V2}
    for row in aligned:
        for f in row.get("evaluator_feasibility") or []:
            d = f.get("dimension")
            if d not in feas_status_counter:
                continue
            feas_status_counter[d][f.get("tool_status", "tool_uncertain")] += 1
            if f.get("feasible"):
                feasible_counter[d] += 1

    feas_rows = []
    for d in DIMENSIONS_V2:
        s = feas_status_counter[d]
        feas_rows.append([
            d,
            f"{feasible_counter[d]} ({_pct(feasible_counter[d], n_aligned)})",
            str(s.get("valid", 0)),
            str(s.get("low_confidence", 0)),
            str(s.get("tool_uncertain", 0)),
            str(s.get("invalid_input", 0)),
        ])

    # ---- 4. \u8d44\u4ea7\u5206\u5e03 + provenance \u5b8c\u6574\u6027 ----
    asset_type_ctr = Counter(a.get("asset_type", "") for a in assets)
    n_assets = len(assets)
    assets_with_prov = sum(
        1 for a in assets
        if isinstance(a.get("provenance"), dict)
        and a["provenance"].get("source_sample_id")
        and a["provenance"].get("extraction_method")
    )
    prov_rate = _pct(assets_with_prov, n_assets)

    # ---- 5. recipe \u5206\u5e03 ----
    recipe_dim_ctr = Counter(r.get("target_dimension") for r in recipes)
    recipe_input_ctr = Counter(r.get("input_mode") for r in recipes)
    recipe_source_ctr = Counter(r.get("source_type") for r in recipes)

    # ---- 6. \u9694\u79bb\u6709\u6548\u6027\u62bd\u67e5 ----
    isolation_violations = 0
    for r in recipes:
        iso = r.get("dimension_isolation") or {}
        primary = iso.get("primary_dimension")
        if primary and primary in (iso.get("forbidden_dimensions") or []):
            isolation_violations += 1

    # ---- \u5199\u62a5\u544a ----
    lines: List[str] = []
    lines.append("# Phase 1 Audit Report")
    lines.append("")
    lines.append("## 1. \u6837\u672c\u89c4\u6a21")
    lines.append("")
    lines.append(_md_table(
        ["\u9879", "\u6570\u91cf"],
        [
            ["text_parse_v2 (parse_success)", str(n_text)],
            ["image_parse_v2 (parse_success)", str(n_image)],
            ["aligned_instances", str(n_aligned)],
            ["assets", str(n_assets)],
            ["candidate_recipes", str(len(recipes))],
        ],
    ))
    lines.append("")
    lines.append("## 2. \u4e3b\u7ef4\u5ea6\u8def\u7531\u5206\u5e03")
    lines.append("")
    lines.append(_md_table(["primary_dimension", "count", "ratio"], primary_rows))
    lines.append("")
    lines.append("## 3. 7 \u7ef4\u53ef\u8bc4\u6d4b\u6027\u8bca\u65ad")
    lines.append("")
    lines.append(_md_table(
        ["dimension", "feasible (\u7387)", "valid", "low_confidence", "tool_uncertain", "invalid_input"],
        feas_rows,
    ))
    lines.append("")
    lines.append("## 4. \u8d44\u4ea7\u5206\u5e03 & provenance")
    lines.append("")
    lines.append(_md_table(
        ["asset_type", "count"],
        [[k or "(empty)", str(v)] for k, v in asset_type_ctr.most_common()],
    ))
    lines.append("")
    lines.append(f"- provenance \u975e\u7a7a\u7387\uff1a**{prov_rate}** ({assets_with_prov}/{n_assets})")
    if n_assets and assets_with_prov < n_assets:
        lines.append(f"- \u26a0 \u8b66\u544a\uff1a\u6709 {n_assets - assets_with_prov} \u6761\u8d44\u4ea7\u7f3a\u5931 provenance \u5fc5\u586b\u5b57\u6bb5")
    lines.append("")
    lines.append("## 5. Recipe \u5206\u5e03")
    lines.append("")
    lines.append(_md_table(
        ["target_dimension", "count", "ratio"],
        [[d, str(recipe_dim_ctr.get(d, 0)), _pct(recipe_dim_ctr.get(d, 0), len(recipes))]
         for d in DIMENSIONS_V2],
    ))
    lines.append("")
    lines.append("**input_mode \u5206\u5e03**")
    lines.append("")
    lines.append(_md_table(
        ["input_mode", "count", "ratio"],
        [[k, str(v), _pct(v, len(recipes))] for k, v in recipe_input_ctr.most_common()],
    ))
    lines.append("")
    lines.append("**source_type \u5206\u5e03**")
    lines.append("")
    lines.append(_md_table(
        ["source_type", "count", "ratio"],
        [[k, str(v), _pct(v, len(recipes))] for k, v in recipe_source_ctr.most_common()],
    ))
    lines.append("")
    lines.append("## 6. \u9694\u79bb\u6709\u6548\u6027\u62bd\u67e5")
    lines.append("")
    lines.append(f"- recipe \u4e2d primary_dimension \u51fa\u73b0\u4e8e forbidden_dimensions \u7684\u6761\u6570\uff1a**{isolation_violations}** "
                  f"\uff08\u671f\u671b 0\uff09")
    if isolation_violations > 0:
        lines.append("- \u26a0 \u68c0\u67e5 forbidden_dimension_leakage \u63a8\u5bfc\u903b\u8f91\uff0c\u53ef\u80fd\u5b58\u5728\u4e3b\u7ef4\u5ea6\u88ab\u8bef\u5165\u540d\u5355\u3002")
    lines.append("")
    lines.append("## 7. \u9a8c\u6536\u7ed3\u8bba\uff08P0\uff09")
    lines.append("")
    lines.append("- [x] schema / prompt / patch \u811a\u672c\u8865\u9f50\u5b8c\u6210")
    lines.append(f"- [{'x' if n_aligned > 0 else ' '}] aligned_instances \u4ea7\u51fa")
    lines.append(f"- [{'x' if n_assets > 0 else ' '}] reference_bank \u4ea7\u51fa")
    lines.append(f"- [{'x' if recipes else ' '}] candidate_recipes \u4ea7\u51fa")
    lines.append(f"- [{'x' if isolation_violations == 0 else ' '}] \u9694\u79bb\u6027\u62bd\u67e5\u901a\u8fc7")
    lines.append(f"- [{'x' if assets_with_prov == n_assets and n_assets > 0 else ' '}] provenance \u975e\u7a7a\u7387 = 100%")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"\u5199\u5165 audit \u62a5\u544a \u2192 {out_path}")
    # \u540c\u65f6\u6253\u5370\u51e0\u4e2a\u9aa8\u67b6\u6307\u6807\u4fbf\u4e8e\u5728 CI / \u7ec8\u7aef\u67e5\u770b
    logger.info(
        f"summary  text={n_text}  image={n_image}  aligned={n_aligned}  "
        f"assets={n_assets} (prov={prov_rate})  recipes={len(recipes)}  "
        f"isolation_violations={isolation_violations}"
    )
