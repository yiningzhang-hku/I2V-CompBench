"""
Step 5: Prior Package Assembly + Report Generation.

Reads all intermediate outputs from step 4, assembles the complete
PriorPackage JSON (the core deliverable), and generates a human-readable
summary report.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from loguru import logger

from i2vcompbench.utils.io_utils import (
    ensure_dir,
    read_jsonl,
    write_csv,
)
from i2vcompbench.schemas.phase1_legacy import (
    ConceptDistribution,
    DimensionPrior,
    ImageAnalysisResult,
    JointAnalysisResult,
    ManifestItem,
    PriorPackage,
    TextAnalysisResult,
    VisualCompositionPrior,
)

DIMENSIONS = [
    "attribute_binding",
    "motion_binding",
    "spatial_relation",
    "action_binding",
    "scene_dynamics",
    "camera_transformation",
]

DIM_DISPLAY_NAMES = {
    "attribute_binding": "Subject Attribute Binding",
    "motion_binding": "Subject Motion Binding",
    "spatial_relation": "Subject Spatial Relation Composition",
    "action_binding": "Subject Action Binding",
    "scene_dynamics": "Scene / Background Dynamics",
    "camera_transformation": "Camera / View Transformation",
}


# ===================================================================
# Prior Package Assembly
# ===================================================================

def _assemble_prior_package(
    config: dict,
    joint_results: List[JointAnalysisResult],
    dim_priors: List[DimensionPrior],
    global_dists: List[ConceptDistribution],
    global_visual: VisualCompositionPrior,
    pika_data: dict,
    cooccurrence: dict,
    manifest_total: int,
    manifest_clean: int,
) -> PriorPackage:
    """Assemble the complete PriorPackage from all intermediate data."""
    hf_split = config.get("sampling", {}).get("hf_split", "Eval")

    return PriorPackage(
        dataset_name="TIP-I2V",
        split=hf_split,
        total_samples=manifest_total,
        clean_samples=manifest_clean,
        analyzed_samples=len(joint_results),
        global_distributions=global_dists,
        global_visual_prior=global_visual,
        pika_camera_distribution=pika_data.get("pika_camera_distribution", []),
        pika_motion_distribution=pika_data.get("pika_motion_distribution", []),
        dimension_priors=dim_priors,
        dimension_cooccurrence=cooccurrence,
    )


# ===================================================================
# Report Generation
# ===================================================================

def _generate_summary_md(
    package: PriorPackage,
    joint_results: List[JointAnalysisResult],
    text_results: List[TextAnalysisResult],
    image_results: List[ImageAnalysisResult],
    report_dir: str,
) -> None:
    """Generate improved summary.md report."""
    total = package.analyzed_samples
    lines = []

    # Title
    lines.append("# TIP-I2V Prior Analysis Report\n")
    lines.append(f"Dataset: **{package.dataset_name}** ({package.split} split)\n")

    # 1. Dataset Overview
    lines.append("## 1. Dataset Overview\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total samples scanned | {package.total_samples} |")
    lines.append(f"| Clean samples | {package.clean_samples} |")
    lines.append(f"| Joint analysis samples | {package.analyzed_samples} |")
    lines.append("")

    # 2. Global Text Distribution
    lines.append("## 2. Global Text Concept Distribution\n")
    for dist in package.global_distributions:
        lines.append(f"### {dist.category}\n")
        lines.append("| Concept | Count | % |")
        lines.append("|---------|-------|---|")
        for entry in dist.entries[:15]:
            lines.append(f"| {entry['name']} | {entry['count']} | {entry['pct']}% |")
        lines.append("")

    # 3. Global Visual Composition
    lines.append("## 3. Global Visual Composition\n")
    vp = package.global_visual_prior
    if vp.typical_subject_categories:
        lines.append("### Top Subject Categories\n")
        lines.append("| Category | Count | % |")
        lines.append("|----------|-------|---|")
        for entry in vp.typical_subject_categories[:10]:
            lines.append(f"| {entry['value']} | {entry['count']} | {entry['pct']}% |")
        lines.append("")

    if vp.shot_type_distribution:
        lines.append("### Shot Type Distribution\n")
        lines.append("| Shot Type | Count | % |")
        lines.append("|-----------|-------|---|")
        for entry in vp.shot_type_distribution:
            lines.append(f"| {entry['value']} | {entry['count']} | {entry['pct']}% |")
        lines.append("")

    if vp.scene_type_distribution:
        lines.append("### Scene Type Distribution (Top 10)\n")
        lines.append("| Lighting / Weather / Time | Count | % |")
        lines.append("|---------------------------|-------|---|")
        for entry in vp.scene_type_distribution[:10]:
            lines.append(f"| {entry['value']} | {entry['count']} | {entry['pct']}% |")
        lines.append("")

    # 4. Pika Parameter Distribution
    lines.append("## 4. Pika Camera/Motion Parameter Distribution\n")
    lines.append("These reflect real Pika users' camera and motion preferences.\n")

    if package.pika_camera_distribution:
        lines.append("### Pika Camera Commands\n")
        lines.append("| Command | Count | % |")
        lines.append("|---------|-------|---|")
        for entry in package.pika_camera_distribution[:15]:
            lines.append(f"| {entry['value']} | {entry['count']} | {entry['pct']}% |")
        lines.append("")

    if package.pika_motion_distribution:
        lines.append("### Pika Motion Levels\n")
        lines.append("| Level | Count | % |")
        lines.append("|-------|-------|---|")
        for entry in package.pika_motion_distribution:
            lines.append(f"| {entry['value']} | {entry['count']} | {entry['pct']}% |")
        lines.append("")

    # 5. Dimension Coverage Analysis
    lines.append("## 5. Dimension Coverage Analysis\n")
    lines.append("| Dimension | Evaluable Count | Coverage % |")
    lines.append("|-----------|-----------------|------------|")
    for dp in package.dimension_priors:
        lines.append(f"| {dp.display_name} | {dp.sample_count} | {dp.coverage_pct}% |")
    lines.append("")

    # 6. Per-Dimension Prior Deep-Dive
    lines.append("## 6. Per-Dimension Prior Analysis\n")
    for dp in package.dimension_priors:
        lines.append(f"### {dp.display_name}\n")
        lines.append(f"- Evaluable samples: **{dp.sample_count}** ({dp.coverage_pct}%)\n")

        # Concept distributions
        if dp.concept_distributions:
            for cd in dp.concept_distributions:
                lines.append(f"**{cd.category}** (Top 10):\n")
                lines.append("| Concept | Count | % |")
                lines.append("|---------|-------|---|")
                for entry in cd.entries[:10]:
                    lines.append(f"| {entry['name']} | {entry['count']} | {entry['pct']}% |")
                lines.append("")

        # Visual composition
        vp = dp.visual_prior
        if vp.typical_subject_categories:
            lines.append("**Typical Image Subjects** (Top 5):\n")
            subjects_str = ", ".join(
                f"{e['value']}({e['pct']}%)" for e in vp.typical_subject_categories[:5]
            )
            lines.append(f"- {subjects_str}\n")

        if vp.shot_type_distribution:
            lines.append("**Shot Types**: ")
            shots_str = ", ".join(
                f"{e['value']}({e['pct']}%)" for e in vp.shot_type_distribution[:5]
            )
            lines.append(f"{shots_str}\n")

        # Structural templates
        if dp.structural_templates:
            lines.append("**Prompt Structural Templates** (Top 5):\n")
            for i, t in enumerate(dp.structural_templates[:5], 1):
                lines.append(f"{i}. `{t['pattern']}` (count={t['count']})")
            lines.append("")

        # Seed examples
        if dp.seed_examples:
            lines.append("**Seed Examples** (Top 3):\n")
            for i, se in enumerate(dp.seed_examples[:3], 1):
                lines.append(f"{i}. **Prompt**: \"{se.clean_prompt}\"")
                lines.append(f"   - Slots: `{json.dumps(se.text_slots, ensure_ascii=False)[:200]}`")
                subj_names = [s.get("name", "?") for s in se.image_subjects[:3]]
                lines.append(f"   - Image subjects: {', '.join(subj_names)}")
            lines.append("")

        # Constraints
        if dp.constraints:
            lines.append("**Dimension Constraints**:\n")
            lines.append(f"- {dp.constraints.get('description', 'N/A')}")
            for k, v in dp.constraints.items():
                if k != "description":
                    lines.append(f"- `{k}`: {v}")
            lines.append("")

        lines.append("---\n")

    # 7. Gap Analysis
    lines.append("## 7. Dimension Gap Analysis\n")
    lines.append("Dimensions with low natural coverage require heavy synthetic prompt construction:\n")
    for dp in package.dimension_priors:
        if dp.coverage_pct < 5:
            status = "VERY LOW - needs extensive synthetic construction"
        elif dp.coverage_pct < 15:
            status = "LOW - needs significant synthetic augmentation"
        elif dp.coverage_pct < 30:
            status = "MODERATE - some natural samples + synthetic fill"
        else:
            status = "GOOD - sufficient natural coverage"
        lines.append(f"- **{dp.display_name}**: {dp.coverage_pct}% → {status}")
    lines.append("")

    # 8. Co-occurrence
    lines.append("## 8. Dimension Co-occurrence\n")
    lines.append("How often two dimensions are both requested in the same prompt:\n")
    if package.dimension_cooccurrence:
        lines.append("| Dimension Pair | Co-occurrence % |")
        lines.append("|---------------|-----------------|")
        sorted_pairs = sorted(
            package.dimension_cooccurrence.items(),
            key=lambda x: x[1], reverse=True
        )
        for pair, pct in sorted_pairs[:15]:
            lines.append(f"| {pair} | {pct}% |")
        lines.append("")

    # 9. Next Steps
    lines.append("## 9. Key Findings & Recommendations\n")
    lines.append("1. The **prior_package.json** file contains all structured priors needed for "
                 "downstream LLM prompt synthesis.\n")
    lines.append("2. Each dimension's `seed_examples` provide annotated few-shot examples for "
                 "LLM-based prompt generation.\n")
    lines.append("3. The `structural_templates` reveal common prompt patterns in real I2V data, "
                 "which can guide template-based prompt synthesis.\n")
    lines.append("4. The `visual_prior` per dimension describes what kind of first-frame images "
                 "are typical, guiding text-to-image model prompt design.\n")
    lines.append("5. Pika camera/motion distributions reflect real user preferences for camera "
                 "control, directly informing the camera_transformation dimension synthesis.\n")

    # Write
    md_path = str(Path(report_dir) / "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Wrote summary report: {md_path}")


# ===================================================================
# Main Entry
# ===================================================================

def generate_prior_package(config: dict) -> None:
    """
    Main entry: assemble Prior Package from step 4 outputs and generate reports.
    """
    output_base = Path(config["paths"]["output_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    joint_dir = output_base / "joint_analysis"
    report_dir = output_base / "reports"
    ensure_dir(str(report_dir))

    # ------ Load step 4 outputs ------
    joint_results = read_jsonl(str(joint_dir / "joint_analysis.jsonl"), JointAnalysisResult)
    if not joint_results:
        logger.error("No joint analysis results found. Run step4 first.")
        return

    dim_priors = read_jsonl(str(joint_dir / "dimension_priors.jsonl"), DimensionPrior)
    global_dists = read_jsonl(str(joint_dir / "global_distributions.jsonl"), ConceptDistribution)

    # Load global visual prior JSON
    gvp_path = joint_dir / "global_visual_prior.json"
    global_visual = VisualCompositionPrior()
    if gvp_path.exists():
        with open(str(gvp_path), "r", encoding="utf-8") as f:
            global_visual = VisualCompositionPrior(**json.load(f))

    # Load Pika distributions
    pika_path = joint_dir / "pika_distributions.json"
    pika_data = {}
    if pika_path.exists():
        with open(str(pika_path), "r", encoding="utf-8") as f:
            pika_data = json.load(f)

    # Load co-occurrence
    cooc_path = joint_dir / "dimension_cooccurrence.json"
    cooccurrence = {}
    if cooc_path.exists():
        with open(str(cooc_path), "r", encoding="utf-8") as f:
            cooccurrence = json.load(f)

    # Load text/image results for report generation
    text_results = read_jsonl(
        str(output_base / "text_analysis" / "text_parse.jsonl"), TextAnalysisResult
    )
    image_results = read_jsonl(
        str(output_base / "image_analysis" / "image_parse.jsonl"), ImageAnalysisResult
    )

    # Manifest counts
    manifest_all = read_jsonl(str(manifest_dir / "manifest.jsonl"))
    manifest_clean = read_jsonl(str(manifest_dir / "manifest_clean.jsonl"))

    # ====== Assemble Prior Package ======
    logger.info("Assembling Prior Package...")
    package = _assemble_prior_package(
        config=config,
        joint_results=joint_results,
        dim_priors=dim_priors,
        global_dists=global_dists,
        global_visual=global_visual,
        pika_data=pika_data,
        cooccurrence=cooccurrence,
        manifest_total=len(manifest_all),
        manifest_clean=len(manifest_clean),
    )

    # Write prior_package.json (core deliverable)
    pp_path = str(report_dir / "prior_package.json")
    with open(pp_path, "w", encoding="utf-8") as f:
        json.dump(package.model_dump(), f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote Prior Package: {pp_path}")

    # ====== CSV Reports ======
    # Dataset overview
    overview_data = [{
        "total_scanned": package.total_samples,
        "clean_samples": package.clean_samples,
        "analyzed_samples": package.analyzed_samples,
        "text_analyzed": len(text_results),
        "image_analyzed": len(image_results),
    }]
    write_csv(str(report_dir / "dataset_overview.csv"), overview_data)

    # Dimension analysis summary
    dim_rows = []
    for dp in dim_priors:
        dim_rows.append({
            "dimension": dp.display_name,
            "evaluable_count": dp.sample_count,
            "coverage_pct": dp.coverage_pct,
            "concept_categories": len(dp.concept_distributions),
            "template_count": len(dp.structural_templates),
            "seed_count": len(dp.seed_examples),
        })
    write_csv(str(report_dir / "dimension_analysis_summary.csv"), dim_rows)

    # Gap analysis
    gap_data = []
    total = len(joint_results)
    for dp in dim_priors:
        text_req = sum(
            1 for r in joint_results
            if getattr(r, f"dim_{dp.dimension}").text_requests
        )
        text_pct = round(text_req / total * 100, 2) if total else 0
        gap_data.append({
            "dimension": dp.display_name,
            "text_request_pct": text_pct,
            "evaluable_pct": dp.coverage_pct,
            "gap_pct": round(text_pct - dp.coverage_pct, 2),
            "evaluable_count": dp.sample_count,
        })
    write_csv(str(report_dir / "dimension_gap_analysis.csv"), gap_data)

    # ====== Summary Report ======
    _generate_summary_md(
        package=package,
        joint_results=joint_results,
        text_results=text_results,
        image_results=image_results,
        report_dir=str(report_dir),
    )

    # ====== Final Log ======
    logger.info("=" * 60)
    logger.info("Prior Package Generation Summary")
    logger.info("=" * 60)
    logger.info(f"Prior Package: {pp_path}")
    for dp in dim_priors:
        logger.info(
            f"  {dp.display_name}: {dp.sample_count} samples, "
            f"{len(dp.concept_distributions)} distributions, "
            f"{len(dp.structural_templates)} templates, "
            f"{len(dp.seed_examples)} seeds"
        )
    logger.info(f"Reports: {report_dir}")
