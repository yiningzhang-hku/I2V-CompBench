"""
Unified CLI entry for I2V-CompBench (Phase 1 + Phase 2).

Usage:
    # Phase 1 (data prior preparation)
    python main.py --config configs/phase1.yaml --step manifest
    python main.py --config configs/phase1.yaml --step image
    python main.py --config configs/phase1.yaml --step text
    python main.py --config configs/phase1.yaml --step joint
    python main.py --config configs/phase1.yaml --step report
    python main.py --config configs/phase1.yaml --step patch
    python main.py --config configs/phase1.yaml --step align
    python main.py --config configs/phase1.yaml --step refbank
    python main.py --config configs/phase1.yaml --step priors2
    python main.py --config configs/phase1.yaml --step recipes
    python main.py --config configs/phase1.yaml --step p1audit
    python main.py --config configs/phase1.yaml --step phase1     # all Phase 1 steps

    # Phase 2 (benchmark dataset synthesis)
    python main.py --config configs/phase2.yaml --step quota
    python main.py --config configs/phase2.yaml --step sample
    python main.py --config configs/phase2.yaml --step plan
    python main.py --config configs/phase2.yaml --step construct
    python main.py --config configs/phase2.yaml --step verify
    python main.py --config configs/phase2.yaml --step finalize
    python main.py --config configs/phase2.yaml --step export
    python main.py --config configs/phase2.yaml --step audit
    python main.py --config configs/phase2.yaml --step phase2     # all Phase 2 steps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

# Load .env (project-local secrets like SILICONFLOW_API_KEY) before any client init.
try:
    from dotenv import load_dotenv  # type: ignore
    _ENV_FILE = Path(__file__).resolve().parent / ".env"
    if _ENV_FILE.exists():
        load_dotenv(dotenv_path=_ENV_FILE, override=False)
        logger.debug(f"Loaded env from {_ENV_FILE}")
except ImportError:
    pass  # python-dotenv 可选；未安装时沉默退化为仅读进程环境变量


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve().parent
    src = here / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


# ============================================================
# Phase 2 step functions
# ============================================================

def _step_quota(cfg, paths):
    from i2vcompbench.phase2.build_quota import build_quota
    from i2vcompbench.utils.io import write_json

    plan = build_quota(cfg)
    write_json(paths["quota_plan"], plan.model_dump())
    logger.info(f"[quota] wrote {paths['quota_plan']}")


def _step_sample(cfg, paths):
    from i2vcompbench.phase2.sample_recipes import sample_recipes
    from i2vcompbench.utils.io import write_json, write_jsonl

    sampled, report = sample_recipes(cfg)
    write_jsonl(paths["sampled_recipes"], [s.model_dump() for s in sampled])
    write_json(paths["quota_unfilled_report"], report)
    logger.info(f"[sample] wrote {len(sampled)} sampled recipes")


def _step_plan(cfg, paths):
    from i2vcompbench.phase2.build_question_plan import build_question_plans
    from i2vcompbench.utils.io import write_jsonl

    plans = build_question_plans(cfg)
    write_jsonl(paths["question_plans"], [p.model_dump() for p in plans])
    logger.info(f"[plan] wrote {len(plans)} question plans")


def _step_construct(cfg, paths):
    from i2vcompbench.phase2.construct_inputs import construct_inputs
    from i2vcompbench.utils.io import write_jsonl

    manifests = construct_inputs(cfg)
    write_jsonl(paths["input_assets_manifest"], [m.model_dump() for m in manifests])
    logger.info(f"[construct] wrote {len(manifests)} input manifests")


def _step_verify(cfg, paths):
    from i2vcompbench.phase2.verify_inputs import verify_inputs

    verify_inputs(cfg)


def _step_finalize(cfg, paths):
    from i2vcompbench.phase2.finalize_prompts import finalize_prompts
    from i2vcompbench.utils.io import write_jsonl

    entries = finalize_prompts(cfg)
    write_jsonl(paths["final_prompts"], [e.model_dump() for e in entries])
    logger.info(f"[finalize] wrote {len(entries)} final prompts")


def _step_export(cfg, paths):
    from i2vcompbench.phase2.export_dataset import export_dataset

    samples = export_dataset(cfg)
    logger.info(f"[export] wrote {len(samples)} samples")


def _step_package(cfg, paths):
    from i2vcompbench.phase2.package_by_dimension import package_by_dimension

    counts = package_by_dimension(cfg)
    logger.info(f"[package] per-dim counts: {counts}")


def _step_audit(cfg, paths):
    from i2vcompbench.phase2.audit_phase2 import audit

    audit(cfg)


# ============================================================
# Phase 1 step functions (config-driven, do not need `paths`)
# ============================================================

def _run_p1(name, func, cfg):
    logger.info(f"=== Phase 1 step: {name} ===")
    start = time.time()
    func(cfg)
    logger.info(f"[p1.{name}] done in {time.time() - start:.1f}s")


def _step_p1_manifest(cfg, paths):
    from i2vcompbench.phase1.step1_manifest import build_manifest
    _run_p1("manifest", build_manifest, cfg)


def _step_p1_image(cfg, paths):
    from i2vcompbench.phase1.step2_image_analysis import analyze_images
    _run_p1("image", analyze_images, cfg)


def _step_p1_text(cfg, paths):
    from i2vcompbench.phase1.step3_text_analysis import analyze_texts
    _run_p1("text", analyze_texts, cfg)


def _step_p1_joint(cfg, paths):
    from i2vcompbench.phase1.step4_joint_analysis import joint_analyze
    _run_p1("joint", joint_analyze, cfg)


def _step_p1_report(cfg, paths):
    from i2vcompbench.phase1.step5_pool_and_report import generate_prior_package
    _run_p1("report", generate_prior_package, cfg)


def _step_p1_patch(cfg, paths):
    from i2vcompbench.phase1.patch_existing_outputs import patch_outputs
    _run_p1("patch", patch_outputs, cfg)


def _step_p1_align(cfg, paths):
    from i2vcompbench.phase1.align_instances import align_instances
    _run_p1("align", align_instances, cfg)


def _step_p1_refbank(cfg, paths):
    from i2vcompbench.phase1.reference_bank import build_reference_bank
    _run_p1("refbank", build_reference_bank, cfg)


def _step_p1_priors2(cfg, paths):
    from i2vcompbench.phase1.priors_enhance import enhance_priors
    _run_p1("priors2", enhance_priors, cfg)


def _step_p1_recipes(cfg, paths):
    from i2vcompbench.phase1.recipes import build_recipes
    _run_p1("recipes", build_recipes, cfg)


def _step_p1_audit(cfg, paths):
    from i2vcompbench.phase1.audit import audit as p1_audit
    _run_p1("audit", p1_audit, cfg)


# ============================================================
# Step registry
# ============================================================

_PHASE2_STEPS = {
    "quota": _step_quota,
    "sample": _step_sample,
    "plan": _step_plan,
    "construct": _step_construct,
    "verify": _step_verify,
    "finalize": _step_finalize,
    "export": _step_export,
    "audit": _step_audit,
    "package": _step_package,
}

_PHASE1_STEPS = {
    "manifest": _step_p1_manifest,
    "image": _step_p1_image,
    "text": _step_p1_text,
    "joint": _step_p1_joint,
    "report": _step_p1_report,
    "patch": _step_p1_patch,
    "align": _step_p1_align,
    "refbank": _step_p1_refbank,
    "priors2": _step_p1_priors2,
    "recipes": _step_p1_recipes,
    "p1audit": _step_p1_audit,
}

_STEPS = {**_PHASE2_STEPS, **_PHASE1_STEPS}

_PHASE2_PIPELINE = [
    "quota", "sample", "plan", "construct",
    "verify", "finalize", "export", "audit", "package",
]

_PHASE1_PIPELINE = [
    "manifest", "image", "text", "joint", "report",
    "patch", "align", "refbank", "priors2", "recipes", "p1audit",
]


def main() -> int:
    _ensure_src_on_path()
    from i2vcompbench.utils.io import benchmark_paths, load_config

    parser = argparse.ArgumentParser(description="I2V-CompBench Phase 1/2 CLI")
    parser.add_argument("--config", required=True, help="Path to phase1.yaml or phase2.yaml")
    parser.add_argument(
        "--step",
        required=True,
        choices=list(_STEPS.keys()) + ["phase1", "phase2"],
    )
    parser.add_argument("--mode", choices=["pilot", "full"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode

    # Phase 2 needs benchmark_paths from output_dir; Phase 1 doesn't.
    paths = None
    if args.step in _PHASE2_PIPELINE or args.step == "phase2":
        paths = benchmark_paths(cfg["output_dir"])

    if args.step == "phase2":
        for step in _PHASE2_PIPELINE:
            logger.info(f"=== running step: {step} ===")
            try:
                _PHASE2_STEPS[step](cfg, paths)
            except Exception as e:
                logger.error(f"step {step} failed: {e}")
                if step in ("quota", "sample", "plan"):
                    return 2
        return 0

    if args.step == "phase1":
        for step in _PHASE1_PIPELINE:
            try:
                _PHASE1_STEPS[step](cfg, paths)
            except Exception as e:
                logger.error(f"phase1 step {step} failed: {e}")
                return 2
        return 0

    _STEPS[args.step](cfg, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
