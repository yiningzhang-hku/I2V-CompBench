"""
Phase 2 · Step 5: structured VQA QC for constructed inputs.

For each question:
  1. Load its primary image (single -> images/<qid>.png; multi -> first reference)
  2. Render the dimension's VQA QC prompt (prompts/vqa_qc/<dim>.txt)
  3. Call Phase2SiliconFlowClient.call_vqa_structured
  4. Aggregate `pass / fail / needs_manual_review`
  5. Write qc_reports/<question_id>.json

needs_manual_review when at least one check has confidence < min_conf but answer=true,
or when the parser returns 0 checks (LLM didn't follow protocol).
fail when any HARD check returns answer=false with high confidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..schemas.phase2 import QCCheck, QCReport
from ..utils.api_client import Phase2SiliconFlowClient
from ..utils.io import (
    benchmark_paths,
    iter_jsonl,
    load_config,
    read_json,
    write_json,
    write_jsonl,
)


_VQA_PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts" / "vqa_qc"


# Hard checks: a high-confidence false answer triggers `fail`
_HARD_CHECK_NAMES = {
    "has_target_subject_visible",
    "single_target_subject",
    "all_required_subjects_visible",
    "subject_visible",
    "target_subject_static",
    "scene_has_subject_room",
    "no_target_attribute_already",
    "no_action_already_in_progress",
    "no_motion_blur_present",
    "interaction_not_yet_started",
    "resolution_ok",
}


def _load_qc_prompt(dimension: str) -> str:
    p = _VQA_PROMPT_DIR / f"{dimension}.txt"
    if not p.exists():
        logger.warning(f"VQA prompt not found: {p}; using empty prompt")
        return ""
    return p.read_text(encoding="utf-8")


def _format_qc_prompt(template: str, plan: Dict[str, Any]) -> str:
    """Substitute {target_subject} {subjects} {expected_change} {subtype} placeholders."""
    target_plan = plan.get("target_plan") or {}
    target_subject = target_plan.get("target_subject", "the subject")
    expected_change = target_plan.get("expected_final_state", "")
    subtype = plan.get("subtype", "")
    subjects = ", ".join(
        [s.get("description") or s.get("role", "") for s in (plan.get("input_plan") or {}).get("required_images", [])]
    )
    repl = {
        "{target_subject}": target_subject,
        "{subjects}": subjects or target_subject,
        "{expected_change}": expected_change,
        "{subtype}": subtype,
    }
    out = template
    for k, v in repl.items():
        out = out.replace(k, str(v))
    return out


def _pick_primary_image(question_id: str, plan: Dict[str, Any], paths: Dict[str, Path]) -> Optional[Path]:
    if plan.get("input_mode") == "single_image":
        p = paths["first_frames"] / f"{question_id}.png"
        return p if p.exists() else None
    # multi_image: scan ref_images/ for any file matching the question_id
    candidates = sorted(paths["ref_images"].glob(f"{question_id}_ref*.png"))
    if candidates:
        return candidates[0]
    return None


def _aggregate_status(
    checks: List[QCCheck],
    min_conf: float,
) -> Tuple[str, List[str]]:
    if not checks:
        return "needs_manual_review", ["empty_or_unparseable_response"]
    risk: List[str] = []
    has_hard_fail = False
    has_low_conf = False
    for c in checks:
        if c.name in _HARD_CHECK_NAMES and c.answer is False and c.confidence >= min_conf:
            has_hard_fail = True
            risk.append(f"hard_fail::{c.name}")
        elif c.answer is False and c.confidence >= min_conf:
            risk.append(f"soft_fail::{c.name}")
        elif c.confidence < min_conf:
            has_low_conf = True
            risk.append(f"low_confidence::{c.name}")
    if has_hard_fail:
        return "fail", risk
    if has_low_conf and any(c.answer is False for c in checks):
        return "needs_manual_review", risk
    if has_low_conf:
        return "needs_manual_review", risk
    return "pass", risk


# ============================================================
# Driver
# ============================================================

def verify_inputs(config: Dict[str, Any]) -> List[QCReport]:
    paths = benchmark_paths(config["output_dir"])
    plans_path = paths["question_plans"]
    if not plans_path.exists():
        raise FileNotFoundError(f"question_plans.jsonl not found at {plans_path}")

    verify_cfg = config.get("verify", {})
    min_conf = float(verify_cfg.get("vqa_min_confidence", 0.7))

    try:
        client = Phase2SiliconFlowClient(config)
    except Exception as e:  # noqa: BLE001
        logger.error(f"VLM client init failed; cannot run verify: {e}")
        return []

    reports: List[QCReport] = []
    failed_to_retry: List[Dict[str, Any]] = []
    manual_queue: List[Dict[str, Any]] = []

    for plan in iter_jsonl(plans_path):
        qid = plan["question_id"]
        primary = _pick_primary_image(qid, plan, paths)
        if primary is None or not primary.exists():
            r = QCReport(
                question_id=qid,
                qc_status="fail",
                checks=[],
                risk_flags=["missing_primary_image"],
                retry_count=0,
                notes="no constructed image found on disk",
            )
            reports.append(r)
            failed_to_retry.append({"question_id": qid, "reason": "missing_primary_image"})
            continue

        prompt_template = _load_qc_prompt(plan["dimension"])
        prompt_text = _format_qc_prompt(prompt_template, plan)
        result = client.call_vqa_structured(str(primary), prompt_text)
        raw_checks = result.get("checks") or []
        parsed_checks: List[QCCheck] = []
        for c in raw_checks:
            try:
                parsed_checks.append(
                    QCCheck(
                        name=str(c.get("name") or ""),
                        answer=bool(c.get("answer", False)),
                        confidence=float(c.get("confidence") or 0.0),
                        rationale=str(c.get("rationale") or ""),
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[{qid}] dropped malformed check: {c} ({e})")
        status, risk = _aggregate_status(parsed_checks, min_conf)
        report = QCReport(
            question_id=qid,
            qc_status=status,  # type: ignore[arg-type]
            checks=parsed_checks,
            risk_flags=risk,
            retry_count=0,
            notes="" if parsed_checks else (result.get("raw") or "")[:300],
        )
        reports.append(report)

        # write per-question file
        per_q_path = paths["qc_reports"] / f"{qid}.json"
        write_json(per_q_path, report.model_dump())

        if status == "fail":
            failed_to_retry.append({"question_id": qid, "risk_flags": risk})
        elif status == "needs_manual_review":
            manual_queue.append({"question_id": qid, "risk_flags": risk})

    if failed_to_retry:
        write_jsonl(paths["qc_failed_to_retry"], failed_to_retry)
    if manual_queue:
        write_jsonl(paths["manual_review_queue"], manual_queue)

    n_pass = sum(1 for r in reports if r.qc_status == "pass")
    n_fail = sum(1 for r in reports if r.qc_status == "fail")
    n_review = sum(1 for r in reports if r.qc_status == "needs_manual_review")
    logger.info(f"Verify done: pass={n_pass} fail={n_fail} review={n_review}")
    return reports


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 verify_inputs")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    verify_inputs(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
