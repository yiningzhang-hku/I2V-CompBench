"""Run Phase 2 verify with per-question progress."""
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))
load_dotenv(Path(__file__).parent / ".env")

from i2vcompbench.phase2.verify_inputs import (
    _pick_primary_image, _load_qc_prompt, _format_qc_prompt, _aggregate_status
)
from i2vcompbench.schemas.phase2 import QCCheck, QCReport
from i2vcompbench.utils.api_client import Phase2SiliconFlowClient
from i2vcompbench.utils.io import load_config, benchmark_paths, iter_jsonl, write_json, write_jsonl

cfg = load_config("configs/phase2.yaml")
paths = benchmark_paths(cfg["output_dir"])
verify_cfg = cfg.get("verify", {})
min_conf = float(verify_cfg.get("vqa_min_confidence", 0.7))

plans = list(iter_jsonl(paths["question_plans"]))
print(f"Loaded {len(plans)} question plans")

# Check which reports already exist (skip already verified)
existing = set()
for f in paths["qc_reports"].glob("*.json"):
    existing.add(f.stem)
print(f"Existing QC reports: {len(existing)}, remaining: {len(plans) - len(existing)}")

client = Phase2SiliconFlowClient(cfg)

reports = []
failed_to_retry = []
manual_queue = []
start = time.time()

for i, plan in enumerate(plans):
    qid = plan["question_id"]
    if qid in existing:
        # skip already verified
        continue
    if i % 20 == 0:
        elapsed = time.time() - start
        print(f"[{i}/{len(plans)}] {qid} (elapsed {elapsed:.1f}s, reports {len(reports)})")

    primary = _pick_primary_image(qid, plan, paths)
    if primary is None or not primary.exists():
        r = QCReport(
            question_id=qid, qc_status="fail", checks=[],
            risk_flags=["missing_primary_image"], retry_count=0,
            notes="no constructed image found on disk",
        )
        reports.append(r)
        failed_to_retry.append({"question_id": qid, "reason": "missing_primary_image"})
        per_q_path = paths["qc_reports"] / f"{qid}.json"
        write_json(per_q_path, r.model_dump())
        continue

    prompt_template = _load_qc_prompt(plan["dimension"])
    prompt_text = _format_qc_prompt(prompt_template, plan)
    try:
        result = client.call_vqa_structured(str(primary), prompt_text)
    except Exception as e:
        r = QCReport(
            question_id=qid, qc_status="fail", checks=[],
            risk_flags=["vqa_call_failed"], retry_count=0,
            notes=str(e)[:300],
        )
        reports.append(r)
        failed_to_retry.append({"question_id": qid, "reason": "vqa_call_failed"})
        per_q_path = paths["qc_reports"] / f"{qid}.json"
        write_json(per_q_path, r.model_dump())
        continue

    raw_checks = result.get("checks") or []
    parsed_checks = []
    for c in raw_checks:
        try:
            parsed_checks.append(QCCheck(
                name=str(c.get("name") or ""),
                answer=bool(c.get("answer", False)),
                confidence=float(c.get("confidence") or 0.0),
                rationale=str(c.get("rationale") or ""),
            ))
        except Exception:
            pass

    status, risk = _aggregate_status(parsed_checks, min_conf)
    report = QCReport(
        question_id=qid, qc_status=status, checks=parsed_checks,
        risk_flags=risk, retry_count=0,
        notes="" if parsed_checks else (result.get("raw") or "")[:300],
    )
    reports.append(report)

    per_q_path = paths["qc_reports"] / f"{qid}.json"
    write_json(per_q_path, report.model_dump())

    if status == "fail":
        failed_to_retry.append({"question_id": qid, "risk_flags": risk})
    elif status == "needs_manual_review":
        manual_queue.append({"question_id": qid, "risk_flags": risk})

print(f"Verify complete: {len(reports)} reports, {len(failed_to_retry)} failed, {len(manual_queue)} manual")

if failed_to_retry:
    write_jsonl(paths["qc_failed_to_retry"], failed_to_retry)
if manual_queue:
    write_jsonl(paths["manual_review_queue"], manual_queue)
