"""Fast finalize prompts with concurrency + checkpoint resume."""
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from i2vcompbench.utils.io import load_config, benchmark_paths, iter_jsonl, read_json, write_jsonl
from i2vcompbench.utils.api_client import Phase2SiliconFlowClient
from i2vcompbench.phase2.finalize_prompts import (
    _qc_passed, _describe_frame, _format_polish_prompt, _load_polish_template,
    _parse_polish_response, _enforce_constraints
)

cfg = load_config("configs/phase2.yaml")
paths = benchmark_paths(cfg["output_dir"])

# Load plans
plans = list(iter_jsonl(paths["question_plans"]))
logger.info(f"Loaded {len(plans)} question plans")

# Filter QC-passed
qc_passed_plans = []
for plan in plans:
    qid = plan["question_id"]
    if _qc_passed(paths["qc_reports"], qid):
        qc_passed_plans.append(plan)
logger.info(f"QC passed: {len(qc_passed_plans)}")

# Load existing results for checkpoint resume
output_file = paths["final_prompts"]
existing = {}
if output_file.exists():
    for line in output_file.read_text(encoding="utf-8").strip().split("\n"):
        if line:
            obj = json.loads(line)
            existing[obj["question_id"]] = obj
    logger.info(f"Existing finalized prompts: {len(existing)}, remaining: {len(qc_passed_plans) - len(existing)}")

# Init client
client = Phase2SiliconFlowClient(cfg)
template = _load_polish_template()

pf_cfg = cfg.get("prompt_finalize", {})
min_words = int(pf_cfg.get("min_words", 8))
max_words_default = int(pf_cfg.get("max_words", 25))
max_words_inter = int(pf_cfg.get("max_words_interaction", 30))

def process_one(plan):
    qid = plan["question_id"]
    if qid in existing:
        return qid, existing[qid]
    
    # locate primary image
    primary = None
    if plan.get("input_mode") == "single_image":
        p = paths["first_frames"] / f"{qid}.png"
        primary = p if p.exists() else None
    else:
        cand = sorted(paths["ref_images"].glob(f"{qid}_ref*.png"))
        primary = cand[0] if cand else None

    frame_desc = _describe_frame(client, primary) if primary else ""
    dimension = plan.get("dimension", "")
    max_words = max_words_inter if dimension == "interaction_reasoning" else max_words_default

    polish_prompt = _format_polish_prompt(template, plan, frame_desc, min_words, max_words)
    
    try:
        raw = client.call_llm(polish_prompt) if template else ""
    except Exception as e:
        logger.warning(f"[{qid}] LLM call failed: {e}")
        raw = ""
    
    parsed = _parse_polish_response(raw) if raw else None
    forbidden = (plan.get("dimension_isolation") or {}).get("forbidden_words") or []
    
    candidate_prompt = ""
    used_fallback = False
    polish_attempts = 1
    
    if parsed:
        candidate_prompt = str(parsed.get("prompt") or "")
    
    if candidate_prompt:
        check = _enforce_constraints(candidate_prompt, forbidden, min_words, max_words, dimension)
        if not check["ok"]:
            candidate_prompt = ""
    
    if not candidate_prompt:
        candidate_prompt = plan.get("prompt_draft") or ""
        polish_attempts += 1
        used_fallback = True
    
    check_final = _enforce_constraints(candidate_prompt, forbidden, min_words, max_words, dimension)
    
    entry = {
        "question_id": qid,
        "prompt": candidate_prompt.strip(),
        "length_words": check_final["word_count"],
        "forbidden_hits": check_final["hits"],
        "polish_attempts": polish_attempts,
        "used_fallback": used_fallback,
        "vlm_caption": frame_desc,
        "failed_check": check_final["failed_check"],
    }
    return qid, entry

# Process with thread pool
results = dict(existing)
todo = [p for p in qc_passed_plans if p["question_id"] not in existing]
logger.info(f"Processing {len(todo)} questions with 3 workers...")

start_time = time.time()
done_count = 0

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {executor.submit(process_one, plan): plan for plan in todo}
    
    for future in as_completed(futures):
        plan = futures[future]
        try:
            qid, entry = future.result()
            results[qid] = entry
            done_count += 1
            
            if done_count % 20 == 0 or done_count == len(todo):
                elapsed = time.time() - start_time
                rate = done_count / elapsed * 60 if elapsed > 0 else 0
                remaining = len(todo) - done_count
                eta_min = remaining / (rate / 60) if rate > 0 else 0
                logger.info(f"[{done_count}/{len(todo)}] {qid} (elapsed {elapsed:.0f}s, {rate:.1f}/min, ETA {eta_min:.0f}min)")
                # Write checkpoint
                write_jsonl(output_file, list(results.values()))
        except Exception as e:
            logger.error(f"[{plan['question_id']}] failed: {e}")

# Final write
write_jsonl(output_file, list(results.values()))
logger.info(f"Done! Wrote {len(results)} final prompts -> {output_file}")
