"""
Phase 2 · Step 6: finalize I2V prompts for QC-passed questions.

Pipeline per question:
  1. Skip if QC status != pass
  2. Read first-frame description (one-sentence VLM caption)
  3. Render prompts/prompt_polish.txt with all constraints
  4. Call LLM to produce final prompt
  5. Enforce length / forbidden_word / active_verb / view-camera 检查；
     失败时回退到 prompt_draft，并写 failed_check 标记

Output: data/benchmark_dataset/prompts/final_prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ..schemas.phase2 import FinalPromptEntry
from ..utils.api_client import Phase2SiliconFlowClient
from ..utils.io import (
    benchmark_paths,
    iter_jsonl,
    load_config,
    read_json,
    write_jsonl,
)
from ..utils.templates import count_words, find_forbidden_hits


_POLISH_PROMPT_PATH = Path(__file__).resolve().parents[3].parent / "prompts" / "prompt_polish.txt"


# ============================================================
# active_verb / view-camera 硬约束
# ============================================================

# SpaCy 模型懒加载（§6.6 要求使用 POS 检测 active verb）
_NLP = None


def _get_nlp():
    """Lazy-load SpaCy English model; fallback to None if unavailable."""
    global _NLP
    if _NLP is False:
        return None
    if _NLP is None:
        try:
            import spacy
            try:
                _NLP = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "SpaCy model 'en_core_web_sm' not found. "
                    "Run: python -m spacy download en_core_web_sm. "
                    "Falling back to regex-based active verb detection."
                )
                _NLP = False
                return None
        except ImportError:
            logger.warning(
                "SpaCy not installed. Falling back to regex-based active verb detection."
            )
            _NLP = False
            return None
    return _NLP


# 仅举静态词头作为反例信号（不构成主动谓语）
_STATIC_LEAD_TOKENS = {
    "a", "an", "the", "this", "that", "these", "those",
    "there", "it", "its",
}
_BE_VERBS = {"is", "are", "was", "were", "be", "being", "been", "am"}

# view_transformation 维度不允许的 camera 词汇
VIEW_CAMERA_BLACKLIST: List[str] = [
    "camera pans", "camera pan", "camera tilts", "camera tilt",
    "camera zooms", "camera zoom", "dolly", "crane shot", "tracking shot",
    "zoom in", "zoom out", "pan left", "pan right", "tilt up", "tilt down",
]


def _has_active_verb(text: str) -> bool:
    """检查 prompt 中是否包含至少一个 active verb。

    优先使用 SpaCy POS 标注（§6.6）：检查是否存在
    VB/VBZ/VBP/VBG 标签且非 be 动词。
    SpaCy 不可用时回退到 regex 后缀检测。
    """
    if not text:
        return False

    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        for token in doc:
            if token.pos_ == "VERB" and token.tag_ in ("VB", "VBZ", "VBP", "VBG", "VBD", "VBN"):
                if token.lemma_.lower() not in _BE_VERBS:
                    return True
        return False

    # Fallback: regex-based suffix matching
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    for tok in tokens:
        if tok in _BE_VERBS or tok in _STATIC_LEAD_TOKENS:
            continue
        if len(tok) >= 4 and (
            tok.endswith("ing") or tok.endswith("ed") or tok.endswith("s")
        ):
            return True
    return False


def _hits_view_camera_blacklist(text: str) -> List[str]:
    """view_transformation 维度的 camera 词检查。"""
    if not text:
        return []
    low = text.lower()
    return [w for w in VIEW_CAMERA_BLACKLIST if w in low]


# ============================================================
# Polish 模板加载与渲染
# ============================================================

def _load_polish_template() -> str:
    if not _POLISH_PROMPT_PATH.exists():
        logger.warning(f"prompt_polish template missing: {_POLISH_PROMPT_PATH}")
        return ""
    return _POLISH_PROMPT_PATH.read_text(encoding="utf-8")


def _format_polish_prompt(
    template: str,
    plan: Dict[str, Any],
    frame_description: str,
    min_words: int,
    max_words: int,
) -> str:
    target_plan = plan.get("target_plan") or {}
    iso = plan.get("dimension_isolation") or {}
    preserve = plan.get("preserve_plan") or []

    # 新版 target_plan 用 target_subjects[]，兼容旧版 target_subject
    subjects_list = target_plan.get("target_subjects") or []
    if subjects_list and isinstance(subjects_list, list):
        target_subjects = ", ".join(
            (s.get("description") or s.get("id") or "") for s in subjects_list
        ) or "the subject"
    else:
        target_subjects = target_plan.get("target_subject") or "the subject"

    reference_subjects = ""
    for it in (plan.get("input_plan") or {}).get("required_images") or []:
        if it.get("role") in ("reference_subject", "attribute_reference", "scene_reference"):
            reference_subjects = (it.get("description") or it.get("role"))
            break

    repl = {
        "{dimension}": plan.get("dimension", ""),
        "{subtype}": plan.get("subtype", ""),
        "{input_mode}": plan.get("input_mode", ""),
        "{frame_description}": frame_description or "",
        "{target_subjects}": target_subjects,
        "{reference_subjects}": reference_subjects or "(none)",
        "{expected_change}": target_plan.get("expected_final_state", ""),
        "{preserve_constraints}": "; ".join(
            f"{p.get('scope')}={p.get('constraint')}" for p in preserve
        ),
        "{forbidden_words}": ", ".join(iso.get("forbidden_words") or []) or "(none)",
        "{camera_constraint}": iso.get("camera_constraint", "forbidden"),
        "{prompt_draft}": plan.get("prompt_draft", ""),
        "{min_words}": str(min_words),
        "{max_words}": str(max_words),
    }
    out = template
    for k, v in repl.items():
        out = out.replace(k, str(v))
    return out


def _parse_polish_response(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidates = [text.strip()]
    for m in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append(m.group(1).strip())
    i = text.find("{")
    if i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[i : j + 1])
                    break
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and ("prompt" in obj or "i2v_prompt" in obj):
                # 兼容旧产物：仅有 i2v_prompt 时回填到 prompt
                if "prompt" not in obj and "i2v_prompt" in obj:
                    obj["prompt"] = obj["i2v_prompt"]
                return obj
        except json.JSONDecodeError:
            continue
    return None


# ============================================================
# QC / VLM caption / 约束检查
# ============================================================

def _qc_passed(qc_dir: Path, qid: str) -> bool:
    p = qc_dir / f"{qid}.json"
    if not p.exists():
        return False
    try:
        data = read_json(p)
    except Exception:
        return False
    return data.get("qc_status") == "pass"


def _describe_frame(client: Phase2SiliconFlowClient, image_path: Path) -> str:
    if not image_path.exists():
        return ""
    try:
        text = client.call_vlm(
            str(image_path),
            "Describe this image in ONE concise English sentence (max 25 words).",
        )
        return text.strip().replace("\n", " ")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"frame caption failed for {image_path}: {e}")
        return ""


def _enforce_constraints(
    prompt: str,
    forbidden: List[str],
    min_words: int,
    max_words: int,
    dimension: str,
) -> Dict[str, Any]:
    """统一返回 ok / hits / word_count / failed_check（首条命中的失败原因）。"""
    hits = find_forbidden_hits(prompt, forbidden)
    wc = count_words(prompt)
    failed_check: Optional[str] = None

    if not (min_words <= wc <= max_words):
        failed_check = "out_of_range"
    elif hits:
        failed_check = "forbidden_hit"
    elif not _has_active_verb(prompt):
        failed_check = "missing_active_verb"
    elif dimension == "view_transformation":
        cam_hits = _hits_view_camera_blacklist(prompt)
        if cam_hits:
            failed_check = "view_camera_cheat"
            hits = list(hits) + cam_hits

    return {
        "ok": failed_check is None,
        "hits": hits,
        "word_count": wc,
        "failed_check": failed_check,
    }


# ============================================================
# Driver
# ============================================================

def finalize_prompts(config: Dict[str, Any]) -> List[FinalPromptEntry]:
    paths = benchmark_paths(config["output_dir"])
    plans_path = paths["question_plans"]
    if not plans_path.exists():
        raise FileNotFoundError(f"question_plans.jsonl not found at {plans_path}")

    pf_cfg = config.get("prompt_finalize", {})
    min_words = int(pf_cfg.get("min_words", 8))
    max_words_default = int(pf_cfg.get("max_words", 25))
    max_words_inter = int(pf_cfg.get("max_words_interaction", 30))

    template = _load_polish_template()
    try:
        client = Phase2SiliconFlowClient(config)
    except Exception as e:  # noqa: BLE001
        logger.error(f"LLM client init failed: {e}")
        return []

    out: List[FinalPromptEntry] = []
    for plan in iter_jsonl(plans_path):
        qid = plan["question_id"]
        if not _qc_passed(paths["qc_reports"], qid):
            continue

        # locate primary image for caption
        primary: Optional[Path] = None
        if plan.get("input_mode") == "single_image":
            p = paths["first_frames"] / f"{qid}.png"
            primary = p if p.exists() else None
        else:
            cand = sorted(paths["ref_images"].glob(f"{qid}_ref*.png"))
            primary = cand[0] if cand else None

        frame_desc = _describe_frame(client, primary) if primary else ""
        dimension = plan.get("dimension", "")
        max_words = (
            max_words_inter if dimension == "interaction_reasoning" else max_words_default
        )

        polish_prompt = _format_polish_prompt(
            template, plan, frame_desc, min_words, max_words
        )
        raw = client.call_llm(polish_prompt) if template else ""
        parsed = _parse_polish_response(raw) if raw else None

        forbidden = (plan.get("dimension_isolation") or {}).get("forbidden_words") or []
        candidate_prompt: str = ""
        used_fallback = False
        polish_attempts = 1
        vlm_caption = frame_desc

        if parsed:
            candidate_prompt = str(parsed.get("prompt") or "")

        if candidate_prompt:
            check = _enforce_constraints(
                candidate_prompt, forbidden, min_words, max_words, dimension
            )
            if not check["ok"]:
                logger.info(
                    f"[{qid}] polish failed checks ({check}); falling back to draft"
                )
                candidate_prompt = ""

        if not candidate_prompt:
            candidate_prompt = plan.get("prompt_draft") or ""
            polish_attempts += 1
            used_fallback = True

        check_final = _enforce_constraints(
            candidate_prompt, forbidden, min_words, max_words, dimension
        )

        entry = FinalPromptEntry(
            question_id=qid,
            prompt=candidate_prompt.strip(),
            length_words=check_final["word_count"],
            forbidden_hits=check_final["hits"],
            polish_attempts=polish_attempts,
            used_fallback=used_fallback,
            vlm_caption=vlm_caption,
            failed_check=check_final["failed_check"],
        )
        out.append(entry)

    logger.info(f"Finalized {len(out)} prompts")
    return out


# ============================================================
# CLI
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 finalize_prompts")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = benchmark_paths(cfg["output_dir"])
    entries = finalize_prompts(cfg)
    write_jsonl(paths["final_prompts"], [e.model_dump() for e in entries])
    logger.info(f"Wrote final prompts -> {paths['final_prompts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
