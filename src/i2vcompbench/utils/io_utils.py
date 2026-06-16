"""
I/O utility functions for reading/writing JSONL, CSV, and YAML files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import jsonlines
import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def read_jsonl(file_path: str, model: Optional[Type[T]] = None) -> List[Any]:
    """
    Read JSONL file. If model is provided, parse each line into that Pydantic model.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"File not found: {file_path}, returning empty list")
        return []

    results = []
    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            if model is not None:
                try:
                    results.append(model.model_validate(obj))
                except Exception as e:
                    logger.warning(f"Failed to parse line into {model.__name__}: {e}")
                    results.append(obj)
            else:
                results.append(obj)
    logger.info(f"Read {len(results)} items from {file_path}")
    return results


def write_jsonl(file_path: str, items: List[Any], append: bool = False) -> None:
    """
    Write list of items to JSONL file. Items can be dicts or Pydantic models.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with jsonlines.open(path, mode=mode) as writer:
        for item in items:
            if isinstance(item, BaseModel):
                writer.write(item.model_dump())
            else:
                writer.write(item)
    logger.info(f"Wrote {len(items)} items to {file_path} (append={append})")


def append_jsonl(file_path: str, item: Any) -> None:
    """Append a single item to JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(path, mode="a") as writer:
        if isinstance(item, BaseModel):
            writer.write(item.model_dump())
        else:
            writer.write(item)


def read_processed_ids(file_path: str) -> set:
    """Read already-processed sample IDs from a JSONL file for resume support."""
    path = Path(file_path)
    if not path.exists():
        return set()

    ids = set()
    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            if "sample_id" in obj:
                ids.add(obj["sample_id"])
    logger.info(f"Found {len(ids)} already-processed IDs in {file_path}")
    return ids


def write_csv(file_path: str, data: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> None:
    """Write list of dicts to CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)
    if columns:
        df = df[columns]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"Wrote CSV with {len(df)} rows to {file_path}")


def write_freq_csv(file_path: str, freq_dict: Dict[str, int], key_col: str = "item") -> None:
    """
    Write a frequency dict to CSV with frequency and percentage columns.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total = sum(freq_dict.values())
    rows = []
    for k, v in sorted(freq_dict.items(), key=lambda x: -x[1]):
        rows.append({
            key_col: k,
            "frequency": v,
            "percentage": round(v / total * 100, 2) if total > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info(f"Wrote freq CSV with {len(df)} rows to {file_path}")


def ensure_dir(dir_path: str) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_prompt_template(prompt_path: str) -> str:
    """Load a prompt template from text file."""
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    text = path.read_text(encoding="utf-8").strip()
    logger.info(f"Loaded prompt template from {prompt_path} ({len(text)} chars)")
    return text


def clean_pika_prompt(raw_prompt: str) -> dict:
    """
    Clean Pika-specific parameters from TIP-I2V text prompts.
    
    Pika prompts contain parameters like:
      -motion 3, -camera zoom in, -neg "...", -fps 24, -gs 12, -ar 16:9, -seed 1234
    
    Returns dict with:
      - clean_text: prompt with all Pika params removed
      - pika_camera: extracted camera command (if any)
      - pika_motion: extracted motion level (if any)
      - pika_neg: extracted negative prompt (if any)
      - pika_params: dict of other extracted params
    """
    import re

    text = raw_prompt.strip()
    pika_camera = None
    pika_motion = None
    pika_neg = None
    pika_params = {}

    # Extract -neg "..." or -neg '...' (quoted negative prompt)
    neg_match = re.search(
        r'-neg\s+["\u201c\u201d\'](.*?)["\u201c\u201d\']',
        text, re.DOTALL | re.IGNORECASE
    )
    if neg_match:
        pika_neg = neg_match.group(1).strip()
        text = text[:neg_match.start()] + text[neg_match.end():]

    # Extract -neg without quotes (until next - or end of string)
    neg_match2 = re.search(
        r'-neg\s+([^-]+)',
        text, re.IGNORECASE
    )
    if neg_match2:
        pika_neg = (pika_neg or "") + neg_match2.group(1).strip()
        text = text[:neg_match2.start()] + text[neg_match2.end():]

    # Extract -camera <command> (e.g., -camera zoom in, -camera pan left)
    cam_match = re.search(
        r'-camera\s+(zoom\s+(?:in|out)|pan\s+(?:left|right)|tilt\s+(?:up|down)|static|rotate\s*\w*)',
        text, re.IGNORECASE
    )
    if cam_match:
        pika_camera = cam_match.group(1).strip().lower()
        text = text[:cam_match.start()] + text[cam_match.end():]

    # Extract -motion <number>
    motion_match = re.search(r'-motion\s+(\d+)', text, re.IGNORECASE)
    if motion_match:
        pika_motion = int(motion_match.group(1))
        text = text[:motion_match.start()] + text[motion_match.end():]

    # Extract other known Pika params: -fps, -gs, -ar, -seed, -zoom
    for param in ["fps", "gs", "ar", "seed"]:
        param_match = re.search(rf'-{param}\s+(\S+)', text, re.IGNORECASE)
        if param_match:
            pika_params[param] = param_match.group(1)
            text = text[:param_match.start()] + text[param_match.end():]

    # Also handle standalone -zoom in / -zoom out (without -camera prefix)
    zoom_match = re.search(r'-zoom\s+(in|out)', text, re.IGNORECASE)
    if zoom_match:
        pika_camera = pika_camera or f"zoom {zoom_match.group(1).lower()}"
        text = text[:zoom_match.start()] + text[zoom_match.end():]

    # Clean up: remove leftover dashes with no content, multiple spaces, leading/trailing
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove leading/trailing punctuation artifacts
    text = text.strip(' ,-_.')

    return {
        "clean_text": text,
        "pika_camera": pika_camera,
        "pika_motion": pika_motion,
        "pika_neg": pika_neg,
        "pika_params": pika_params,
    }


def parse_json_from_text(text: str) -> Optional[dict]:
    """
    Try to parse JSON from LLM/VLM output text.
    Strategy: direct parse -> extract ```json...``` block -> extract {...} block
    """
    text = text.strip()

    # Strip <think>...</think> blocks from thinking models (e.g. Qwen3.5)
    import re as _re
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract ```json ... ``` block
    import re
    json_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract outermost { ... }
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None
