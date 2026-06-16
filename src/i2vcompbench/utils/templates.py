"""
Template loader for Phase 2.

Each template lives in `configs/templates/<dimension>.yaml` and contains:

  dimension: motion_binding
  subtypes:
    - id: type_a_absolute_single
      input_mode: single_image
      required_images:
        - {role: first_frame, source_preference: [tip_derived_reference, t2i_generated]}
      prompt_pattern: "The {target_subject} moves {direction}."
      t2i_prompt_pattern: "..."        # optional, used by construct_inputs when T2I
      forbidden_words: ["color", "wear"]
      camera_constraint: forbidden
      evaluator_tools: [grounding, tracking, flow]
      evaluator_E_target_pattern: "..."
      evaluator_P_constraints: ["..."]
      evaluator_C_criteria: ["..."]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger


_DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "configs" / "templates"


class DimensionTemplate:
    def __init__(self, dimension: str, data: Dict[str, Any]):
        self.dimension = dimension
        self.data = data
        self.subtypes: List[Dict[str, Any]] = data.get("subtypes", [])

    def find_subtype(self, subtype_id: Optional[str], input_mode: Optional[str] = None) -> Dict[str, Any]:
        # 1) exact match by id
        if subtype_id:
            for st in self.subtypes:
                if st.get("id") == subtype_id:
                    return st
        # 2) match by input_mode
        if input_mode:
            for st in self.subtypes:
                if st.get("input_mode") == input_mode:
                    return st
        # 3) fallback: first
        if self.subtypes:
            return self.subtypes[0]
        return {}


class TemplateRegistry:
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = Path(templates_dir) if templates_dir else _DEFAULT_TEMPLATES_DIR
        self._cache: Dict[str, DimensionTemplate] = {}

    def get(self, dimension: str) -> DimensionTemplate:
        if dimension in self._cache:
            return self._cache[dimension]
        path = self.templates_dir / f"{dimension}.yaml"
        if not path.exists():
            logger.warning(f"Template not found: {path}; using empty template")
            tpl = DimensionTemplate(dimension, {"dimension": dimension, "subtypes": []})
        else:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            tpl = DimensionTemplate(dimension, data)
        self._cache[dimension] = tpl
        return tpl


# ============================================================
# Slot rendering
# ============================================================

_SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_\.]*)\}")


def render_template(pattern: str, slots: Dict[str, Any]) -> str:
    """Replace `{key}` and `{key.subkey}` with values from slots; missing -> empty string."""
    if not pattern:
        return ""

    def _lookup(key: str) -> str:
        cur: Any = slots
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return ""
        return str(cur) if cur is not None else ""

    return _SLOT_RE.sub(lambda m: _lookup(m.group(1)), pattern)


# ============================================================
# Forbidden word detection
# ============================================================

def find_forbidden_hits(text: str, forbidden_words: List[str]) -> List[str]:
    if not text or not forbidden_words:
        return []
    low = text.lower()
    hits: List[str] = []
    for w in forbidden_words:
        if not w:
            continue
        if w.lower() in low:
            hits.append(w)
    return hits


def count_words(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])
