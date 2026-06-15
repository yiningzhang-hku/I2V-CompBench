"""ID generators for Phase 2."""

from __future__ import annotations

import hashlib
from typing import Dict


# 计数器（按 dim_short + mode 维护单调序号）
_COUNTERS: Dict[str, int] = {}


def reset_counters() -> None:
    _COUNTERS.clear()


def next_question_id(dim_short: str, mode_short: str) -> str:
    """
    Generate `{dim_short}_{mode_short}_{seq:04d}` style id.
    Example: attr_multi_0001 / motion_B_0042.
    """
    key = f"{dim_short}_{mode_short}"
    _COUNTERS[key] = _COUNTERS.get(key, 0) + 1
    return f"{key}_{_COUNTERS[key]:04d}"


def asset_id_for(question_id: str, role: str, idx: int = 0) -> str:
    return f"{question_id}__{role}__{idx:02d}"


def stable_pair_id(recipe_id_a: str, recipe_id_b: str) -> str:
    """Deterministic id for a contrastive pair (order-independent)."""
    a, b = sorted([recipe_id_a, recipe_id_b])
    h = hashlib.md5(f"{a}|{b}".encode("utf-8")).hexdigest()[:8]
    return f"pair_{h}"
