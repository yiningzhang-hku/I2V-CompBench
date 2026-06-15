"""
IO helpers for Phase 2.

Includes:
- config loader (yaml)
- jsonl read/write
- phase1_bundle loader (manifest_clean / image_parse / text_parse / aligned_instances /
  reference_bank/assets / candidate_recipes / prior_package / compatibility_matrix)
- output dir helpers
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import yaml
from loguru import logger


# ============================================================
# Phase 1 path bootstrap (kept for backward compatibility)
# ============================================================

def bootstrap_phase1_path(repo_root: Optional[Path] = None) -> None:
    """No-op shim retained for backward compatibility.

    Phase 1 modules now live inside the i2vcompbench package, so importing them
    no longer requires manipulating sys.path. This function is kept so legacy
    callers do not break.
    """
    return None


# ============================================================
# Config
# ============================================================

def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


# ============================================================
# JSON / JSONL
# ============================================================

def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skip malformed JSONL line in {path}: {e}")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")


# ============================================================
# Output directory layout
# ============================================================

def benchmark_paths(output_dir: str | Path) -> Dict[str, Path]:
    """Return canonical sub-paths under benchmark_dataset/."""
    root = Path(output_dir)
    paths = {
        "root": root,
        "first_frames": root / "first_frames",
        "ref_images": root / "ref_images",
        "samples": root / "samples",
        "prompts": root / "prompts",
        "qc_reports": root / "qc_reports",
        "quota_plan": root / "quota_plan.json",
        "sampled_recipes": root / "sampled_recipes.jsonl",
        "quota_unfilled_report": root / "quota_unfilled_report.json",
        "question_plans": root / "question_plans.jsonl",
        "input_assets_manifest": root / "input_assets_manifest.jsonl",
        "qc_failed_to_retry": root / "qc_failed_to_retry.jsonl",
        "manual_review_queue": root / "manual_review_queue.jsonl",
        "final_prompts": root / "prompts" / "final_prompts.jsonl",
        "phase3_manifest": root / "phase3_manifest.jsonl",
        "contrastive_pairs": root / "contrastive_pairs.jsonl",
        "dataset_card": root / "dataset_card.md",
    }
    for key in ("root", "first_frames", "ref_images", "samples", "prompts", "qc_reports"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


# ============================================================
# Phase 1 bundle loader
# ============================================================

class Phase1Bundle:
    """Lazy-loading wrapper around phase1_bundle/ directory."""

    def __init__(self, bundle_dir: str | Path):
        self.dir = Path(bundle_dir)
        if not self.dir.exists():
            raise FileNotFoundError(f"phase1_bundle_dir not found: {bundle_dir}")
        self._manifest: Optional[Dict[str, Dict[str, Any]]] = None
        self._image_parse: Optional[Dict[str, Dict[str, Any]]] = None
        self._text_parse: Optional[Dict[str, Dict[str, Any]]] = None
        self._aligned: Optional[Dict[str, Dict[str, Any]]] = None
        self._assets: Optional[Dict[str, Dict[str, Any]]] = None
        self._recipes: Optional[List[Dict[str, Any]]] = None
        self._prior_package: Optional[Dict[str, Any]] = None
        self._compat_matrix: Optional[Dict[str, Any]] = None

    # ---- lookups ----
    def _candidate_files(self, names: List[str]) -> Optional[Path]:
        for n in names:
            p = self.dir / n
            if p.exists():
                return p
        # also try a nested phase1_bundle/ directory
        nested = self.dir / "phase1_bundle"
        if nested.exists():
            for n in names:
                p = nested / n
                if p.exists():
                    return p
        return None

    # ---- properties ----
    @property
    def manifest(self) -> Dict[str, Dict[str, Any]]:
        if self._manifest is None:
            p = self._candidate_files(["manifest_clean.jsonl", "manifest.jsonl"])
            if p is None:
                logger.warning("manifest_clean.jsonl not found in bundle; using empty dict")
                self._manifest = {}
            else:
                self._manifest = {row["sample_id"]: row for row in iter_jsonl(p) if "sample_id" in row}
                logger.info(f"Loaded {len(self._manifest)} manifest rows from {p}")
        return self._manifest

    @property
    def image_parse(self) -> Dict[str, Dict[str, Any]]:
        if self._image_parse is None:
            p = self._candidate_files(["image_parse.jsonl"])
            if p is None:
                self._image_parse = {}
            else:
                self._image_parse = {row["sample_id"]: row for row in iter_jsonl(p) if "sample_id" in row}
                logger.info(f"Loaded {len(self._image_parse)} image_parse rows from {p}")
        return self._image_parse

    @property
    def text_parse(self) -> Dict[str, Dict[str, Any]]:
        if self._text_parse is None:
            p = self._candidate_files(["text_parse.jsonl"])
            if p is None:
                self._text_parse = {}
            else:
                self._text_parse = {row["sample_id"]: row for row in iter_jsonl(p) if "sample_id" in row}
                logger.info(f"Loaded {len(self._text_parse)} text_parse rows from {p}")
        return self._text_parse

    @property
    def aligned(self) -> Dict[str, Dict[str, Any]]:
        if self._aligned is None:
            p = self._candidate_files(["aligned_instances.jsonl"])
            if p is None:
                self._aligned = {}
            else:
                self._aligned = {row["sample_id"]: row for row in iter_jsonl(p) if "sample_id" in row}
                logger.info(f"Loaded {len(self._aligned)} aligned_instances rows from {p}")
        return self._aligned

    @property
    def assets(self) -> Dict[str, Dict[str, Any]]:
        """reference_bank/assets.jsonl indexed by asset_id."""
        if self._assets is None:
            p = self._candidate_files([
                "reference_bank/assets.jsonl",
                "assets.jsonl",
            ])
            if p is None:
                self._assets = {}
            else:
                self._assets = {row["asset_id"]: row for row in iter_jsonl(p) if "asset_id" in row}
                logger.info(f"Loaded {len(self._assets)} reference assets from {p}")
        return self._assets

    @property
    def recipes(self) -> List[Dict[str, Any]]:
        if self._recipes is None:
            p = self._candidate_files(["candidate_recipes.jsonl"])
            if p is None:
                self._recipes = []
            else:
                self._recipes = read_jsonl(p)
                logger.info(f"Loaded {len(self._recipes)} candidate recipes from {p}")
        return self._recipes

    @property
    def prior_package(self) -> Dict[str, Any]:
        if self._prior_package is None:
            p = self._candidate_files(["prior_package.json"])
            self._prior_package = read_json(p) if p else {}
        return self._prior_package

    @property
    def compatibility_matrix(self) -> Dict[str, Any]:
        if self._compat_matrix is None:
            p = self._candidate_files(["compatibility_matrix.json"])
            self._compat_matrix = read_json(p) if p else {}
        return self._compat_matrix

    # ---- helpers ----
    def get_image_path(self, sample_id: str) -> Optional[str]:
        """
        Resolve absolute path to TIP first frame for a given sample_id.
        Tries manifest.image_path; falls back to <bundle>/images/<sample_id>.jpg.
        """
        row = self.manifest.get(sample_id)
        if row:
            ip = row.get("image_path") or row.get("local_image_path")
            if ip:
                p = Path(ip)
                if not p.is_absolute():
                    p = self.dir / ip
                if p.exists():
                    return str(p)
        # fallback by convention
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = self.dir / "images" / f"{sample_id}{ext}"
            if p.exists():
                return str(p)
        return None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = self.dir / "images" / f"{sample_id}{ext}"
            if p.exists():
                return str(p)
        return None
