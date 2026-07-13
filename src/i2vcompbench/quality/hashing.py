"""
Hashing utilities for quality experiment reproducibility.

Provides SHA-256 helpers for files, bytes, canonical JSON,
and stable ordering keys.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Reads in 64 KiB chunks to handle large files efficiently.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hex SHA-256 string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes.

    Args:
        data: Bytes to hash.

    Returns:
        Lowercase hex SHA-256 string.
    """
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(obj: Any) -> str:
    """Compute SHA-256 of a canonical JSON serialization.

    Serialization rules:
    - Keys sorted recursively
    - No extra whitespace (separators=(',', ':'))
    - ensure_ascii=False for stable Unicode handling

    Args:
        obj: JSON-serializable Python object.

    Returns:
        Lowercase hex SHA-256 of the canonical JSON bytes.
    """
    canonical = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def stable_order_key(seed: int, question_id: str) -> str:
    """Generate a stable selection order key via SHA-256(seed || question_id).

    Args:
        seed: Integer seed for randomization.
        question_id: The question identifier.

    Returns:
        Lowercase hex SHA-256 string used for deterministic ordering.
    """
    payload = f"{seed}{question_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
