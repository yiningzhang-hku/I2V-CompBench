"""兼容wrapper — 实际逻辑已迁至 src/i2vcompbench/quality/audit.py

This script preserves the original CLI interface while delegating to the
new quality audit module. It can still be invoked directly:

    python scripts/audit_final_candidates.py --root data/benchmark_dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the import path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from i2vcompbench.quality.audit import run_audit_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_audit_cli())
