"""Smoke tests for i2vcompbench.quality.cli module."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import subprocess

import pytest

# The project src directory — needed for subprocess calls
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_DIR = str(_PROJECT_ROOT / "src")


def _run_cli(args: list[str], cwd: str = None) -> subprocess.CompletedProcess:
    """Run CLI command with PYTHONPATH set to include src/."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _SRC_DIR + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, "-m", "i2vcompbench.quality.cli"] + args,
        capture_output=True,
        text=True,
        cwd=cwd or str(_PROJECT_ROOT),
        timeout=30,
        env=env,
    )


class TestCLIHelp:
    """CLI --help works correctly."""

    def test_help_exits_zero(self):
        """python -m i2vcompbench.quality.cli --help 正常输出。"""
        result = _run_cli(["--help"])
        assert result.returncode == 0
        assert "I2V-CompBench Quality Experiments CLI" in result.stdout


class TestCLIStubCommands:
    """STUB commands return exit code 0 and print NOT_IMPLEMENTED."""

    @pytest.mark.parametrize("cmd", [
        "repair-targets",
        "prompt-variants",
        "clarity-variants",
        "cost-estimate",
        "report",
    ])
    def test_stub_command_not_implemented(self, cmd, tmp_path):
        """STUB命令返回退出码0并输出NOT_IMPLEMENTED。"""
        # Create a minimal config file for the CLI
        config = tmp_path / "quality_experiments.yaml"
        config.write_text("run:\n  seed: 42\n", encoding="utf-8")

        result = _run_cli(["--config", str(config), cmd])
        assert result.returncode == 0
        assert "NOT_IMPLEMENTED" in result.stdout


class TestCLIDryRun:
    """--dry-run mode does not produce output files."""

    def test_dry_run_audit_no_files(self, tmp_path):
        """--dry-run 模式不产生输出文件（除了框架自身的目录/manifest）。"""
        # Create minimal config with valid benchmark_root pointing to tmp
        benchmark = tmp_path / "benchmark"
        benchmark.mkdir()
        (benchmark / "phase3_manifest.jsonl").write_text("", encoding="utf-8")
        (benchmark / "input_assets_manifest.jsonl").write_text("", encoding="utf-8")
        prompts_dir = benchmark / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "final_prompts.jsonl").write_text("", encoding="utf-8")
        (benchmark / "question_plans.jsonl").write_text("", encoding="utf-8")

        output_dir = tmp_path / "output"

        config_content = (
            f"run:\n"
            f"  seed: 42\n"
            f"  output_root: {str(output_dir).replace(chr(92), '/')}\n"
            f"input:\n"
            f"  benchmark_root: {str(benchmark).replace(chr(92), '/')}\n"
        )
        config = tmp_path / "quality_experiments.yaml"
        config.write_text(config_content, encoding="utf-8")

        result = _run_cli(["--config", str(config), "audit", "--dry-run"])
        assert result.returncode == 0
        # In dry-run mode, the audit placeholder doesn't produce audit output files
        # (no candidate_quality_rows.jsonl, etc.)
        # The framework creates the run dir structure, but audit itself skips writing
        if output_dir.exists():
            # Check there's no actual audit data produced
            audit_files = list(output_dir.rglob("candidate_quality_rows.jsonl"))
            assert len(audit_files) == 0
