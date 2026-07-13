"""Unified CLI entry point for I2V-CompBench quality experiments.

Usage:
    python -m i2vcompbench.quality.cli --config configs/quality_experiments.yaml --run-id thesis_v1 <command> [options]
    python -m i2vcompbench.quality --config configs/quality_experiments.yaml audit --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from i2vcompbench.quality.config import load_quality_config
from i2vcompbench.quality.hashing import file_sha256
from i2vcompbench.quality.paths import ensure_run_dirs, run_output_dir
from i2vcompbench.quality.audit import run_audit
from i2vcompbench.quality.split import run_split
from i2vcompbench.quality.asset_manifest import build_asset_manifest
from i2vcompbench.quality.asset_lineage import build_asset_lineage
from i2vcompbench.quality.environment import run_environment_check
from i2vcompbench.quality.prompt_rules import run_prompt_rules, load_dimension_forbidden_words


# ============================================================
# Constants
# ============================================================

# Commands that have IMPL framework
IMPL_COMMANDS = frozenset([
    "audit",
    "build-asset-manifest",
    "build-asset-lineage",
    "split",
    "environment-check",
    "prompt-rules",
    "all-local",
])

# Commands that are STUB (not yet implemented)
STUB_COMMANDS = frozenset([
    "repair-targets",
    "prompt-variants",
    "prompt-metrics",
    "clarity-variants",
    "clarity-metrics",
    "aspect-variants",
    "aspect-metrics",
    "tag-subjects",
    "calibrate-difficulty",
    "orthogonal-assign",
    "orthogonal-analyze",
    "ablation-run",
    "ablation-analyze",
    "cost-estimate",
    "decision-matrix",
    "prepare-human",
    "import-human",
    "select-final",
    "report",
])

ALL_COMMANDS = sorted(IMPL_COMMANDS | STUB_COMMANDS)

# Steps executed by all-local (order matters)
ALL_LOCAL_STEPS = [
    "environment-check",
    "build-asset-manifest",
    "build-asset-lineage",
    "audit",
    "split",
    "prompt-rules",
]


# ============================================================
# Helpers
# ============================================================

def _get_code_version() -> str:
    """Retrieve current git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _load_or_create_manifest(manifest_path: Path, run_id: str, config_path: str, config_hash: str) -> Dict[str, Any]:
    """Load existing run_manifest.json or create a new one."""
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": config_path,
        "config_sha256": config_hash,
        "commands_executed": [],
        "code_version": _get_code_version(),
    }


def _save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    """Write run_manifest.json atomically."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tmp.replace(manifest_path)


def _record_command(
    manifest: Dict[str, Any],
    command: str,
    params: Dict[str, Any],
    status: str,
    started_at: str,
    completed_at: str,
    error: Optional[str] = None,
) -> None:
    """Append a command execution record to the manifest."""
    manifest["commands_executed"].append({
        "command": command,
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "params": params,
        "error": error,
    })


# ============================================================
# Command dispatch — actual implementations
# ============================================================


def _resolve_benchmark_root(config: Any) -> Path:
    """Resolve benchmark_root to an absolute path."""
    br = Path(config.input.benchmark_root)
    if not br.is_absolute():
        br = Path.cwd() / br
    return br.resolve()


def _resolve_project_root() -> Path:
    """Resolve project root (cwd)."""
    return Path.cwd().resolve()


def _run_environment_check_cmd(args: argparse.Namespace, config: Any, run_dir: Path) -> None:
    """Environment check command."""
    logger.info("Running environment-check...")
    if args.dry_run:
        logger.info("[DRY-RUN] Would run environment check. No changes made.")
        return
    benchmark_root = _resolve_benchmark_root(config)
    env_config = {
        "phase1_bundle_dir": config.input.phase1_bundle_dir,
        "model_weights": None,
    }
    run_environment_check(
        benchmark_root=benchmark_root,
        output_dir=run_dir,
        config=env_config,
    )


def _run_audit_cmd(args: argparse.Namespace, config: Any, run_dir: Path) -> None:
    """Audit command — calls run_audit."""
    logger.info("Running audit...")
    if args.dry_run:
        logger.info("[DRY-RUN] Would audit quality candidates. No changes made.")
        return
    benchmark_root = _resolve_benchmark_root(config)
    audit_dir = run_dir / "audit"
    audit_config = {
        "min_prompt_words": config.prompt.min_words,
        "max_prompt_words": config.prompt.max_words,
    }
    run_audit(
        benchmark_root=benchmark_root,
        output_dir=audit_dir,
        config=audit_config,
        limit=args.limit,
        only_qid=args.only_qid,
        resume=args.resume,
    )


def _run_build_asset_manifest_cmd(args: argparse.Namespace, config: Any, run_dir: Path) -> None:
    """Build asset manifest — calls build_asset_manifest."""
    logger.info("Running build-asset-manifest...")
    if args.dry_run:
        logger.info("[DRY-RUN] Would build source asset manifest. No changes made.")
        return
    project_root = _resolve_project_root()
    audit_dir = run_dir / "audit"
    build_asset_manifest(
        benchmark_root=project_root,
        output_dir=audit_dir,
        config=None,
        limit=args.limit,
        resume=args.resume,
    )


def _run_build_asset_lineage_cmd(args: argparse.Namespace, config: Any, run_dir: Path) -> None:
    """Build asset lineage — calls build_asset_lineage."""
    logger.info("Running build-asset-lineage...")
    if args.dry_run:
        logger.info("[DRY-RUN] Would build asset lineage manifest. No changes made.")
        return
    project_root = _resolve_project_root()
    audit_dir = run_dir / "audit"
    # Use the source_asset_manifest from previous step if available
    asset_manifest_path = audit_dir / "source_asset_manifest.jsonl"
    if not asset_manifest_path.exists():
        asset_manifest_path = None
    build_asset_lineage(
        benchmark_root=project_root,
        output_dir=audit_dir,
        asset_manifest_path=asset_manifest_path,
        config=None,
        limit=args.limit,
        resume=args.resume,
    )


def _run_split_cmd(args: argparse.Namespace, config: Any, run_dir: Path) -> None:
    """Split command — calls run_split."""
    logger.info("Running split...")
    if args.dry_run:
        logger.info("[DRY-RUN] Would split dataset into dev/val. No changes made.")
        return
    benchmark_root = _resolve_benchmark_root(config)
    split_config = {
        "seed": config.run.seed,
        "dev_per_dimension": config.split.development_per_dimension,
        "val_per_dimension": config.split.validation_per_dimension,
    }
    run_split(
        benchmark_root=benchmark_root,
        output_dir=run_dir,
        config=split_config,
        seed=config.run.seed,
        dev_per_dim=config.split.development_per_dimension,
        val_per_dim=config.split.validation_per_dimension,
    )


def _run_prompt_rules_cmd(args: argparse.Namespace, config: Any, run_dir: Path) -> None:
    """Prompt rules check — calls run_prompt_rules."""
    logger.info("Running prompt-rules...")
    if args.dry_run:
        logger.info("[DRY-RUN] Would run prompt rules check. No changes made.")
        return
    benchmark_root = _resolve_benchmark_root(config)
    manifest_path = benchmark_root / "phase3_manifest.jsonl"
    if not manifest_path.exists():
        logger.error(f"phase3_manifest.jsonl not found: {manifest_path}")
        return

    # Load candidates
    candidates: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    logger.info(f"Loaded {len(candidates)} candidates for prompt-rules check")

    # Load forbidden words
    dimension_forbidden_words = load_dimension_forbidden_words()

    # Run prompt rules
    pr_config = {
        "min_words": config.prompt.min_words,
        "max_words": config.prompt.max_words,
        "zipf_threshold": config.prompt.zipf_threshold,
    }
    report = run_prompt_rules(
        candidates=candidates,
        dimension_forbidden_words=dimension_forbidden_words,
        config=pr_config,
        limit=args.limit,
    )

    # Write report
    prompt_dir = run_dir / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    summary_path = prompt_dir / "prompt_rules_summary.json"
    summary_data = {
        "total": report["total"],
        "checked": report["checked"],
        "issues_found": report["issues_found"],
        "clean_count": report["clean_count"],
        "issue_distribution": report["issue_distribution"],
        "rare_words_found": report["rare_words_found"],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Prompt rules summary written to {summary_path}")

    # Detailed results
    results_path = prompt_dir / "prompt_rules_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in report["results"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Prompt rules details written to {results_path}")

    logger.info(
        f"Prompt rules complete: {report['checked']} checked | "
        f"clean={report['clean_count']} | issues={report['issues_found']}"
    )


# Map command name → handler function
IMPL_HANDLERS = {
    "environment-check": _run_environment_check_cmd,
    "audit": _run_audit_cmd,
    "build-asset-manifest": _run_build_asset_manifest_cmd,
    "build-asset-lineage": _run_build_asset_lineage_cmd,
    "split": _run_split_cmd,
    "prompt-rules": _run_prompt_rules_cmd,
}


# ============================================================
# Core execution logic
# ============================================================

def _execute_impl_command(
    command: str,
    args: argparse.Namespace,
    config: Any,
    run_dir: Path,
    manifest: Dict[str, Any],
    manifest_path: Path,
) -> bool:
    """Execute an IMPL command with full framework wrappers.

    Returns True on success, False on failure.
    """
    params = {
        "limit": args.limit,
        "resume": args.resume,
        "dry_run": args.dry_run,
        "only_qid": args.only_qid,
        "allow_api": args.allow_api,
        "max_api_calls": args.max_api_calls,
    }
    started_at = datetime.now().isoformat(timespec="seconds")
    status = "success"
    error_msg = None

    try:
        handler = IMPL_HANDLERS[command]
        handler(args, config, run_dir)
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        if args.log_level == "DEBUG":
            logger.exception(f"Command '{command}' failed")
        else:
            logger.error(f"Command '{command}' failed: {exc}")
        return False
    finally:
        completed_at = datetime.now().isoformat(timespec="seconds")
        _record_command(manifest, command, params, status, started_at, completed_at, error_msg)
        _save_manifest(manifest_path, manifest)

    return True


def _run_all_local(args: argparse.Namespace, config: Any, run_dir: Path, manifest: Dict[str, Any], manifest_path: Path) -> None:
    """Execute all local (non-API) steps sequentially."""
    if args.allow_api:
        logger.warning("all-local ignores --allow-api; no network calls will be made.")

    # Force no API for all-local
    args.allow_api = False

    results: List[Dict[str, Any]] = []
    for step in ALL_LOCAL_STEPS:
        logger.info(f"{'=' * 40}")
        logger.info(f"all-local: executing step '{step}'")
        success = _execute_impl_command(step, args, config, run_dir, manifest, manifest_path)
        results.append({"step": step, "success": success})

    # Summary
    logger.info(f"{'=' * 40}")
    logger.info("all-local summary:")
    failed = [r for r in results if not r["success"]]
    for r in results:
        icon = "OK" if r["success"] else "FAIL"
        logger.info(f"  [{icon}] {r['step']}")
    if failed:
        logger.warning(f"{len(failed)} step(s) failed. Check logs above.")


# ============================================================
# Argument parsing
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="i2vcompbench.quality.cli",
        description="I2V-CompBench Quality Experiments CLI",
    )

    # Global arguments (before subcommand)
    parser.add_argument("--config", required=True, help="Path to quality_experiments.yaml config file")
    parser.add_argument("--run-id", default=None, help="Run identifier (auto-generated if omitted)")
    parser.add_argument("--log-level", choices=["INFO", "DEBUG"], default="INFO", help="Logging level")

    # Shared arguments parent (inherited by all subcommands)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--resume", action="store_true", default=False, help="Skip already-completed records")
    shared.add_argument("--limit", type=int, default=None, help="Only process N records")
    shared.add_argument("--only-qid", default=None, help="Only process a specific question_id")
    shared.add_argument("--dry-run", action="store_true", default=False, help="Report planned operations without executing")
    shared.add_argument("--allow-api", action="store_true", default=False, help="Allow API calls (default: forbidden)")
    shared.add_argument("--max-api-calls", type=int, default=None, help="Maximum number of API calls")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Register all commands with shared parent
    for cmd in ALL_COMMANDS:
        sub = subparsers.add_parser(
            cmd,
            parents=[shared],
            help=f"{'[IMPL]' if cmd in IMPL_COMMANDS else '[STUB]'} {cmd}",
        )
        # Extra params for select-final
        if cmd == "select-final":
            sub.add_argument(
                "--stage",
                choices=["provisional", "commit"],
                default="provisional",
                help="Selection stage: 'commit' fails fast without human review",
            )

    return parser


# ============================================================
# Main entry point
# ============================================================

def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate command was provided
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging
    logger.remove()
    log_format = (
        "<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    logger.add(sys.stderr, level=args.log_level, format=log_format)

    # Handle STUB commands early
    if args.command in STUB_COMMANDS:
        print(f"[NOT_IMPLEMENTED] Command '{args.command}' is not yet implemented. Planned for future task.")
        sys.exit(0)

    # --- IMPL command path ---
    try:
        # Build overrides from CLI
        overrides: Dict[str, Any] = {}
        if args.run_id:
            overrides["run"] = {"run_id": args.run_id}

        # Load configuration
        config = load_quality_config(config_path=args.config, overrides=overrides)
        run_id = config.run.run_id

        # Create run directory structure
        run_dir = run_output_dir(config.run.output_root, run_id)
        ensure_run_dirs(run_dir)

        # Resolve config file hash
        config_path_resolved = Path(args.config)
        if not config_path_resolved.is_absolute():
            config_path_resolved = Path.cwd() / config_path_resolved
        config_hash = file_sha256(config_path_resolved)

        # Load/create manifest
        manifest_path = run_dir / "run_manifest.json"
        manifest = _load_or_create_manifest(manifest_path, run_id, args.config, config_hash)

        # Dispatch
        if args.command == "all-local":
            _run_all_local(args, config, run_dir, manifest, manifest_path)
        else:
            success = _execute_impl_command(args.command, args, config, run_dir, manifest, manifest_path)
            if not success:
                sys.exit(1)

    except FileNotFoundError as exc:
        logger.error(f"File not found: {exc}")
        sys.exit(1)
    except ValueError as exc:
        logger.error(f"Configuration error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Unexpected error")
        else:
            logger.error(f"Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
