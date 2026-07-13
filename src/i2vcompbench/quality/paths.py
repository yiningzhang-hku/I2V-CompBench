"""
Path utilities for quality experiment outputs.

All path outputs use POSIX format (forward slashes).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

# Standard subdirectories created for each run
_RUN_SUBDIRS = (
    "audit",
    "splits",
    "prompt",
    "clarity",
    "aspect",
    "subject",
    "difficulty",
    "annotation",
    "selection",
    "lineage",
)


def to_posix(path: str) -> str:
    """Convert a Windows backslash path to POSIX forward-slash format."""
    return path.replace("\\", "/")


def resolve_image_path(base_dir: Path, relative_path: str) -> Path:
    """Resolve a POSIX relative image path against a base directory.

    Args:
        base_dir: The root directory to resolve against.
        relative_path: A POSIX-style relative path string.

    Returns:
        Resolved absolute Path object.
    """
    # Normalize to POSIX first, then resolve
    normalized = to_posix(relative_path)
    return (base_dir / PurePosixPath(normalized)).resolve()


def run_output_dir(output_root: str, run_id: str) -> Path:
    """Generate the output directory path for a specific run.

    Args:
        output_root: Root output directory (may be Windows or POSIX).
        run_id: Unique identifier for the run.

    Returns:
        Path object for the run directory.
    """
    return Path(output_root) / run_id


def ensure_run_dirs(run_dir: Path) -> dict[str, Path]:
    """Create standard subdirectories under a run directory.

    Creates: audit/, splits/, prompt/, clarity/, aspect/,
             subject/, difficulty/, annotation/, selection/, lineage/

    Args:
        run_dir: The root directory for this run.

    Returns:
        Mapping from subdirectory name to its Path.
    """
    result: dict[str, Path] = {}
    for subdir_name in _RUN_SUBDIRS:
        subdir = run_dir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        result[subdir_name] = subdir
    return result
