"""Path helpers — workspace layout and POLARIS workspace conventions."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path | str) -> Path:
    """Create the directory (and parents) if it doesn't exist; return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def workspace_layout(workspace: Path | str) -> dict[str, Path]:
    """Standard subdirectories under a study workspace.

    Returns paths only — does not create them. Call ``ensure_dir`` per entry
    as needed.
    """
    root = Path(workspace).resolve()
    return {
        "root": root,
        "experiments": root / "experiments",
        "logs": root / "logs",
        "scripts": root / "scripts",
        "db": root / "polarisopt.db",
    }
