"""Path helpers — workspace layout and POLARIS workspace conventions."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path | str) -> Path:
    """Create the directory (and parents) if it doesn't exist; return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def workspace_layout(workspace: Path | str) -> dict[str, Path]:
    """Return the standard sub-paths under a study workspace.

    Returns paths only — does not create them. Caller mkdir's what it
    actually uses. ``StudyRunner`` only auto-creates ``experiments/``.

    Keys
    ----
    root         The workspace directory itself.
    experiments  Per-sample directories (``sim-NNNNNN``) live here.
    logs         **Reserved** for application-level master logs — empty
                 unless the user writes there. polarisopt's per-sample
                 logs are inside ``experiments/sim-NNN/`` (e.g.
                 ``polaris.stdout.log``), not here.
    scripts      **Reserved** for auxiliary scripts (e.g. the EQSQL
                 compat shim's sbatch scripts). Empty unless a backend
                 uses it.
    db           The SQLite SampleStore path.
    """
    root = Path(workspace).resolve()
    return {
        "root": root,
        "experiments": root / "experiments",
        "logs": root / "logs",
        "scripts": root / "scripts",
        "db": root / "polarisopt.db",
    }
