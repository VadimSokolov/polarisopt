"""Construct a Runner from a YAML config block.

Kept separate from :mod:`polarisopt.runners.base` so the registry doesn't
pull every concrete runner into its module-level imports.
"""

from __future__ import annotations

from typing import Any

from polarisopt.runners.base import Runner, runner_registry


def make_runner(spec: dict[str, Any]) -> Runner:
    """Build a Runner from ``{"type": "...", "options": {...}}``."""
    if "type" not in spec:
        raise ValueError(f"runner spec missing 'type': {spec!r}")
    cls = runner_registry.get(spec["type"])
    options = spec.get("options", {}) or {}
    # SlurmResources lives inside the slurm runner; allow nested "default_resources" dict
    if spec["type"] == "slurm" and "default_resources" in options:
        from polarisopt.runners.slurm import SlurmResources

        if isinstance(options["default_resources"], dict):
            options = {**options, "default_resources": SlurmResources(**options["default_resources"])}
    return cls(**options)
