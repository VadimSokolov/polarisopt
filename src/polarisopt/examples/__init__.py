"""Bundled example study YAMLs.

Shipped inside the package so users can list and copy them with
``polarisopt examples list`` / ``polarisopt examples copy <name>``.

Each example is a complete YAML file using ``{{ env.<VAR> }}`` for any
machine-specific paths, so it runs as-is after the user sets a few
environment variables.
"""

from importlib import resources
from pathlib import Path


def list_examples() -> list[str]:
    """Return the names of bundled example study YAMLs (without extension)."""
    root = resources.files("polarisopt.examples")
    return sorted(
        p.name.removesuffix(".yaml")
        for p in root.iterdir()
        if p.name.endswith(".yaml")
    )


def example_path(name: str) -> Path:
    """Return the filesystem path to a bundled example YAML.

    Parameters
    ----------
    name : str
        Example name (no extension), e.g. ``"branin"``.

    Raises
    ------
    FileNotFoundError
        If ``<name>.yaml`` isn't bundled.
    """
    root = resources.files("polarisopt.examples")
    candidate = root / f"{name}.yaml"
    if not candidate.is_file():
        available = ", ".join(list_examples()) or "<none>"
        raise FileNotFoundError(
            f"unknown example {name!r}; available: {available}"
        )
    return Path(str(candidate))


def read_example(name: str) -> str:
    """Return the example YAML text."""
    return example_path(name).read_text()
