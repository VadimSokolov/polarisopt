"""Inject calibration values into POLARIS JSON files.

POLARIS configuration JSONs are two-level dicts: top-level categories like
``"General simulation controls"`` containing flat parameter dicts. This
module walks the second level looking for each parameter name and replaces
the value in place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from polarisopt.parameters.space import (
    Parameter,
    ParameterSpace,
    parameter_space_from_records,
)
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)


def load_parameter_file(path: Path | str) -> ParameterSpace:
    """Load a ``ParameterSpace`` from a YAML or JSON parameter spec.

    The file format is a flat list of records::

        - { name: trip_threshold, file: ActivityChoice.json, min: 0.1, max: 0.9, type: float }
        - { name: top_k,          file: DestinationChoice.json, min: 1,  max: 20, type: int   }
    """
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        records = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        records = json.loads(text)
    else:
        raise ValueError(f"Unsupported parameter-file extension: {p.suffix}")
    if not isinstance(records, list):
        raise ValueError(f"Parameter file {p} must contain a list of records, got {type(records).__name__}")
    return parameter_space_from_records(records)


def _set_in_two_level_dict(d: dict[str, Any], key: str, value: Any) -> bool:
    """Walk the second level of a POLARIS-style dict; set ``key`` if found."""
    found = False
    for category in d.values():
        if isinstance(category, dict) and key in category:
            category[key] = value
            found = True
    return found


def inject_values(
    sample: np.ndarray,
    space: ParameterSpace,
    target_dir: Path | str,
    *,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Inject a sample vector into POLARIS JSONs under ``target_dir``.

    For each parameter, the corresponding JSON file is opened, the value is
    placed (respecting two-level POLARIS structure), and the file is rewritten.

    Returns a dict of file → list of parameter names that were *not* found
    in that file. An empty dict means every parameter was placed.
    """
    if sample.shape != (space.ndim,):
        raise ValueError(f"sample must have shape ({space.ndim},), got {sample.shape}")
    target_dir = Path(target_dir)
    clipped = space.clip(sample)
    by_file: dict[str, list[Parameter]] = space.by_file()

    missing: dict[str, list[str]] = {}
    for relpath, params in by_file.items():
        json_path = target_dir / relpath
        if not json_path.exists():
            log.warning("Parameter target %s not found under %s; skipping", relpath, target_dir)
            missing[relpath] = [p.name for p in params]
            continue
        data = json.loads(json_path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"POLARIS config {json_path} must be a JSON object at top level")
        file_missing: list[str] = []
        for p in params:
            idx = space.parameters.index(p)
            value = clipped[idx]
            if not _set_in_two_level_dict(data, p.name, _native(value)):
                file_missing.append(p.name)
        if file_missing:
            missing[relpath] = file_missing
        if dry_run:
            continue
        json_path.write_text(json.dumps(data, indent=4))
    return missing


def _native(value: Any) -> Any:
    """Convert numpy scalars to plain Python for JSON-safe serialization."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
