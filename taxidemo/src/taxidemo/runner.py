"""Slave entry point: evaluate one taxi-simulator sample from an inputs file.

Usage::

    python -m taxidemo.runner inputs.json outputs.json

``inputs.json`` schema::

    {
      "params":    {"taxi_count": 42, "base_fare": 6.5, ...},   # TaxiParams fields
      "seed":      12345,        # optional, default 0
      "n_repeats": 3             # optional, default 1
    }

The simulation is run ``n_repeats`` times with seeds ``seed, seed+1, ...``
and the per-output means are written to ``outputs.json`` (top level), with
the individual runs preserved under ``"repeats"``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from taxidemo.simulator import OUTPUT_KEYS, run_taxi_simulation


def evaluate(params: dict, seed: int = 0, n_repeats: int = 1) -> dict:
    """Run the simulator ``n_repeats`` times and average the outputs."""
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    repeats = [run_taxi_simulation(seed=seed + i, **params) for i in range(n_repeats)]
    means = {key: sum(r[key] for r in repeats) / n_repeats for key in OUTPUT_KEYS}
    return {**means, "seed": seed, "n_repeats": n_repeats, "repeats": repeats}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    input_path, output_path = Path(argv[0]), Path(argv[1])
    spec = json.loads(input_path.read_text())
    result = evaluate(
        params=spec.get("params", {}),
        seed=int(spec.get("seed", 0)),
        n_repeats=int(spec.get("n_repeats", 1)),
    )
    output_path.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
