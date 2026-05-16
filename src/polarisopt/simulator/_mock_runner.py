"""CLI helper invoked by :class:`MockSimulator` jobs — runs in the *slave* process.

Usage::

    python -m polarisopt.simulator._mock_runner <function_name> <input.json> <output.json>

The input JSON is ``{"inputs": [x0, x1, ...]}``; the output JSON is
``{"function": "<name>", "inputs": [...], "value": <float>}``.

Kept separate from the master code so the orchestrator never imports the
benchmark functions or runs them in-process — same boundary POLARIS uses.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from polarisopt.simulator.benchmarks import BENCHMARKS


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 3:
        sys.stderr.write(
            "usage: python -m polarisopt.simulator._mock_runner <fn> <input.json> <output.json>\n"
        )
        return 2

    fn_name, input_path, output_path = args
    if fn_name not in BENCHMARKS:
        sys.stderr.write(f"unknown benchmark function: {fn_name!r}\n")
        return 2

    payload = json.loads(Path(input_path).read_text())
    x = np.asarray(payload["inputs"], dtype=float)

    t0 = time.perf_counter()
    value = BENCHMARKS[fn_name](x)
    elapsed = time.perf_counter() - t0

    Path(output_path).write_text(
        json.dumps(
            {
                "function": fn_name,
                "inputs": x.tolist(),
                "value": float(value),
                "runtime_s": elapsed,
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
