"""Canonical Sample dataclass — one shape across the whole library."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from polarisopt.utils._compat import StrEnum


class SampleStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Sample:
    """A single evaluation of the simulator at one point in parameter space.

    Created in PENDING state by a Design or SampleGenerator. The Study moves
    it through RUNNING → FINISHED/FAILED/CANCELLED, populating ``folder``,
    ``runner_task_id``, ``metric``, and ``runtime_s`` as work progresses.

    ``metric`` is always a 1-D array, length 1 for single-objective and >1
    for multi-objective studies. ``None`` until the sample finishes.
    """

    id: int | None = None
    phase: str = ""
    iteration: int = 0
    inputs: np.ndarray = field(default_factory=lambda: np.empty(0))
    status: SampleStatus = SampleStatus.PENDING
    metric: np.ndarray | None = None
    folder: Path | None = None
    runtime_s: float | None = None
    runner_task_id: str | None = None
    message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, np.ndarray):
            self.inputs = np.asarray(self.inputs, dtype=float)
        if self.metric is not None and not isinstance(self.metric, np.ndarray):
            self.metric = np.asarray(self.metric, dtype=float)
        if isinstance(self.status, str):
            self.status = SampleStatus(self.status)
        if self.folder is not None and not isinstance(self.folder, Path):
            self.folder = Path(self.folder)

    def is_terminal(self) -> bool:
        return self.status in {SampleStatus.FINISHED, SampleStatus.FAILED, SampleStatus.CANCELLED}
