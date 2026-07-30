"""Pydantic models for the study YAML.

The schema is permissive at the leaves: each plugin section (design,
surrogate, acquisition, runner, ...) is an arbitrary dict that the
plugin class validates itself when instantiated. This avoids re-declaring
every plugin's parameters here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunnerConfig(_Base):
    """How simulations are submitted (local, slurm, ...)."""

    type: str
    options: dict[str, Any] = Field(default_factory=dict)


class SimulatorConfig(_Base):
    """Which simulator implementation to use (polaris, mock, ...)."""

    type: str
    options: dict[str, Any] = Field(default_factory=dict)


class ParametersConfig(_Base):
    """How parameters are declared. Either an external file or inline records."""

    source: Path | None = None
    inline: list[dict[str, Any]] | None = None

    @field_validator("source", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path | None:
        return Path(v) if v is not None else None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ParametersConfig:
        if (self.source is None) == (self.inline is None):
            raise ValueError("parameters: provide exactly one of 'source' or 'inline'")
        return self


class MetricConfig(_Base):
    """How simulation outputs become a scalar/vector metric."""

    type: str
    options: dict[str, Any] = Field(default_factory=dict)


class DesignConfig(_Base):
    """Static DOE: which design + its options (e.g. n, num_levels)."""

    type: str
    options: dict[str, Any] = Field(default_factory=dict)


class StoppingConfig(_Base):
    """Stopping criterion or combinator."""

    type: str
    options: dict[str, Any] = Field(default_factory=dict)
    criteria: list[StoppingConfig] | None = None


StoppingConfig.model_rebuild()


class StaticPhaseConfig(_Base):
    """One static-DOE phase: name + design."""

    name: str
    type: Literal["static"]
    design: DesignConfig


class SequentialPhaseConfig(_Base):
    """One sequential-DOE phase: warm-up design + generator + stop.

    The generator owns its own surrogate/acquisition sub-specs — see
    :class:`polarisopt.generators.acquisition.AcquisitionGenerator`.
    """

    name: str
    type: Literal["sequential"]
    warm_up: DesignConfig | None = None
    generator: dict[str, Any]
    batch_size: int = 1
    stop: StoppingConfig
    minimize: bool = True


class HistoryMatchingPhaseConfig(_Base):
    """One history-matching wave: warm-up (LHS) → per-moment GPs →
    implausibility filter → NROY.

    v0.31 P3. Vernon-Goldstein-Bower 2010 methodology. Consumes a
    ``moment_set`` metric via the study's shared metric config.
    """

    name: str
    type: Literal["history_matching"]
    warm_up: DesignConfig
    # Emulator: currently "gp_per_moment" (default) is the only shipped
    # option. Extensions (deep-learning dim-reduction, single multi-output
    # GP) may follow — the dict lets the caller pass extras verbatim.
    emulator: dict[str, Any] = Field(
        default_factory=lambda: {"type": "gp_per_moment", "options": {}}
    )
    # Implausibility spec: {type: max|second_max, cutoff: float,
    # include_prior_terms: bool}
    implausibility: dict[str, Any] = Field(
        default_factory=lambda: {"type": "max", "cutoff": 3.0, "include_prior_terms": True}
    )
    # Subset of moment names to include this wave (empty = all).
    moments_included: list[str] = Field(default_factory=list)
    # NROY dense-grid size (Sobol) used to prune the retained region.
    nroy_grid_size: int = 8192
    # Wall path — where to write nroy_wave{N}.parquet artifacts.
    # Default: <workspace>/history_matching/
    output_dir: str | None = None


PhaseConfig = StaticPhaseConfig | SequentialPhaseConfig | HistoryMatchingPhaseConfig


class StudyConfig(_Base):
    """Top-level study config — what the YAML deserializes into."""

    name: str
    workspace: Path
    seed: int | None = None
    simulator: SimulatorConfig
    runner: RunnerConfig
    parameters: ParametersConfig
    metric: MetricConfig
    phases: list[PhaseConfig] = Field(..., min_length=1)

    @field_validator("workspace", mode="before")
    @classmethod
    def _coerce_workspace(cls, v: Any) -> Path:
        return Path(v).expanduser()
