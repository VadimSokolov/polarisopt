"""SampleStore — SQLAlchemy-backed persistence for Samples and phase state.

This is the single source of truth for a study's evaluation history.
SQLite with WAL mode by default; the SQLAlchemy abstraction allows swapping
to Postgres if a deployment ever needs >100 concurrent writers.

Tables
------
- ``studies(id, name, config_yaml, created_at)``
- ``samples(...)`` — one row per Sample
- ``phase_state(study_id, phase, iteration, rng_state, surrogate_state)`` — for restart
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    select,
)
from sqlalchemy.engine import Engine

from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.utils.logging import get_logger

log = get_logger(__name__)

_metadata = MetaData()

studies_table = Table(
    "studies",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, unique=True, nullable=False),
    Column("config_yaml", Text, nullable=True),
    Column("created_at", DateTime, nullable=False),
)

samples_table = Table(
    "samples",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("study_id", Integer, ForeignKey("studies.id"), nullable=False),
    Column("phase", String, nullable=False),
    Column("iteration", Integer, nullable=False, default=0),
    Column("inputs_json", Text, nullable=False),
    Column("status", String, nullable=False),
    Column("metric_json", Text, nullable=True),
    Column("folder", String, nullable=True),
    Column("runtime_s", Float, nullable=True),
    Column("runner_task_id", String, nullable=True),
    Column("message", Text, nullable=True),
    Column("extra_json", Text, nullable=True),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
)

phase_state_table = Table(
    "phase_state",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("study_id", Integer, ForeignKey("studies.id"), nullable=False),
    Column("phase", String, nullable=False),
    Column("iteration", Integer, nullable=False),
    Column("rng_state", LargeBinary, nullable=True),
    Column("surrogate_state", LargeBinary, nullable=True),
    Column("updated_at", DateTime, nullable=False),
)


@event.listens_for(Engine, "connect")
def _enable_sqlite_wal(dbapi_connection, _connection_record):  # type: ignore[no-untyped-def]
    """Apply WAL + foreign-key pragmas on every fresh SQLite connection."""
    try:
        import sqlite3
    except ImportError:  # pragma: no cover
        return
    if not isinstance(dbapi_connection, sqlite3.Connection):
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.close()


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class SampleStore:
    """Persistence layer for a study.

    Open with :meth:`open` (or :meth:`open_memory` for tests). Each study has a
    unique name; reopening with the same name attaches to the existing record.
    """

    def __init__(self, engine: Engine, study_id: int, study_name: str) -> None:
        self._engine = engine
        self._study_id = study_id
        self._study_name = study_name

    # ---------- construction ----------

    @classmethod
    def open(cls, db_path: Path | str, study_name: str, *, config_yaml: str | None = None) -> SampleStore:
        """Open or create a SQLite-backed store at ``db_path`` for ``study_name``."""
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{path}", future=True)
        return cls._bind(engine, study_name, config_yaml=config_yaml)

    @classmethod
    def open_memory(cls, study_name: str = "test", *, config_yaml: str | None = None) -> SampleStore:
        """In-memory store for tests."""
        engine = create_engine("sqlite:///:memory:", future=True)
        return cls._bind(engine, study_name, config_yaml=config_yaml)

    @classmethod
    def _bind(cls, engine: Engine, study_name: str, *, config_yaml: str | None) -> SampleStore:
        _metadata.create_all(engine)
        with engine.begin() as conn:
            existing = conn.execute(
                select(studies_table.c.id).where(studies_table.c.name == study_name)
            ).scalar_one_or_none()
            if existing is None:
                result = conn.execute(
                    studies_table.insert().values(
                        name=study_name,
                        config_yaml=config_yaml,
                        created_at=_now(),
                    )
                )
                study_id = int(result.inserted_primary_key[0])
            else:
                study_id = int(existing)
                if config_yaml is not None:
                    conn.execute(
                        studies_table.update()
                        .where(studies_table.c.id == study_id)
                        .values(config_yaml=config_yaml)
                    )
        return cls(engine=engine, study_id=study_id, study_name=study_name)

    # ---------- introspection ----------

    @property
    def study_id(self) -> int:
        return self._study_id

    @property
    def study_name(self) -> str:
        return self._study_name

    @property
    def engine(self) -> Engine:
        return self._engine

    @contextmanager
    def connection(self) -> Iterator:  # pragma: no cover - thin wrapper
        with self._engine.begin() as conn:
            yield conn

    # ---------- sample operations ----------

    def add(self, sample: Sample) -> Sample:
        """Insert a new sample; populates ``sample.id`` and timestamps."""
        if sample.id is not None:
            raise ValueError("Sample already has an id; use update() instead")
        now = _now()
        row = {
            "study_id": self._study_id,
            "phase": sample.phase,
            "iteration": sample.iteration,
            "inputs_json": json.dumps(sample.inputs.tolist()),
            "status": sample.status.value,
            "metric_json": _maybe_json_array(sample.metric),
            "folder": str(sample.folder) if sample.folder else None,
            "runtime_s": sample.runtime_s,
            "runner_task_id": sample.runner_task_id,
            "message": sample.message,
            "extra_json": json.dumps(sample.extra) if sample.extra else None,
            "created_at": now,
            "updated_at": now,
        }
        with self._engine.begin() as conn:
            result = conn.execute(samples_table.insert().values(**row))
            sample.id = int(result.inserted_primary_key[0])
        sample.created_at = now
        sample.updated_at = now
        return sample

    def add_many(self, samples: Iterable[Sample]) -> list[Sample]:
        out = []
        for s in samples:
            out.append(self.add(s))
        return out

    def update(self, sample: Sample) -> Sample:
        """Persist the current state of ``sample``."""
        if sample.id is None:
            raise ValueError("Sample has no id; insert with add() first")
        now = _now()
        with self._engine.begin() as conn:
            conn.execute(
                samples_table.update()
                .where(samples_table.c.id == sample.id)
                .values(
                    phase=sample.phase,
                    iteration=sample.iteration,
                    inputs_json=json.dumps(sample.inputs.tolist()),
                    status=sample.status.value,
                    metric_json=_maybe_json_array(sample.metric),
                    folder=str(sample.folder) if sample.folder else None,
                    runtime_s=sample.runtime_s,
                    runner_task_id=sample.runner_task_id,
                    message=sample.message,
                    extra_json=json.dumps(sample.extra) if sample.extra else None,
                    updated_at=now,
                )
            )
        sample.updated_at = now
        return sample

    def get(self, sample_id: int) -> Sample:
        with self._engine.begin() as conn:
            row = conn.execute(
                samples_table.select().where(samples_table.c.id == sample_id)
            ).mappings().first()
        if row is None:
            raise KeyError(f"No sample with id={sample_id}")
        return _row_to_sample(row)

    def list(
        self,
        *,
        phase: str | None = None,
        status: SampleStatus | None = None,
        iteration: int | None = None,
    ) -> list[Sample]:
        """List samples in this study, optionally filtered."""
        stmt = samples_table.select().where(samples_table.c.study_id == self._study_id)
        if phase is not None:
            stmt = stmt.where(samples_table.c.phase == phase)
        if status is not None:
            stmt = stmt.where(samples_table.c.status == status.value)
        if iteration is not None:
            stmt = stmt.where(samples_table.c.iteration == iteration)
        stmt = stmt.order_by(samples_table.c.id)
        with self._engine.begin() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [_row_to_sample(r) for r in rows]

    def count(self, *, phase: str | None = None, status: SampleStatus | None = None) -> int:
        from sqlalchemy import func

        stmt = select(func.count()).select_from(samples_table).where(samples_table.c.study_id == self._study_id)
        if phase is not None:
            stmt = stmt.where(samples_table.c.phase == phase)
        if status is not None:
            stmt = stmt.where(samples_table.c.status == status.value)
        with self._engine.begin() as conn:
            return int(conn.execute(stmt).scalar_one())

    def to_dataframe(self) -> pd.DataFrame:
        """All samples in this study as a flat pandas DataFrame."""
        with self._engine.begin() as conn:
            rows = conn.execute(
                samples_table.select().where(samples_table.c.study_id == self._study_id)
            ).mappings().all()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "id", "phase", "iteration", "inputs", "status", "metric",
                    "folder", "runtime_s", "runner_task_id", "message",
                    "created_at", "updated_at",
                ]
            )
        records = []
        for r in rows:
            records.append(
                {
                    "id": r["id"],
                    "phase": r["phase"],
                    "iteration": r["iteration"],
                    "inputs": json.loads(r["inputs_json"]),
                    "status": r["status"],
                    "metric": json.loads(r["metric_json"]) if r["metric_json"] else None,
                    "folder": r["folder"],
                    "runtime_s": r["runtime_s"],
                    "runner_task_id": r["runner_task_id"],
                    "message": r["message"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
            )
        return pd.DataFrame.from_records(records)

    # ---------- phase state (restart support) ----------

    def save_phase_state(
        self,
        phase: str,
        iteration: int,
        *,
        rng_state: bytes | None = None,
        surrogate_state: bytes | None = None,
    ) -> None:
        """Checkpoint phase-level state. Newer rows supersede older ones; we never delete."""
        with self._engine.begin() as conn:
            conn.execute(
                phase_state_table.insert().values(
                    study_id=self._study_id,
                    phase=phase,
                    iteration=iteration,
                    rng_state=rng_state,
                    surrogate_state=surrogate_state,
                    updated_at=_now(),
                )
            )

    def load_phase_state(self, phase: str) -> dict[str, bytes | int | None] | None:
        """Latest checkpoint for ``phase``, or ``None`` if no checkpoint exists."""
        stmt = (
            phase_state_table.select()
            .where(phase_state_table.c.study_id == self._study_id)
            .where(phase_state_table.c.phase == phase)
            .order_by(phase_state_table.c.id.desc())
            .limit(1)
        )
        with self._engine.begin() as conn:
            row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        return {
            "iteration": row["iteration"],
            "rng_state": row["rng_state"],
            "surrogate_state": row["surrogate_state"],
            "updated_at": row["updated_at"],
        }


def _maybe_json_array(arr: np.ndarray | None) -> str | None:
    return None if arr is None else json.dumps(np.asarray(arr).tolist())


def _row_to_sample(row: dict) -> Sample:
    metric = None
    if row["metric_json"]:
        metric = np.asarray(json.loads(row["metric_json"]), dtype=float)
    return Sample(
        id=row["id"],
        phase=row["phase"],
        iteration=row["iteration"],
        inputs=np.asarray(json.loads(row["inputs_json"]), dtype=float),
        status=SampleStatus(row["status"]),
        metric=metric,
        folder=Path(row["folder"]) if row["folder"] else None,
        runtime_s=row["runtime_s"],
        runner_task_id=row["runner_task_id"],
        message=row["message"],
        extra=json.loads(row["extra_json"]) if row["extra_json"] else {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
