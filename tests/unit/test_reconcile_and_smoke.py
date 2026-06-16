"""Tests for resume reconcile + the smoke-test subcommand."""

from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent

import numpy as np
from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.config import load_study_config
from polarisopt.runners.base import Job, JobSpec, JobStatus
from polarisopt.runners.slurm import SlurmRunner
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.ops import reconcile_running
from polarisopt.utils.paths import workspace_layout

# ---------- reconcile_running ----------


def _yaml(workspace: Path, runner_type: str = "slurm") -> str:
    runner_block = (
        "runner: { type: local, options: {} }"
        if runner_type == "local"
        else "runner: { type: slurm, options: {} }"
    )
    return dedent(
        f"""\
        name: reconcile-{workspace.name}
        workspace: {workspace}
        simulator: {{ type: mock, options: {{ function: quadratic }} }}
        {runner_block}
        parameters:
          inline:
            - {{ name: x, file: a.json, min: 0.0, max: 1.0 }}
        metric: {{ type: identity, options: {{ keys: value }} }}
        phases:
          - name: p
            type: static
            design: {{ type: manual, options: {{ points: [[0.5]] }} }}
        """
    )


def _ok(stdout: str = "", stderr: str = "", rc: int = 0):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


class _ScriptedRunner(SlurmRunner):
    """Slurm-shaped runner with a programmable status() return."""

    def __init__(self, statuses: dict[str, JobStatus]) -> None:
        super().__init__(shell_runner=lambda cmd: _ok())
        self._statuses = statuses

    def status(self, job: Job) -> Job:
        job.status = self._statuses.get(job.task_id, JobStatus.UNKNOWN)
        return job


def _seed_running_samples(workspace: Path, name: str, jobids: list[str]) -> list[Sample]:
    workspace.mkdir(parents=True, exist_ok=True)
    store = SampleStore.open(workspace_layout(workspace)["db"], name)
    rows = []
    for i, jid in enumerate(jobids):
        s = store.add(Sample(phase="p", inputs=np.array([float(i) * 0.1])))
        s.status = SampleStatus.RUNNING
        s.runner_task_id = jid
        s.folder = workspace / "experiments" / f"sim-{i:06d}"
        store.update(s)
        rows.append(s)
    return rows


def test_reconcile_picks_up_terminal_jobs(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    _seed_running_samples(workspace, cfg.name, ["fin-1", "fail-1", "cancel-1", "unk-1"])

    statuses = {
        "fin-1":    JobStatus.FINISHED,
        "fail-1":   JobStatus.FAILED,
        "cancel-1": JobStatus.CANCELLED,
        "unk-1":    JobStatus.UNKNOWN,
    }
    # Patch build_runner to return our scripted runner.
    monkeypatch.setattr(
        "polarisopt.studies.ops.build_runner",
        lambda cfg: _ScriptedRunner(statuses),
    )

    store = SampleStore.open(workspace_layout(workspace)["db"], cfg.name)
    reconciled = reconcile_running(cfg, store=store)

    # FINISHED stays RUNNING (orchestrator loop will collect output later).
    # FAILED, CANCELLED, UNKNOWN-as-orphan are reconciled.
    reconciled_ids = {s.runner_task_id for s in reconciled}
    assert reconciled_ids == {"fail-1", "cancel-1", "unk-1"}

    by_id = {s.runner_task_id: s for s in store.list()}
    assert by_id["fin-1"].status is SampleStatus.RUNNING
    assert by_id["fail-1"].status is SampleStatus.FAILED
    assert by_id["cancel-1"].status is SampleStatus.CANCELLED
    assert by_id["unk-1"].status is SampleStatus.FAILED
    assert by_id["unk-1"].message is not None and "orphan" in by_id["unk-1"].message


def test_reconcile_empty_when_nothing_running(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace, runner_type="local"))
    cfg = load_study_config(cfg_path)
    workspace.mkdir(parents=True, exist_ok=True)
    SampleStore.open(workspace_layout(workspace)["db"], cfg.name)
    assert reconcile_running(cfg) == []


# ---------- smoke-test CLI ----------


def test_smoke_test_passes(tmp_path: Path) -> None:
    res = CliRunner().invoke(cli, ["smoke-test", "--workspace", str(tmp_path / "ws"), "--keep"])
    assert res.exit_code == 0, res.output
    assert "4 samples finished" in res.output
    summary = tmp_path / "ws" / "smoke-summary.json"
    assert summary.exists()


def test_smoke_test_preserves_workspace_on_failure(tmp_path: Path, monkeypatch) -> None:
    """If the runner fails, smoke-test must keep the workspace for postmortem."""
    # Force the LocalRunner to always FAIL.
    from polarisopt.runners.local import LocalRunner

    class _FailingLocal(LocalRunner):
        def submit(self, spec: JobSpec) -> Job:
            return Job(spec=spec, task_id="x", status=JobStatus.FAILED, exit_code=1)

        def status(self, job: Job) -> Job:
            return job

    monkeypatch.setattr("polarisopt.runners.local.LocalRunner", _FailingLocal)
    monkeypatch.setattr("polarisopt.runners.factory.runner_registry._items",
                        {"local": _FailingLocal, "slurm": SlurmRunner})

    res = CliRunner().invoke(cli, ["smoke-test", "--workspace", str(tmp_path / "ws")])
    assert res.exit_code != 0
    assert "FAILED" in res.output
    # Workspace preserved
    assert (tmp_path / "ws").exists()
