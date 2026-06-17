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
from polarisopt.studies.ops import reconcile_running, recover_from_disk
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


def _write_mock_output(folder: Path, value: float) -> None:
    """Write the MockSimulator's outputs.json so collect_output works."""
    import json as _json
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "outputs.json").write_text(_json.dumps({"value": value, "runtime_s": 1.0}))


def test_reconcile_picks_up_terminal_jobs_without_disk_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    """When disk artifacts are missing, fall back to runner verdict."""
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
    monkeypatch.setattr(
        "polarisopt.studies.ops.build_runner",
        lambda cfg: _ScriptedRunner(statuses),
    )

    store = SampleStore.open(workspace_layout(workspace)["db"], cfg.name)
    reconciled = reconcile_running(cfg, store=store)

    reconciled_ids = {s.runner_task_id for s in reconciled}
    # All four go terminal: FINISHED-but-missing-output is now FAILED
    # (was the pre-v0.10.1 "left as RUNNING" hole).
    assert reconciled_ids == {"fin-1", "fail-1", "cancel-1", "unk-1"}

    by_id = {s.runner_task_id: s for s in store.list()}
    assert by_id["fin-1"].status is SampleStatus.FAILED
    assert "FINISHED on resume but output is missing" in by_id["fin-1"].message
    assert by_id["fail-1"].status is SampleStatus.FAILED
    assert "runner FAILED on resume" in by_id["fail-1"].message
    assert by_id["cancel-1"].status is SampleStatus.CANCELLED
    assert by_id["unk-1"].status is SampleStatus.FAILED
    assert "orphan" in by_id["unk-1"].message


def test_reconcile_running_or_queued_left_alone(
    tmp_path: Path, monkeypatch
) -> None:
    """Live jobs (RUNNING/QUEUED) must be left RUNNING — don't race a partial write."""
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    _seed_running_samples(workspace, cfg.name, ["live-1", "queued-1"])

    statuses = {"live-1": JobStatus.RUNNING, "queued-1": JobStatus.QUEUED}
    monkeypatch.setattr(
        "polarisopt.studies.ops.build_runner",
        lambda cfg: _ScriptedRunner(statuses),
    )
    store = SampleStore.open(workspace_layout(workspace)["db"], cfg.name)
    reconciled = reconcile_running(cfg, store=store)
    assert reconciled == []
    by_id = {s.runner_task_id: s for s in store.list()}
    assert by_id["live-1"].status is SampleStatus.RUNNING
    assert by_id["queued-1"].status is SampleStatus.RUNNING


def test_reconcile_recovers_from_disk_when_runner_says_unknown(
    tmp_path: Path, monkeypatch
) -> None:
    """The DOE-agent zombie case: sacct GC'd the jobid but the outputs are
    on disk. Disk wins over the UNKNOWN verdict.
    """
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    rows = _seed_running_samples(workspace, cfg.name, ["zombie-1", "really-orphan"])
    _write_mock_output(rows[0].folder, value=0.04)
    # rows[1].folder is intentionally not created → real orphan.

    statuses = {"zombie-1": JobStatus.UNKNOWN, "really-orphan": JobStatus.UNKNOWN}
    monkeypatch.setattr(
        "polarisopt.studies.ops.build_runner",
        lambda cfg: _ScriptedRunner(statuses),
    )
    store = SampleStore.open(workspace_layout(workspace)["db"], cfg.name)
    reconciled = reconcile_running(cfg, store=store)

    by_id = {s.runner_task_id: s for s in store.list()}
    assert by_id["zombie-1"].status is SampleStatus.FINISHED
    assert by_id["zombie-1"].metric is not None
    assert "recovered from disk" in by_id["zombie-1"].message
    assert by_id["really-orphan"].status is SampleStatus.FAILED
    assert "orphan" in by_id["really-orphan"].message
    assert {s.runner_task_id for s in reconciled} == {"zombie-1", "really-orphan"}


def test_reconcile_disk_wins_over_runner_failed(
    tmp_path: Path, monkeypatch
) -> None:
    """If the binary wrote outputs before exiting non-zero, the outputs
    are valid; harvest them rather than discarding as FAILED."""
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    rows = _seed_running_samples(workspace, cfg.name, ["wrote-then-failed"])
    _write_mock_output(rows[0].folder, value=0.16)

    statuses = {"wrote-then-failed": JobStatus.FAILED}
    monkeypatch.setattr(
        "polarisopt.studies.ops.build_runner",
        lambda cfg: _ScriptedRunner(statuses),
    )
    store = SampleStore.open(workspace_layout(workspace)["db"], cfg.name)
    reconciled = reconcile_running(cfg, store=store)
    assert len(reconciled) == 1
    [s] = reconciled
    assert s.status is SampleStatus.FINISHED
    assert "recovered from disk" in s.message


def test_reconcile_cancelled_skips_disk_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    """User intent: a CANCELLED sample stays CANCELLED even if outputs exist."""
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace))
    cfg = load_study_config(cfg_path)
    rows = _seed_running_samples(workspace, cfg.name, ["user-cancelled"])
    _write_mock_output(rows[0].folder, value=0.0)

    statuses = {"user-cancelled": JobStatus.CANCELLED}
    monkeypatch.setattr(
        "polarisopt.studies.ops.build_runner",
        lambda cfg: _ScriptedRunner(statuses),
    )
    store = SampleStore.open(workspace_layout(workspace)["db"], cfg.name)
    [reconciled] = reconcile_running(cfg, store=store)
    assert reconciled.status is SampleStatus.CANCELLED


def test_recover_from_disk_sweeps_running_and_failed(tmp_path: Path) -> None:
    """The standalone retroactive sweep: scoops RUNNING + FAILED samples
    whose outputs are on disk into FINISHED, without consulting the runner.
    """
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace, runner_type="local"))
    cfg = load_study_config(cfg_path)
    workspace.mkdir(parents=True, exist_ok=True)
    store = SampleStore.open(workspace_layout(workspace)["db"], cfg.name)

    # Three samples, each with disk output: RUNNING / FAILED / CANCELLED.
    folders = []
    for i, status in enumerate(
        [SampleStatus.RUNNING, SampleStatus.FAILED, SampleStatus.CANCELLED]
    ):
        s = store.add(Sample(phase="p", inputs=np.array([float(i) * 0.1])))
        s.status = status
        s.folder = workspace / "experiments" / f"sim-{i:06d}"
        _write_mock_output(s.folder, value=float(i))
        store.update(s)
        folders.append(s.folder)

    # Default: CANCELLED skipped.
    recovered = recover_from_disk(cfg, store=store)
    assert len(recovered) == 2
    by_status = {s.status: s for s in recovered}
    assert SampleStatus.FINISHED in {s.status for s in recovered}
    assert by_status[SampleStatus.FINISHED].metric is not None

    # CANCELLED sample is still CANCELLED.
    cancelled_samples = [s for s in store.list() if s.status is SampleStatus.CANCELLED]
    assert len(cancelled_samples) == 1

    # include_cancelled=True scoops it too. (No reset needed — it's still CANCELLED.)
    recovered2 = recover_from_disk(cfg, store=store, include_cancelled=True)
    assert all(s.runner_task_id is None for s in recovered2)  # all of ours have no jobid
    final_cancelled = [s for s in store.list() if s.status is SampleStatus.CANCELLED]
    assert len(final_cancelled) == 0


def test_recover_from_disk_skips_samples_without_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace, runner_type="local"))
    cfg = load_study_config(cfg_path)
    workspace.mkdir(parents=True, exist_ok=True)
    store = SampleStore.open(workspace_layout(workspace)["db"], cfg.name)

    s = store.add(Sample(phase="p", inputs=np.array([0.5])))
    s.status = SampleStatus.RUNNING
    s.folder = workspace / "experiments" / "sim-000000"
    # No outputs.json written.
    store.update(s)
    assert recover_from_disk(cfg, store=store) == []
    assert store.get(s.id).status is SampleStatus.RUNNING


def test_recover_from_disk_cli(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace, runner_type="local"))
    cfg = load_study_config(cfg_path)
    workspace.mkdir(parents=True, exist_ok=True)
    store = SampleStore.open(workspace_layout(workspace)["db"], cfg.name)
    s = store.add(Sample(phase="p", inputs=np.array([0.5])))
    s.status = SampleStatus.RUNNING
    s.folder = workspace / "experiments" / "sim-000000"
    _write_mock_output(s.folder, value=0.25)
    store.update(s)

    res = CliRunner().invoke(cli, ["recover-from-disk", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "recovered 1 sample" in res.output


def test_recover_from_disk_cli_no_recoverable(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace, runner_type="local"))
    cfg = load_study_config(cfg_path)
    workspace.mkdir(parents=True, exist_ok=True)
    SampleStore.open(workspace_layout(workspace)["db"], cfg.name)
    res = CliRunner().invoke(cli, ["recover-from-disk", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "no recoverable samples" in res.output


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


# ---------- resume CLI (config-drift + recover-from-disk pass-through) ----------


def test_resume_cli_refuses_on_config_drift(tmp_path: Path) -> None:
    """resume must check the sample fingerprints and refuse if the YAML
    has drifted since the existing samples ran (symmetric with retry-failed)."""
    from polarisopt.studies.ops import EXTRA_FINGERPRINT_KEY

    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace, runner_type="local"))
    cfg = load_study_config(cfg_path)
    workspace.mkdir(parents=True, exist_ok=True)
    store = SampleStore.open(workspace_layout(workspace)["db"], cfg.name)
    s = store.add(Sample(phase="p", inputs=np.array([0.5])))
    s.status = SampleStatus.FINISHED
    s.metric = np.array([0.25])
    s.extra[EXTRA_FINGERPRINT_KEY] = "stale-fingerprint-from-prior"
    store.update(s)

    res = CliRunner().invoke(cli, ["resume", str(cfg_path)])
    assert res.exit_code != 0
    assert "config has changed" in res.output
    assert "--force" in res.output

    # --force allows the resume to proceed.
    res2 = CliRunner().invoke(cli, ["resume", str(cfg_path), "--force"])
    assert res2.exit_code == 0, res2.output


def test_resume_cli_runs_recover_from_disk(tmp_path: Path) -> None:
    """A RUNNING sample without a runner_task_id (so reconcile_running
    skips it) but with outputs on disk should be picked up by the
    recover_from_disk pass that resume calls automatically.
    """
    workspace = tmp_path / "ws"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(_yaml(workspace, runner_type="local"))
    cfg = load_study_config(cfg_path)
    workspace.mkdir(parents=True, exist_ok=True)
    store = SampleStore.open(workspace_layout(workspace)["db"], cfg.name)
    # No runner_task_id → reconcile_running can't touch it (filters on
    # samples with a task_id). Disk artifacts present → recover_from_disk
    # should pick it up.
    zombie = store.add(Sample(phase="p", inputs=np.array([0.5])))
    zombie.status = SampleStatus.RUNNING
    zombie.folder = workspace / "experiments" / "sim-zombie"
    _write_mock_output(zombie.folder, value=0.25)
    store.update(zombie)

    res = CliRunner().invoke(cli, ["resume", str(cfg_path)])
    assert res.exit_code == 0, res.output
    final = store.get(zombie.id)
    assert final.status is SampleStatus.FINISHED, res.output
    assert "recovered" in res.output
