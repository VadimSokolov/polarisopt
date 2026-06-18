"""Click invocations of the new cancel / abort / logs subcommands."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
from click.testing import CliRunner

from polarisopt.cli import cli
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.utils.paths import workspace_layout


def _yaml(workspace: Path) -> str:
    return dedent(
        f"""\
        name: cli-ext-{workspace.name}
        workspace: {workspace}
        simulator:
          type: mock
          options: {{ function: quadratic }}
        runner:
          type: local
          options: {{}}
        parameters:
          inline:
            - {{ name: x, file: a.json, min: 0.0, max: 1.0 }}
        metric:
          type: identity
          options: {{ keys: value }}
        phases:
          - name: p
            type: static
            design:
              type: manual
              options:
                points: [[0.5]]
        """
    )


def _seed(tmp_path: Path) -> tuple[Path, int]:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = tmp_path / "c.yaml"
    cfg.write_text(_yaml(workspace))
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    s = store.add(Sample(phase="p", inputs=np.array([0.5])))
    return cfg, s.id  # type: ignore[return-value]


def test_cancel_subcommand(tmp_path: Path) -> None:
    cfg_path, sid = _seed(tmp_path)
    res = CliRunner().invoke(cli, ["cancel", str(cfg_path), str(sid)])
    assert res.exit_code == 0, res.output
    assert "cancelled" in res.output


def test_abort_subcommand(tmp_path: Path) -> None:
    cfg_path, _ = _seed(tmp_path)
    res = CliRunner().invoke(cli, ["abort", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "aborted" in res.output


def test_logs_subcommand_missing(tmp_path: Path) -> None:
    cfg_path, sid = _seed(tmp_path)
    res = CliRunner().invoke(cli, ["logs", str(cfg_path), str(sid)])
    # No folder yet → ClickException
    assert res.exit_code != 0
    assert "no log files" in res.output


def test_logs_subcommand_prints_files(tmp_path: Path) -> None:
    cfg_path, sid = _seed(tmp_path)
    # Give the sample a folder with logs
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    sample = store.get(sid)
    sample.folder = workspace / "experiments" / "sim-0001"
    sample.folder.mkdir(parents=True)
    (sample.folder / "polaris.stdout.log").write_text("hello stdout")
    (sample.folder / "polaris.stderr.log").write_text("hello stderr")
    sample.status = SampleStatus.FINISHED
    sample.metric = np.array([0.25])
    store.update(sample)

    res = CliRunner().invoke(cli, ["logs", str(cfg_path), str(sid)])
    assert res.exit_code == 0, res.output
    assert "hello stdout" in res.output
    assert "hello stderr" in res.output


def test_logs_binary_flag_finds_progress_log(tmp_path: Path) -> None:
    """``polarisopt logs --binary`` surfaces ``log/polaris_progress.log``
    instead of the wrapper logs.
    """
    cfg_path, sid = _seed(tmp_path)
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    sample = store.get(sid)
    sample.folder = workspace / "experiments" / "sim-0001"
    sample.folder.mkdir(parents=True)
    (sample.folder / "polaris.stdout.log").write_text("wrapper noise")
    # polarislib-style nested progress log:
    log_dir = sample.folder / "DFW_01_abm_init_iteration_2" / "log"
    log_dir.mkdir(parents=True)
    (log_dir / "polaris_progress.log").write_text("sim hour 12 of 24\n")
    sample.status = SampleStatus.RUNNING
    store.update(sample)

    res = CliRunner().invoke(cli, ["logs", str(cfg_path), str(sid), "--binary"])
    assert res.exit_code == 0, res.output
    assert "sim hour 12 of 24" in res.output
    assert "wrapper noise" not in res.output


def test_logs_binary_iteration_filter_picks_matching_dir(tmp_path: Path) -> None:
    """When a sample produced abm_init AND normal_iteration dirs, the
    --iteration filter picks the right one.
    """
    cfg_path, sid = _seed(tmp_path)
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    sample = store.get(sid)
    sample.folder = workspace / "experiments" / "sim-0001"
    sample.folder.mkdir(parents=True)
    abm = sample.folder / "DFW_01_abm_init_iteration" / "log"
    abm.mkdir(parents=True)
    (abm / "polaris_progress.log").write_text("abm_init sim hour 12\n")
    # Normal iteration dir, mtime'd LATER so plain --binary picks it.
    normal = sample.folder / "DFW_dta_iteration_1" / "log"
    normal.mkdir(parents=True)
    (normal / "polaris_progress.log").write_text("normal iter sim hour 8\n")
    import os
    os.utime(normal / "polaris_progress.log", (10_000_000, 10_000_000))
    os.utime(abm / "polaris_progress.log", (9_000_000, 9_000_000))
    store.update(sample)

    # Without filter: picks the latest mtime (normal iter).
    res = CliRunner().invoke(cli, ["logs", str(cfg_path), str(sid), "--binary"])
    assert res.exit_code == 0, res.output
    assert "normal iter sim hour 8" in res.output

    # With --iteration=abm_init: pinned to abm_init dir.
    res_filt = CliRunner().invoke(
        cli, ["logs", str(cfg_path), str(sid), "--binary", "--iteration", "abm_init"]
    )
    assert res_filt.exit_code == 0, res_filt.output
    assert "abm_init sim hour 12" in res_filt.output
    assert "normal iter sim hour 8" not in res_filt.output


def test_logs_iteration_without_binary_errors(tmp_path: Path) -> None:
    cfg_path, sid = _seed(tmp_path)
    res = CliRunner().invoke(
        cli, ["logs", str(cfg_path), str(sid), "--iteration", "abm_init"]
    )
    assert res.exit_code != 0
    assert "--iteration only applies with --binary" in res.output


def test_logs_binary_flag_errors_when_missing(tmp_path: Path) -> None:
    cfg_path, sid = _seed(tmp_path)
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    sample = store.get(sid)
    sample.folder = workspace / "experiments" / "sim-0001"
    sample.folder.mkdir(parents=True)
    store.update(sample)

    res = CliRunner().invoke(cli, ["logs", str(cfg_path), str(sid), "--binary"])
    assert res.exit_code != 0
    assert "polaris_progress.log" in res.output


def test_status_verbose_shows_per_sample_rows(tmp_path: Path) -> None:
    cfg_path, sid = _seed(tmp_path)
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    sample = store.get(sid)
    sample.status = SampleStatus.RUNNING
    sample.runner_task_id = "12345678"
    sample.folder = workspace / "experiments" / "sim-0001"
    sample.folder.mkdir(parents=True)
    (sample.folder / "polaris.stdout.log").write_text("last interesting line\n")
    store.update(sample)

    res = CliRunner().invoke(cli, ["status", str(cfg_path), "--verbose"])
    assert res.exit_code == 0, res.output
    assert "12345678" in res.output
    assert "running" in res.output
    assert "last interesting line" in res.output


def test_status_verbose_status_filter(tmp_path: Path) -> None:
    cfg_path, sid = _seed(tmp_path)
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    # Add a second sample so we have something to filter against.
    other = store.add(Sample(phase="p", inputs=np.array([0.7])))
    other.status = SampleStatus.FINISHED
    other.metric = np.array([0.49])
    store.update(other)

    res = CliRunner().invoke(
        cli, ["status", str(cfg_path), "--verbose", "--status", "finished"]
    )
    assert res.exit_code == 0, res.output
    assert "finished" in res.output
    # The PENDING-only sample from _seed should be filtered out.
    rows_for_other = [ln for ln in res.output.splitlines() if str(other.id) in ln]
    assert len(rows_for_other) >= 1
    rows_for_sid = [ln for ln in res.output.splitlines() if ln.lstrip().startswith(str(sid)) and " pending " in ln]
    assert not rows_for_sid


def test_clean_failed_removes_failed_sample_workspaces(tmp_path: Path) -> None:
    """polarisopt clean --failed deletes on-disk workspaces of FAILED
    samples but leaves the store rows in place (so the message survives)."""
    cfg_path, sid = _seed(tmp_path)
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    # One FAILED sample with a populated folder, one FINISHED (should be untouched).
    failed_sample = store.get(sid)
    failed_sample.status = SampleStatus.FAILED
    failed_sample.folder = workspace / "experiments" / "sim-000001"
    failed_sample.folder.mkdir(parents=True)
    (failed_sample.folder / "wasted.log").write_text("dead bytes\n")
    store.update(failed_sample)

    finished = store.add(Sample(phase="p", inputs=np.array([0.9])))
    finished.status = SampleStatus.FINISHED
    finished.metric = np.array([0.81])
    finished.folder = workspace / "experiments" / "sim-000002"
    finished.folder.mkdir(parents=True)
    (finished.folder / "result.h5").write_text("ok\n")
    store.update(finished)

    res = CliRunner().invoke(cli, ["clean", str(cfg_path), "--failed"])
    assert res.exit_code == 0, res.output
    assert "removed 1 workspace" in res.output
    assert not failed_sample.folder.exists()
    assert finished.folder.exists()
    # Store row survives.
    assert store.get(sid).status is SampleStatus.FAILED


def test_clean_failed_dry_run_does_not_delete(tmp_path: Path) -> None:
    cfg_path, sid = _seed(tmp_path)
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    sample = store.get(sid)
    sample.status = SampleStatus.FAILED
    sample.folder = workspace / "experiments" / "sim-000001"
    sample.folder.mkdir(parents=True)
    (sample.folder / "stuff.log").write_text("x" * 1024)
    store.update(sample)

    res = CliRunner().invoke(cli, ["clean", str(cfg_path), "--failed", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "DRY RUN" in res.output
    assert sample.folder.exists(), "dry-run must NOT delete"


def test_clean_failed_no_candidates(tmp_path: Path) -> None:
    cfg_path, _ = _seed(tmp_path)
    res = CliRunner().invoke(cli, ["clean", str(cfg_path), "--failed"])
    assert res.exit_code == 0, res.output
    assert "nothing to clean" in res.output


def test_clean_requires_failed_flag(tmp_path: Path) -> None:
    cfg_path, _ = _seed(tmp_path)
    res = CliRunner().invoke(cli, ["clean", str(cfg_path)])
    assert res.exit_code != 0
    assert "--failed" in res.output


def test_status_verbose_prefers_recent_progress_log_over_stale_wrapper(
    tmp_path: Path,
) -> None:
    """``polarisopt status --verbose`` should surface the polarislib binary's
    progress log when the wrapper log has gone silent.

    This is the case the DFW DOE agent hit on Improv at 60 minutes in:
    polaris.stdout.log was stuck at the binary's boot banner while
    polaris_progress.log was minutes-fresh with sim-hour progress.
    """
    import os

    cfg_path, sid = _seed(tmp_path)
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    sample = store.get(sid)
    sample.status = SampleStatus.RUNNING
    sample.runner_task_id = "7628354.imgt1"
    sample.folder = workspace / "experiments" / "sim-000001"
    sample.folder.mkdir(parents=True)
    # Wrapper log: written at startup, mtime aged into the past.
    wrapper = sample.folder / "polaris.stdout.log"
    wrapper.write_text("=> Spreading across nodes [[Node 0 free=16]]\n")
    os.utime(wrapper, (9_000_000, 9_000_000))
    # Binary progress log under the iteration dir: minutes-fresh.
    iter_log = sample.folder / "DFW_01_abm_init_iteration" / "log"
    iter_log.mkdir(parents=True)
    progress = iter_log / "polaris_progress.log"
    progress.write_text("sim hour 12 of 24 — events: 1.2M\n")
    os.utime(progress, (10_000_000, 10_000_000))
    store.update(sample)

    res = CliRunner().invoke(cli, ["status", str(cfg_path), "--verbose"])
    assert res.exit_code == 0, res.output
    assert "sim hour 12 of 24" in res.output
    # The stale wrapper line must NOT show up — it's the bug we're fixing.
    assert "Spreading across nodes" not in res.output


def test_best_cli_returns_argmin(tmp_path: Path) -> None:
    cfg_path, _ = _seed(tmp_path)
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    # Three finished samples — best (argmin) is the lowest metric.
    for inputs, metric in [(0.2, 0.9), (0.5, 0.25), (0.8, 0.7)]:
        s = store.add(Sample(phase="p", inputs=np.array([inputs])))
        s.status = SampleStatus.FINISHED
        s.metric = np.array([metric])
        store.update(s)

    res = CliRunner().invoke(cli, ["best", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "0.25" in res.output
    # The lowest-metric sample is the one with inputs=0.5.
    assert "0.5" in res.output


def test_best_cli_maximize_flag(tmp_path: Path) -> None:
    cfg_path, _ = _seed(tmp_path)
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    for inputs, metric in [(0.2, 0.1), (0.5, 0.25), (0.8, 0.9)]:
        s = store.add(Sample(phase="p", inputs=np.array([inputs])))
        s.status = SampleStatus.FINISHED
        s.metric = np.array([metric])
        store.update(s)

    res = CliRunner().invoke(cli, ["best", str(cfg_path), "--maximize"])
    assert res.exit_code == 0, res.output
    assert "0.9" in res.output


def test_best_cli_json_output(tmp_path: Path) -> None:
    import json as _json

    cfg_path, _ = _seed(tmp_path)
    workspace = tmp_path / "ws"
    layout = workspace_layout(workspace)
    store = SampleStore.open(layout["db"], f"cli-ext-{workspace.name}")
    s = store.add(Sample(phase="p", inputs=np.array([0.5])))
    s.status = SampleStatus.FINISHED
    s.metric = np.array([0.42])
    store.update(s)

    res = CliRunner().invoke(cli, ["best", str(cfg_path), "--json"])
    assert res.exit_code == 0, res.output
    payload = _json.loads(res.output.strip())
    assert payload["objective_value"] == 0.42
    assert payload["inputs"] == [0.5]
    assert payload["metric"] == [0.42]


def test_best_cli_errors_on_empty_store(tmp_path: Path) -> None:
    cfg_path, _ = _seed(tmp_path)
    res = CliRunner().invoke(cli, ["best", str(cfg_path)])
    assert res.exit_code != 0
    assert "no FINISHED" in res.output
