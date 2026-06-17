"""The ``polarisopt`` command-line entry point."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import click

from polarisopt import __version__
from polarisopt.config import load_study_config
from polarisopt.examples import example_path, list_examples, read_example
from polarisopt.samples.sample import Sample, SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.diff import diff_studies
from polarisopt.studies.ops import (
    ConfigDriftError,
    abort_study,
    cancel_sample,
    open_store,
    reconcile_running,
    recover_from_disk,
    retry_failed,
    sample_log_paths,
)
from polarisopt.studies.plan import plan_study
from polarisopt.studies.runner import StudyRunner
from polarisopt.studies.validate import validate_study
from polarisopt.utils.logging import configure
from polarisopt.utils.paths import workspace_layout
from polarisopt.utils.plugins import load_external_plugins


@click.group(name="polarisopt")
@click.version_option(__version__, prog_name="polarisopt")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
)
def cli(log_level: str) -> None:
    """polarisopt — design-of-experiments and Bayesian optimization for POLARIS."""
    configure(log_level.upper())  # type: ignore[arg-type]
    # Discover external plugin packages so their @register decorators run
    # before any factory is called.
    load_external_plugins()


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--quiet-heartbeat", is_flag=True,
    help=(
        "Suppress the periodic '[heartbeat] N sample(s) outstanding…' log "
        "lines. Useful for long studies whose log would otherwise be dominated "
        "by heartbeats. State transitions still log normally."
    ),
)
def run(config: Path, quiet_heartbeat: bool) -> None:
    """Run all phases in CONFIG (a study YAML)."""
    cfg = load_study_config(config)
    if quiet_heartbeat:
        _silence_heartbeat()
    runner = StudyRunner(cfg)
    samples = runner.run()
    finished = sum(1 for s in samples if s.status is SampleStatus.FINISHED)
    failed = sum(1 for s in samples if s.status is SampleStatus.FAILED)
    click.echo(f"completed: {finished}/{len(samples)} samples (failed: {failed})")


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--warnings-as-errors",
    is_flag=True,
    help="Treat warnings as errors (exit nonzero if any warning fires).",
)
@click.option(
    "--deep",
    is_flag=True,
    help=(
        "Also stage sample 0 and render its JobSpec (catches missing modules, "
        "wrong scenario keys, runner-script typos). Slower but more thorough."
    ),
)
def validate(config: Path, warnings_as_errors: bool, deep: bool) -> None:
    """Pre-flight validation. Exits 0 if the study is ready to run."""
    report = validate_study(config)
    click.echo(report.render())
    if not report.ok or (warnings_as_errors and report.warnings):
        raise click.exceptions.Exit(1)
    if deep:
        click.echo("---")
        click.echo("running deep validation (staging sample 0)...")
        plan = plan_study(config)
        click.echo(plan.render())
        if not plan.ok:
            raise click.exceptions.Exit(1)


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override the per-sample workspace dir (default: <study workspace>/experiments/plan-sample).",
)
def plan(config: Path, workspace: Path | None) -> None:
    """Dry-run: stage sample 0, render its JobSpec, stop before sbatch.

    The staged folder is left intact so you can inspect what would have
    been submitted. Catches the operational failures that schema-only
    validate misses (missing modules, scenario JSON key typos, runner
    script paths, parameter file relpaths, ...).
    """
    report = plan_study(config, workspace_override=workspace)
    click.echo(report.render())
    if not report.ok:
        raise click.exceptions.Exit(1)


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--verbose", "-v", is_flag=True,
    help="One row per sample: id, phase, status, jobid, runtime, folder, last log line.",
)
@click.option(
    "--status", "status_filter", default=None,
    type=click.Choice([s.value for s in SampleStatus], case_sensitive=False),
    help="(--verbose only) restrict to samples in this status.",
)
def status(config: Path, verbose: bool, status_filter: str | None) -> None:
    """Show sample counts (or a verbose per-sample table with -v)."""
    cfg = load_study_config(config)
    layout = workspace_layout(cfg.workspace)
    if not layout["db"].exists():
        click.echo(f"no store yet at {layout['db']}")
        return
    store = SampleStore.open(layout["db"], cfg.name)
    rows = store.list()
    if not rows:
        click.echo("(no samples)")
        return
    if not verbose:
        by_phase: dict[str, dict[str, int]] = {}
        for s in rows:
            by_phase.setdefault(s.phase, {}).setdefault(s.status.value, 0)
            by_phase[s.phase][s.status.value] = by_phase[s.phase].get(s.status.value, 0) + 1
        for phase, counts in by_phase.items():
            click.echo(f"{phase}: {counts}")
        return
    if status_filter:
        wanted = SampleStatus(status_filter.lower())
        rows = [s for s in rows if s.status is wanted]
    if not rows:
        click.echo("(no samples match)")
        return
    rows.sort(key=lambda s: (s.id or 0))
    header = f"{'id':>4} {'phase':<14} {'status':<9} {'retry':>5} {'jobid':<14} {'runtime':>10}  folder / last log line"
    click.echo(header)
    click.echo("-" * len(header))
    for s in rows:
        rt = _fmt_runtime(s)
        folder = str(s.folder) if s.folder else "-"
        last_line = _last_log_line(s.folder) if s.folder else ""
        retry_count = s.extra.get("retry_count", 0)
        retry_cell = str(retry_count) if retry_count else "-"
        click.echo(
            f"{(s.id or 0):>4} {s.phase[:14]:<14} {s.status.value:<9} {retry_cell:>5} "
            f"{(s.runner_task_id or '-')[:14]:<14} {rt:>10}  {folder}"
        )
        if last_line:
            click.echo(f"     └─ {last_line}")


def _fmt_runtime(sample: Sample) -> str:
    if sample.runtime_s is not None:
        return _fmt_seconds(sample.runtime_s)
    if sample.status is SampleStatus.RUNNING and sample.updated_at is not None:
        from datetime import datetime

        from polarisopt.utils._compat import UTC

        # store rows may be naive in older DBs; treat naive as UTC.
        updated = sample.updated_at
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
        delta = (datetime.now(UTC) - updated).total_seconds()
        return f"{_fmt_seconds(delta)}+"
    return "-"


def _fmt_seconds(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}m"
    return f"{secs / 3600:.1f}h"


def _find_binary_progress_log(
    folder: Path | None, *, iteration_match: str | None = None
) -> Path | None:
    """Locate the POLARIS binary's ``log/polaris_progress.log`` under ``folder``.

    polarislib writes the progress log to
    ``<folder>/<output_dirname>_<iter_str>[_<N>]/log/polaris_progress.log``.
    We search for any ``polaris_progress.log`` 2 levels deep so we don't
    need to know which iteration_type / output_dirname the sample used.

    Parameters
    ----------
    folder :
        Per-sample workspace folder.
    iteration_match :
        Substring filter on the iteration directory name (e.g. ``abm_init``).
        When set, only matches under directories whose name contains this
        string are considered. Use to disambiguate when a sample produced
        both ``*_abm_init_iteration`` and ``*_normal_iteration_1`` dirs.
    """
    if folder is None or not folder.exists():
        return None
    matches = list(folder.glob("*/log/polaris_progress.log"))
    if iteration_match is not None:
        matches = [m for m in matches if iteration_match in m.parent.parent.name]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _last_log_line(folder: Path | None, max_chars: int = 200) -> str:
    if folder is None or not folder.exists():
        return ""
    files = []
    for pat in ("*.log", "*.out", "*.err"):
        files.extend(folder.glob(pat))
    files = [p for p in files if p.is_file() and p.stat().st_size > 0]
    if not files:
        return ""
    target = max(files, key=lambda p: p.stat().st_mtime)
    try:
        text = target.read_text(errors="replace")
    except OSError:
        return ""
    for line in reversed(text.splitlines()):
        line = line.rstrip()
        if line:
            if len(line) > max_chars:
                line = line[:max_chars] + "..."
            return f"{target.name}: {line}"
    return ""


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--skip-reconcile",
    is_flag=True,
    help="Don't reconcile RUNNING samples (skips runner.status + disk recovery).",
)
@click.option(
    "--force", is_flag=True,
    help=(
        "Resume even if the simulator/runner config has drifted since the "
        "existing samples ran. Default: refuse to mix runs across edited YAMLs."
    ),
)
@click.option(
    "--quiet-heartbeat", is_flag=True,
    help=(
        "Suppress the periodic '[heartbeat] N sample(s) outstanding…' log "
        "lines. Useful for long studies whose log would otherwise be dominated "
        "by heartbeats. State transitions still log normally."
    ),
)
def resume(
    config: Path, skip_reconcile: bool, force: bool, quiet_heartbeat: bool,
) -> None:
    """Pick up an interrupted study (re-evaluates any PENDING samples).

    Before running:

    1. Optional config-drift check (skipped with --force) — refuses to
       resume if the simulator/runner config has been edited since the
       existing samples ran. Mirrors retry-failed's check.
    2. reconcile_running — for each previously-RUNNING sample,
       runner.status + disk recovery (FINISHED if outputs parse).
    3. recover_from_disk — sweeps any still-non-FINISHED samples for
       on-disk artifacts. Catches zombies that reconcile missed
       (runner.status raised, metric changed, etc.).
    """
    cfg = load_study_config(config)
    layout = workspace_layout(cfg.workspace)
    if not layout["db"].exists():
        raise click.ClickException(f"no store at {layout['db']}; run first")
    store = SampleStore.open(layout["db"], cfg.name)

    if not force:
        _check_resume_drift(cfg, store)

    if quiet_heartbeat:
        _silence_heartbeat()

    if not skip_reconcile:
        reconciled = reconcile_running(cfg, store=store)
        if reconciled:
            click.echo(f"reconciled {len(reconciled)} previously-RUNNING sample(s)")
        recovered = recover_from_disk(cfg, store=store)
        if recovered:
            click.echo(f"recovered {len(recovered)} sample(s) from disk")

    runner = StudyRunner(cfg, store=store)
    samples = runner.run()
    click.echo(f"resume complete: {len(samples)} samples processed")


def _check_resume_drift(cfg, store) -> None:
    """Raise ClickException if any existing sample's recorded fingerprint
    differs from the current simulator+runner config.
    """
    from polarisopt.studies.ops import (
        EXTRA_FINGERPRINT_KEY,
        simulator_config_fingerprint,
    )
    current_fp = simulator_config_fingerprint(cfg)
    drifted = [
        s for s in store.list()
        if (rec := s.extra.get(EXTRA_FINGERPRINT_KEY)) is not None
        and rec != current_fp
    ]
    if drifted:
        recorded = sorted({s.extra.get(EXTRA_FINGERPRINT_KEY) for s in drifted})
        raise click.ClickException(
            f"simulator/runner config has changed since {len(drifted)} "
            f"existing sample(s) ran (recorded: {recorded}; current: "
            f"{current_fp!r}). Pass --force to resume under the new config, "
            f"or use a distinct workspace per variant."
        )


def _silence_heartbeat() -> None:
    """Downgrade the studies.base heartbeat INFO line to DEBUG.

    Cheaper than a separate log file and lets state-transition lines
    keep their default INFO visibility. Heartbeats still happen — they
    just don't print at the default log level.
    """
    import logging
    studies_log = logging.getLogger("polarisopt.studies.base")

    class _SkipHeartbeat(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "[heartbeat]" not in record.getMessage()

    studies_log.addFilter(_SkipHeartbeat())


@cli.command(name="retry-failed")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--id",
    "ids",
    type=int,
    multiple=True,
    help="Specific sample id(s) to retry (repeat for many). Default: every FAILED sample.",
)
@click.option(
    "--run/--no-run",
    default=False,
    help="After flipping FAILED → PENDING, immediately re-run the study to evaluate them.",
)
@click.option(
    "--force", is_flag=True,
    help=(
        "Retry even if the simulator/runner config has drifted since "
        "the failed samples ran. Without this, retry-failed refuses to "
        "mix runs across edited YAMLs in one SampleStore."
    ),
)
def retry_failed_cmd(config: Path, ids: tuple[int, ...], run: bool, force: bool) -> None:
    """Flip FAILED samples back to PENDING (and optionally re-run the study)."""
    cfg = load_study_config(config)
    try:
        retried = retry_failed(cfg, sample_ids=list(ids) if ids else None, force=force)
    except ConfigDriftError as exc:
        raise click.ClickException(f"{exc}\n(use --force to override)") from exc
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not retried:
        click.echo("no FAILED samples to retry")
        return
    click.echo(f"retried {len(retried)} sample(s): {sorted(s.id for s in retried)}")  # type: ignore[type-var]
    if run:
        store = open_store(cfg)
        click.echo("re-running study...")
        samples = StudyRunner(cfg, store=store).run()
        finished = sum(1 for s in samples if s.status is SampleStatus.FINISHED)
        failed = sum(1 for s in samples if s.status is SampleStatus.FAILED)
        click.echo(f"completed: {finished}/{len(samples)} samples (failed: {failed})")


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--objective", "objective_index", type=int, default=0, show_default=True,
    help="Which objective column to optimize over (multi-objective studies).",
)
@click.option(
    "--maximize", is_flag=True,
    help="Argmax instead of argmin (default).",
)
@click.option(
    "--phase", default=None,
    help="Restrict to samples from this phase (default: all phases).",
)
@click.option(
    "--json", "as_json", is_flag=True,
    help="Print the result as a JSON object (id, inputs, metric, folder).",
)
def best(
    config: Path,
    objective_index: int,
    maximize: bool,
    phase: str | None,
    as_json: bool,
) -> None:
    """Print the best finished sample's id, inputs, and metric.

    Wraps ``SampleStore.best_so_far``. ``--maximize`` flips the
    direction; ``--phase`` restricts the search; ``--json`` makes the
    output machine-readable for shell pipelines.
    """
    import json as _json
    cfg = load_study_config(config)
    layout = workspace_layout(cfg.workspace)
    if not layout["db"].exists():
        raise click.ClickException(f"no store at {layout['db']}; run first")
    store = SampleStore.open(layout["db"], cfg.name)
    result = store.best_so_far(
        objective_index=objective_index,
        minimize=not maximize,
        phase=phase,
    )
    if result is None:
        raise click.ClickException("no FINISHED samples in the store")
    sample, value = result
    if as_json:
        click.echo(
            _json.dumps(
                {
                    "id": sample.id,
                    "phase": sample.phase,
                    "iteration": sample.iteration,
                    "inputs": list(sample.inputs),
                    "metric": list(sample.metric) if sample.metric is not None else None,
                    "objective_value": value,
                    "folder": str(sample.folder) if sample.folder else None,
                }
            )
        )
        return
    direction = "max" if maximize else "min"
    click.echo(f"best sample (arg{direction} over obj[{objective_index}])")
    click.echo(f"  id:        {sample.id}")
    click.echo(f"  phase:     {sample.phase}")
    click.echo(f"  iteration: {sample.iteration}")
    click.echo(f"  inputs:    {list(sample.inputs)}")
    if sample.metric is not None:
        click.echo(f"  metric:    {list(sample.metric)}")
    click.echo(f"  obj[{objective_index}]:    {value:.6g}  ({direction})")
    if sample.folder:
        click.echo(f"  folder:    {sample.folder}")


@cli.command(name="recover-from-disk")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--include-cancelled", is_flag=True,
    help="Also sweep CANCELLED samples (default: skip them to preserve user intent).",
)
def recover_from_disk_cmd(config: Path, include_cancelled: bool) -> None:
    """Harvest non-FINISHED samples whose outputs are already on disk.

    For when the master died mid-study and the runner no longer remembers
    the jobids (sacct retention aged them out). Tries
    simulator.collect_output + metric.compute on each RUNNING/FAILED
    sample's folder; on success the sample becomes FINISHED with the
    metric value persisted, no compute re-run needed.
    """
    cfg = load_study_config(config)
    try:
        recovered = recover_from_disk(cfg, include_cancelled=include_cancelled)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    if not recovered:
        click.echo("no recoverable samples (nothing on disk or schemas don't parse)")
        return
    ids = sorted(s.id for s in recovered if s.id is not None)
    click.echo(f"recovered {len(recovered)} sample(s) from disk: {ids}")


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("sample_id", type=int)
def cancel(config: Path, sample_id: int) -> None:
    """Cancel a single sample (scancel its Slurm job and mark CANCELLED)."""
    cfg = load_study_config(config)
    try:
        sample = cancel_sample(sample_id, config=cfg)
    except (FileNotFoundError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"sample {sample.id}: status={sample.status.value}")


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def abort(config: Path) -> None:
    """Cancel every non-terminal sample in the study."""
    cfg = load_study_config(config)
    try:
        cancelled = abort_study(cfg)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"aborted {len(cancelled)} non-terminal sample(s)")


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("sample_id", type=int)
@click.option("--follow", "-f", is_flag=True, help="Stream new log lines (like tail -f).")
@click.option("--lines", "-n", type=int, default=0, help="Print last N lines first (default: all).")
@click.option(
    "--binary", is_flag=True,
    help=(
        "Show the POLARIS binary's per-iteration progress log "
        "(``<output_dir>/log/polaris_progress.log``) instead of the "
        "polarisopt wrapper logs. This is what tells you what sim-hour "
        "the run is in. polaris_convergence simulators only."
    ),
)
@click.option(
    "--iteration",
    default=None,
    help=(
        "(--binary only) restrict the search to dirs containing this "
        "substring, e.g. ``abm_init``. Use when a sample produced "
        "multiple iteration dirs (abm_init + normal_iteration) and the "
        "default 'latest mtime' picks the wrong one."
    ),
)
def logs(
    config: Path,
    sample_id: int,
    follow: bool,
    lines: int,
    binary: bool,
    iteration: str | None,
) -> None:
    """Print stdout / stderr / *.log files for a sample."""
    cfg = load_study_config(config)
    try:
        store = open_store(cfg)
        sample = store.get(sample_id)
    except (FileNotFoundError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc

    if binary:
        progress = _find_binary_progress_log(sample.folder, iteration_match=iteration)
        if progress is None:
            hint = f" matching {iteration!r}" if iteration else ""
            raise click.ClickException(
                f"no polaris_progress.log found under {sample.folder}{hint} "
                f"(binary may not have started writing yet)"
            )
        files = [progress]
    elif iteration is not None:
        raise click.ClickException("--iteration only applies with --binary")
    else:
        files = sample_log_paths(sample)
    if not files:
        raise click.ClickException(f"no log files in {sample.folder}")

    for path in files:
        click.secho(f"==> {path} <==", fg="cyan")
        text = path.read_text(errors="replace")
        if lines > 0:
            text = "\n".join(text.splitlines()[-lines:])
        click.echo(text)

    if not follow:
        return

    # Naive single-file follow (the largest log) — terminates on Ctrl-C.
    target = max(files, key=lambda p: p.stat().st_size)
    click.secho(f"\n[follow] {target}", fg="cyan")
    with target.open("r") as fh:
        fh.seek(0, 2)  # end
        try:
            while True:
                line = fh.readline()
                if line:
                    click.echo(line, nl=False)
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass


@cli.command(name="smoke-test")
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Workspace directory for the smoke run (default: a fresh /tmp dir).",
)
@click.option("--keep", is_flag=True, help="Keep the workspace dir after the test passes.")
def smoke_test(workspace: Path | None, keep: bool) -> None:
    """End-to-end install check — runs a 4-point LHS study with the mock simulator.

    Verifies the core packages import cleanly, the SampleStore opens,
    the LocalRunner forks subprocesses, and the metric round-trip works.
    Takes about 5 seconds. No POLARIS, BoTorch, or Slurm needed.

    Exits 0 on success, 1 if anything fails. On failure the workspace
    is preserved so you can inspect logs.
    """
    import json as _json
    import shutil as _shutil
    import tempfile as _tempfile
    from importlib.metadata import version as _version

    workspace = workspace or Path(_tempfile.mkdtemp(prefix="polarisopt-smoke-"))
    click.echo(f"polarisopt smoke-test ({_version('polarisopt')})")
    click.echo(f"workspace: {workspace}")

    config_path = workspace / "smoke.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_text = (
        f"name: smoke\n"
        f"workspace: {workspace}\n"
        f"seed: 1\n"
        f"simulator: {{ type: mock, options: {{ function: quadratic }} }}\n"
        f"runner: {{ type: local, options: {{}} }}\n"
        f"parameters:\n"
        f"  inline:\n"
        f"    - {{ name: x, file: dummy.json, min: -1.0, max: 1.0 }}\n"
        f"    - {{ name: y, file: dummy.json, min: -1.0, max: 1.0 }}\n"
        f"metric: {{ type: identity, options: {{ keys: value }} }}\n"
        f"phases:\n"
        f"  - name: smoke\n"
        f"    type: static\n"
        f"    design: {{ type: lhs, options: {{ n: 4 }} }}\n"
    )
    config_path.write_text(config_text)

    try:
        report = validate_study(config_path)
        if not report.ok:
            click.echo(report.render())
            raise click.ClickException("validate failed")

        cfg = load_study_config(config_path)
        samples = StudyRunner(cfg).run()
        finished = sum(1 for s in samples if s.status is SampleStatus.FINISHED)
        if finished != len(samples):
            failed = [s for s in samples if s.status is SampleStatus.FAILED]
            for s in failed:
                click.echo(f"  failed sample {s.id}: {s.message}")
            raise click.ClickException(
                f"only {finished}/{len(samples)} samples finished — see {workspace}"
            )

        store = open_store(cfg)
        best = store.best_so_far()
        assert best is not None
        click.echo(f"ok: {len(samples)} samples finished, best={best[1]:.4g}")

        df = store.to_dataframe()
        assert len(df) == len(samples)
        click.echo(f"ok: SampleStore round-trip ({len(df)} rows)")

        summary = workspace / "smoke-summary.json"
        summary.write_text(
            _json.dumps(
                {
                    "version": _version("polarisopt"),
                    "samples": len(samples),
                    "best": best[1],
                },
                indent=2,
            )
        )
        click.echo(f"ok: summary written to {summary}")
    except Exception:
        click.echo(f"FAILED. Workspace preserved at: {workspace}")
        raise

    if not keep:
        _shutil.rmtree(workspace, ignore_errors=True)


@cli.command()
@click.argument("config_a", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("config_b", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def diff(config_a: Path, config_b: Path) -> None:
    """Compare two study runs side by side."""
    try:
        d = diff_studies(config_a, config_b)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(d.render())


@cli.group()
def examples() -> None:
    """Bundled example study YAMLs."""


@examples.command("list")
def examples_list() -> None:
    """List the bundled example YAMLs."""
    names = list_examples()
    if not names:
        click.echo("(no examples bundled)")
        return
    for name in names:
        click.echo(name)


@examples.command("show")
@click.argument("name")
def examples_show(name: str) -> None:
    """Print the YAML text of a bundled example to stdout."""
    try:
        click.echo(read_example(name), nl=False)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc


@examples.command("copy")
@click.argument("name")
@click.argument("destination", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--force", is_flag=True, help="Overwrite DESTINATION if it exists.")
def examples_copy(name: str, destination: Path, force: bool) -> None:
    """Copy a bundled example YAML to DESTINATION for local editing."""
    try:
        src = example_path(name)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    if destination.exists() and not force:
        raise click.ClickException(
            f"{destination} already exists; pass --force to overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, destination)
    click.echo(f"copied {name} -> {destination}")


def main() -> None:
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
