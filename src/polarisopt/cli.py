"""The ``polarisopt`` command-line entry point."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import click

from polarisopt import __version__
from polarisopt.config import load_study_config
from polarisopt.examples import example_path, list_examples, read_example
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.diff import diff_studies
from polarisopt.studies.ops import (
    abort_study,
    cancel_sample,
    open_store,
    reconcile_running,
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
def run(config: Path) -> None:
    """Run all phases in CONFIG (a study YAML)."""
    cfg = load_study_config(config)
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
def status(config: Path) -> None:
    """Show per-phase counts of samples in the store referenced by CONFIG."""
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
    by_phase: dict[str, dict[str, int]] = {}
    for s in rows:
        by_phase.setdefault(s.phase, {}).setdefault(s.status.value, 0)
        by_phase[s.phase][s.status.value] = by_phase[s.phase].get(s.status.value, 0) + 1
    for phase, counts in by_phase.items():
        click.echo(f"{phase}: {counts}")


@cli.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--skip-reconcile",
    is_flag=True,
    help="Don't reconcile RUNNING samples with the runner at startup.",
)
def resume(config: Path, skip_reconcile: bool) -> None:
    """Pick up an interrupted study (re-evaluates any PENDING samples).

    Before running, reconciles every previously-RUNNING sample with the
    runner: if Slurm has already finished/failed/lost it, the store row
    is transitioned accordingly so resume isn't blocked.
    """
    cfg = load_study_config(config)
    layout = workspace_layout(cfg.workspace)
    if not layout["db"].exists():
        raise click.ClickException(f"no store at {layout['db']}; run first")
    store = SampleStore.open(layout["db"], cfg.name)
    if not skip_reconcile:
        reconciled = reconcile_running(cfg, store=store)
        if reconciled:
            click.echo(f"reconciled {len(reconciled)} previously-RUNNING sample(s)")
    runner = StudyRunner(cfg, store=store)
    samples = runner.run()
    click.echo(f"resume complete: {len(samples)} samples processed")


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
def retry_failed_cmd(config: Path, ids: tuple[int, ...], run: bool) -> None:
    """Flip FAILED samples back to PENDING (and optionally re-run the study)."""
    cfg = load_study_config(config)
    try:
        retried = retry_failed(cfg, sample_ids=list(ids) if ids else None)
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
def logs(config: Path, sample_id: int, follow: bool, lines: int) -> None:
    """Print stdout / stderr / *.log files for a sample."""
    cfg = load_study_config(config)
    try:
        store = open_store(cfg)
        sample = store.get(sample_id)
    except (FileNotFoundError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc

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
