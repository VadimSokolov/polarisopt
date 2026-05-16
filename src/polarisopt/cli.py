"""The ``polarisopt`` command-line entry point."""

from __future__ import annotations

import time
from pathlib import Path

import click

from polarisopt import __version__
from polarisopt.config import load_study_config
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.studies.ops import (
    abort_study,
    cancel_sample,
    open_store,
    sample_log_paths,
)
from polarisopt.studies.runner import StudyRunner
from polarisopt.utils.logging import configure
from polarisopt.utils.paths import workspace_layout


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
def resume(config: Path) -> None:
    """Pick up an interrupted study (re-evaluates any PENDING samples)."""
    cfg = load_study_config(config)
    layout = workspace_layout(cfg.workspace)
    if not layout["db"].exists():
        raise click.ClickException(f"no store at {layout['db']}; run first")
    store = SampleStore.open(layout["db"], cfg.name)
    runner = StudyRunner(cfg, store=store)
    samples = runner.run()
    click.echo(f"resume complete: {len(samples)} samples processed")


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


def main() -> None:
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
