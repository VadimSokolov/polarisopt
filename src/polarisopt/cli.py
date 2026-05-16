"""The ``polarisopt`` command-line entry point."""

from __future__ import annotations

from pathlib import Path

import click

from polarisopt import __version__
from polarisopt.config import load_study_config
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore
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


def main() -> None:
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
