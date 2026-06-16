# polarisopt

Modular design-of-experiments and Bayesian optimization for [POLARIS](https://polaris.taps.anl.gov/).

> **Status: 0.9.x.** Pre-1.0 but battle-tested on live POLARIS calibration
> work since v0.6 (May 2026). The plugin / YAML / SampleStore surface has
> been additive across releases — no breaking changes since v0.6. 1.0
> will lock the surface once the choice-model calibration workflow lands
> its second production run.

## What it does

`polarisopt` orchestrates POLARIS calibration and exploration studies. Two algorithm families:

1. **Static design of experiments** — Latin Hypercube, Morris, Sobol, manual designs. One-shot sample generation for screening and sensitivity analysis.
2. **Sequential design of experiments** — warm-up + surrogate-driven Bayesian optimization. Plug-in surrogates, acquisition functions, stopping criteria. Single- and multi-objective.

Studies are configured in YAML and executed via the `polarisopt` CLI or programmatically. Sample state is persisted to SQLite so studies can be resumed after interruptions.

## Install

Python 3.10+.

```bash
pip install polarisopt              # core
pip install polarisopt[bo]          # + Bayesian opt (BoTorch + GPyTorch)
pip install polarisopt[anl]         # + Argonne integrations (Globus via polaris-studio)
pip install polarisopt[dev]         # + dev tooling
```

## Quick start

```bash
polarisopt validate study.yaml        # schema + plugin-option signature check
polarisopt plan     study.yaml        # stage sample 0, render sbatch script, don't submit
polarisopt run      study.yaml        # submit + poll
polarisopt status   study.yaml -v     # per-sample table: jobid, runtime, last log line
polarisopt logs     study.yaml 42 --binary --iteration=abm_init
polarisopt resume   study.yaml        # pick up an interrupted run
polarisopt retry-failed study.yaml --run
```

See [docs/getting-started.md](docs/getting-started.md) for a worked
example and [docs/how-to/common-mistakes.md](docs/how-to/common-mistakes.md)
for the typo class `validate` catches in <1s.

## Architecture

Every swappable piece is an ABC with a registry. Adding a new design, surrogate, acquisition, generator, stopping criterion, metric, simulator, or runner means writing a class + a `@register(...)` decorator. The YAML loader looks up plugins by name.

```
polarisopt/
├── parameters/        # ParameterSpace, value injection into POLARIS JSONs
├── samples/           # Sample, SQLite-backed SampleStore (single source of truth)
├── config/            # pydantic study config + Jinja2 templating
├── design/            # Static DOE: LHS, Morris, Sobol, manual
├── surrogates/        # GP (BoTorch), MultiTaskGP, ...
├── acquisition/       # LogEI, qLogEI, qLogEHVI, ...
├── generators/        # SampleGenerator strategies (batch-first)
├── stop/              # Stopping criteria (max_iter, epsilon, plateau, hypervolume)
├── metrics/           # Metric ABC — identity, link_moe, choice_share, constant
├── simulator/         # Simulator ABC + MockSimulator + PolarisSimulator + PolarisConvergenceSimulator
├── runners/           # Runner ABC — LocalRunner + SlurmRunner (direct sbatch)
├── transfer/          # File staging — LocalTransfer + AnlTransfer + GlobusTransfer
├── studies/           # Orchestrators: static, sequential + StudyRunner
├── utils/             # Logging, paths, plugin discovery, _compat shims
├── cli.py             # Click entry point: validate, plan, run, status, resume, retry-failed, ...
└── eqsql_compat.py    # Drop-in shim for polaris.hpc.eqsql
```

## EQSQL compatibility

Argonne users coming from `polaris.hpc.eqsql` can use `polarisopt.eqsql_compat` as a drop-in replacement. It offers the same `insert_task` / `Task.from_id` / `Task.cancel` surface but routes everything through plain `sbatch`/`squeue`/`scancel` — no Postgres, no worker pool, no cross-user contamination by construction.

```python
from polarisopt import eqsql_compat

with eqsql_compat.open_queue("/path/to/workspace") as queue:
    result = queue.insert_task(
        definition={"task-type": "bash-script", "command": "/path/to/run.sh"},
        exp_id="my-experiment",
    )
    task = result.value
    # task.task_id, task.status, task.get_logs(), task.cancel()
```

See [docs/how-to/migrate-from-eqsql.md](docs/how-to/migrate-from-eqsql.md)
for the full migration table and why per-user worker pinning drops out
under the SlurmRunner model.

## License

BSD 3-Clause. See [License.txt](License.txt).
