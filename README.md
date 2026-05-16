# polarisopt

Modular design-of-experiments and Bayesian optimization for [POLARIS](https://polaris.taps.anl.gov/).

> **Status: 0.1.0 development.** API surface is in flux. Not yet ready for production use.

## What it does

`polarisopt` orchestrates POLARIS calibration and exploration studies. Two algorithm families:

1. **Static design of experiments** — Latin Hypercube, Morris, Sobol, manual designs. One-shot sample generation for screening and sensitivity analysis.
2. **Sequential design of experiments** — warm-up + surrogate-driven Bayesian optimization. Plug-in surrogates, acquisition functions, stopping criteria. Single- and multi-objective.

Studies are configured in YAML and executed via the `polarisopt` CLI or programmatically. Sample state is persisted to SQLite so studies can be resumed after interruptions.

## Install

```bash
pip install polarisopt              # core
pip install polarisopt[bo]          # + Bayesian opt (BoTorch + GPyTorch)
pip install polarisopt[anl]         # + Argonne integrations (Globus via polaris-studio)
pip install polarisopt[dev]         # + dev tooling
```

## Quick start

```bash
polarisopt run study.yaml
polarisopt status study.yaml
polarisopt resume study.yaml
```

## Architecture

Every swappable piece is an ABC with a registry. Adding a new design, surrogate, acquisition, generator, stopping criterion, metric, simulator, or runner means writing a class + a `@register(...)` decorator. The YAML loader looks up plugins by name.

```
polarisopt/
├── parameters/        # ParameterSpace, value injection into POLARIS JSONs
├── samples/           # Sample, SQLite-backed SampleStore (single source of truth)
├── config/            # pydantic study config + Jinja2 templating
├── design/            # Static DOE: LHS, Morris, Sobol, manual
├── surrogates/        # GP (BoTorch), ...
├── acquisition/       # EI, qEI, qEHVI, ...
├── generators/        # SampleGenerator strategies (batch-first)
├── stop/              # Stopping criteria
├── metrics/           # Metric ABC — scalar or vector outputs
├── simulator/         # Simulator ABC + MockSimulator + PolarisSimulator
├── runners/           # Runner ABC — local + slurm
├── studies/           # Orchestrators: static, sequential, pipeline
├── compat/            # Compatibility shims (e.g. EQSQL-shaped API over Slurm)
└── cli/
```

## EQSQL compatibility

Argonne users coming from `polaris.hpc.eqsql` can use `polarisopt.compat.eqsql` as a drop-in replacement. The wrapper offers the same `insert_task` / `Task.from_id` / `Task.cancel` surface but delegates to plain `sbatch`/`squeue`/`scancel` underneath — no Postgres, no worker pool.

```python
from polarisopt.compat import eqsql

result = eqsql.insert_task(
    definition={"task-type": "bash-script", "command": "/path/to/run.sh"},
    exp_id="my-experiment",
)
task = result.value
# task.task_id, task.status, task.get_logs(), task.cancel()
```

## License

BSD 3-Clause. See [License.txt](License.txt).
