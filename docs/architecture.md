# Architecture

`polarisopt` is built around a small set of abstract base classes, each
backed by a plugin registry that maps YAML ``type: ...`` strings to
concrete implementations.

## Module map

```
polarisopt/
├── parameters/    ParameterSpace, value injection into POLARIS JSONs
├── samples/       Sample, SQLite-backed SampleStore (single source of truth)
├── config/        pydantic study config + Jinja2 templating
├── design/        Static DOE: LHS, Morris, Sobol, manual
├── surrogates/    GP via BoTorch (single- and multi-output)
├── acquisition/   LogEI, qLogEI, qLogEHVI
├── generators/    SampleGenerator strategies (batch-first)
├── stop/          MaxIter, Epsilon, Plateau, Hypervolume, any/all
├── metrics/       Identity (benchmarks), LinkMoe, ChoiceShare
├── simulator/     Simulator ABC, MockSimulator, PolarisSimulator
├── transfer/      LocalTransfer, AnlTransfer (Globus via polaris-studio)
├── runners/       LocalRunner, SlurmRunner
├── studies/       Static and Sequential orchestrators + pipeline
├── compat/        Compatibility shims (EQSQL-shaped Slurm wrapper)
└── cli/           polarisopt run | status | resume
```

## The master/slave boundary

| Layer        | Lives where                  | Touches POLARIS? |
|--------------|------------------------------|------------------|
| `Study`      | Master Python process        | No               |
| `Surrogate`  | Master Python process        | No               |
| `Acquisition`| Master Python process        | No               |
| `Generator`  | Master Python process        | No               |
| `Runner`     | Master + (sbatch) Slurm queue| No               |
| `Simulator`  | Master process (staging) +   | **Yes** (slave)  |
|              | slave subprocess (execution) |                  |
| `Metric`     | Master Python process        | No (reads files) |
| `Transfer`   | Master Python process        | No               |

The master never imports POLARIS. ``PolarisSimulator.prepare`` stages a
per-sample workspace and returns a ``JobSpec``; ``Runner.submit`` hands
it to Slurm; the POLARIS binary executes on a compute node as the
slave; once the runner reports terminal, ``PolarisSimulator.collect_output``
reads the result file paths back, and ``Metric.compute`` reduces them
to a numeric objective.

## Plugin registries

Each pluggable ABC owns a ``Registry[T]`` instance and a corresponding
``make_xxx`` factory:

```python
from polarisopt.design.base import design_registry, make_design

@design_registry.register("my_method")
class MyDesign(Design):
    def generate(self, space, *, rng): ...

# in YAML
# phases:
#   - design: { type: my_method, options: { ... } }
```

The full set of registries (with their names) is:

| Family       | Registry constant      | Built-in names                                            |
|--------------|------------------------|-----------------------------------------------------------|
| Design       | `design_registry`      | `lhs`, `morris`, `sobol`, `manual`                        |
| Simulator    | `simulator_registry`   | `mock`, `polaris`                                         |
| Metric       | `metric_registry`      | `identity`, `link_moe`, `choice_share`                    |
| Runner       | `runner_registry`      | `local`, `slurm`                                          |
| Transfer     | `transfer_registry`    | `local`, `anl` (if `polaris-studio` installed)            |
| Surrogate    | `surrogate_registry`   | `gp` (if `botorch` installed)                             |
| Acquisition  | `acquisition_registry` | `ei`, `qei`, `qehvi` (if `botorch` installed)             |
| Generator    | `generator_registry`   | `random`, `acquisition`                                   |
| Stop         | `stop_registry`        | `max_iter`, `epsilon`, `plateau`, `hypervolume`, `any`, `all` |

## Persistence — SampleStore

A single SQLite database (`workspace/polarisopt.db`) is the source of truth.

```
studies(id, name, config_yaml, created_at)
samples(id, study_id, phase, iteration, inputs_json, status,
        metric_json, folder, runtime_s, runner_task_id, message,
        extra_json, created_at, updated_at)
phase_state(id, study_id, phase, iteration, rng_state,
            surrogate_state, updated_at)
```

WAL mode is enabled per-connection. Every Sample state transition goes
through `SampleStore.update`, so the master can be killed and resumed
via `polarisopt resume study.yaml`.

## Configuration — Jinja2 + pydantic

`config/loader.py` runs the YAML through Jinja2 (env vars and `now()`
available), then validates with pydantic models in `config/schema.py`.
Plugin sub-specs are kept as raw dicts so each plugin validates its own
options when it's instantiated.
