# Use polarisopt from a Jupyter notebook

Everything `polarisopt` does from the CLI it can do from a Python cell.
The CLI is a thin Click wrapper over a small set of importable
functions; notebooks call those functions directly.

The recommended workflow for POLARIS-scale studies:

| Step | Where | Why |
|---|---|---|
| `validate`, `plan`, design exploration | notebook | iterate fast on YAML edits |
| Long-running `run` | terminal (`polarisopt run study.yaml`) | survives notebook kernel restarts; doesn't block the notebook |
| `status`, `logs`, post-run analysis | notebook | the `SampleStore` is the single source of truth |

For toy / mock studies that finish in seconds (`mock` simulator,
Branin/Rosenbrock), the whole loop fits in one cell.

## Programmatic API mirror of the CLI

Every CLI subcommand has an importable equivalent:

| CLI | Notebook |
|---|---|
| `polarisopt validate study.yaml` | `validate_study("study.yaml")` |
| `polarisopt validate --deep study.yaml` | `validate_study(...)` + `plan_study(...)` |
| `polarisopt plan study.yaml` | `plan_study("study.yaml")` |
| `polarisopt run study.yaml` | `StudyRunner(cfg).run()` |
| `polarisopt resume study.yaml` | `reconcile_running(cfg)` + `StudyRunner(cfg, store=...).run()` |
| `polarisopt status study.yaml` | `store.list()` / `store.to_dataframe()` |
| `polarisopt cancel study.yaml 42` | `cancel_sample(42, config=cfg)` |
| `polarisopt abort study.yaml` | `abort_study(cfg)` |
| `polarisopt logs study.yaml 42` | `sample_log_paths(store.get(42))` |
| `polarisopt retry-failed study.yaml` | `retry_failed(cfg)` |
| `polarisopt diff a.yaml b.yaml` | `diff_studies("a.yaml", "b.yaml")` |

### Imports

```python
from polarisopt.config import load_study_config
from polarisopt.samples.store import SampleStore
from polarisopt.samples.sample import SampleStatus
from polarisopt.studies.runner import StudyRunner
from polarisopt.studies.validate import validate_study
from polarisopt.studies.plan import plan_study
from polarisopt.studies.diff import diff_studies
from polarisopt.studies.ops import (
    open_store,
    cancel_sample,
    abort_study,
    retry_failed,
    reconcile_running,
    sample_log_paths,
)
from polarisopt.utils.paths import workspace_layout
```

### Validate before running

```python
report = validate_study("study.yaml")
if not report.ok:
    print(report.render())
    raise SystemExit("fix errors before submitting")
```

`report.errors`, `report.warnings`, `report.info` are plain lists if
you want to render them in a custom widget instead.

### Render the sbatch script without submitting

```python
plan = plan_study("study.yaml")
print(plan.render())                              # human-readable summary
print(plan.script_path.read_text() if plan.script_path else "(local runner — no script)")
```

### Drive a short study end-to-end

For studies small enough to finish in the notebook:

```python
cfg = load_study_config("branin.yaml")
samples = StudyRunner(cfg).run()
finished = [s for s in samples if s.status is SampleStatus.FINISHED]
print(f"{len(finished)}/{len(samples)} finished")
```

For studies that take hours, drive `polarisopt run` in a terminal and
analyze from the notebook (see below).

## Analyze a running or finished study

`SampleStore` is the single source of truth. Open it from any notebook
that can read the workspace's `polarisopt.db`:

```python
cfg = load_study_config("study.yaml")
store = open_store(cfg)        # convenience for SampleStore.open(layout["db"], cfg.name)
df = store.to_dataframe()
df.head()
```

### Partitioning by phase and iteration

Every sample row has both a `phase` (the YAML phase name) and an
`iteration` (integer batch index within that phase). For sequential
phases, `iteration=0` is the warm-up batch and `iteration=1, 2, …`
are the BO rounds. For static phases all samples share `iteration=0`.

This means analysis queries don't need to infer batch boundaries from
sample id ranges:

```python
# Best metric per BO iteration
df = store.to_dataframe()
df = df[df["status"] == "finished"]
df["objective"] = df["metric"].apply(lambda m: m[0] if m else None)
best_per_iter = df.groupby(["phase", "iteration"])["objective"].min()
```

The same column is in `Sample.iteration` if you're iterating samples
in Python instead of going through pandas.

### Analysis helpers (the full set)

The store ships five helpers built for notebook work. None of them
need the run to be finished — they operate on whatever's been written
so far.

```python
# All FINISHED samples (optionally filtered by phase)
finished = store.finished_samples(phase="bo")

# (n_finished, n_objectives) numpy array — drop into matplotlib / seaborn
Y = store.metric_matrix(phase="bo")

# Single best sample (argmin/argmax over one objective column)
best = store.best_so_far(objective_index=0, minimize=True, phase="bo")
if best is not None:
    sample, value = best
    print(f"best sample: id={sample.id}, value={value:.4g}, inputs={sample.inputs}")

# Multi-objective non-dominated set (collapses to 1 sample for single-objective)
front = store.pareto_front(minimize=True, phase="bo")
print(f"Pareto front: {len(front)} samples")

# Or just the flat DataFrame
df = store.to_dataframe()
df = df[df["status"] == "finished"]
```

`SampleStore` instances are cheap to construct and discard. You don't
need to keep one alive — re-open in each cell if you want a fresh
read.

## Read while a study is still running

`SampleStore` uses SQLite in WAL mode (set automatically when the
study creates the database). WAL gives you concurrent reads alongside
a single writer — so a notebook can poll the store *while* the CLI
study is writing to it, without blocking the writer.

```python
import time
import pandas as pd
from IPython.display import clear_output

cfg = load_study_config("study.yaml")

for _ in range(60):                                # poll for up to 5 minutes
    store = open_store(cfg)
    df = store.to_dataframe()
    counts = df["status"].value_counts().to_dict()
    clear_output(wait=True)
    print(counts)
    if df["status"].isin(["finished", "failed", "cancelled"]).all() and len(df):
        break
    time.sleep(5)
```

You can also shell out to the verbose CLI status from a cell:

```python
!polarisopt status study.yaml --verbose
!polarisopt logs   study.yaml 42 --binary --iteration=abm_init -n 50
```

## Plotting recipes

For ready-made plots (convergence, Pareto front, Morris sensitivity,
comparing two studies), see [docs/notebooks/](../notebooks/index.md).
Each page is a self-contained recipe you can paste into your own
notebook.

## See also

- [Notebook gallery](../notebooks/index.md) — convergence, Pareto, Morris, compare-runs.
- [Getting started](../getting-started.md) — workspace conventions and a worked example.
- [Common mistakes](common-mistakes.md) — the `validate` → `plan` workflow that catches
  option typos in <1s.
- [Debug failed samples](debug-failed-samples.md) — for when something
  the notebook surfaces was actually a runtime failure.
