# How to migrate from EQSQL

If your existing code uses `polaris.hpc.eqsql` to submit POLARIS runs,
polarisopt offers two migration paths.

## Path A — drop-in compatibility shim

Fastest. Change the import and keep the rest of your code:

```diff
- from polaris.hpc.eqsql.eq import insert_task
- from polaris.hpc.eqsql.task import Task
+ from polarisopt import eqsql_compat
```

The `polarisopt.eqsql_compat` module exposes:

| Real EQSQL | polarisopt shim |
|---|---|
| `insert_task(conn, definition, ...)` | `queue.insert_task(definition=..., ...)` |
| `Task.from_id(engine, task_id)` | `queue.task_from_id(task_id)` |
| `Task.status` | `Task.status` |
| `Task.cancel()` | `Task.cancel()` |
| `Task.get_logs(engine)` | `Task.get_logs()` / `queue.task_logs(task)` |
| Statuses `"queued"`/`"running"`/`"finished"`/`"failed"`/`"cancelled"` | byte-identical |

Under the hood the shim routes everything to `sbatch`/`squeue`/
`scancel`. No Postgres, no worker pool.

```python
from polarisopt import eqsql_compat

with eqsql_compat.open_queue("/path/to/workspace") as queue:
    result = queue.insert_task(
        definition={"task-type": "bash-script", "command": "/path/to/run.sh"},
        exp_id="my-experiment",
        worker_id="xover.vsokolov.*",   # stored but not used (no workers)
    )
    task = result.value
    while not task.is_terminal():
        task = queue.task_from_id(task.task_id)
        time.sleep(30)
```

### Limitations vs. real EQSQL

- Only `task-type: bash-script` is supported. The `python-script`,
  `python-module`, `bash-module`, and `control-task` types aren't
  ported.
- `worker_id` regex is accepted for API parity but **not used** to
  route — there's no worker pool.
- `priority` is stored but Slurm-side priority depends on your cluster's
  QOS configuration.

If you only use `bash-script`, the shim is a clean drop-in.

## Path B — new Study API (recommended for new code)

For new pipelines, replace the EQSQL queue + worker pool model entirely
with polarisopt's Study API. You get:

- YAML-driven configuration
- SampleStore with restart support
- Pluggable Design / Surrogate / Acquisition / Generator / Stop
- Polish CLI: `polarisopt run|status|resume|cancel|abort|logs`

See [Tutorial 05 · First POLARIS run](../tutorials/05-first-polaris.md).

Sketch of the migration:

| EQSQL pattern | polarisopt equivalent |
|---|---|
| One `insert_task` per design point in a notebook | YAML `phases[].design` + `polarisopt run` |
| Custom polling loop in a notebook | Master orchestrator (auto) |
| Globus copy-back in `async_end_of_loop_fn` | `PolarisSimulator.collect_output` + Transfer |
| `task_log` table | `polarisopt logs <sample_id>` |
| `EQ_ABORT` control task | `polarisopt abort study.yaml` |

The biggest mental shift: **no long-lived workers**. Each sample is a
fresh sbatch job. For DFW-class iterations (30+ min each), per-job
startup overhead is negligible.

## See also

- [EQSQL compatibility reference](../eqsql-compat.md)
- [Tutorial 05 · First POLARIS run](../tutorials/05-first-polaris.md)
