# EQSQL compatibility

Argonne users coming from `polaris.hpc.eqsql` can use
``polarisopt.compat.eqsql`` as a drop-in replacement that routes
everything through plain Slurm — no Postgres, no worker pool, no
``worker_id`` regex pinning.

The API surface mirrors the polarislib EQSQL: ``insert_task``,
``Task.from_id``, ``Task.cancel``, ``Task.get_logs``. The status strings
(``queued``, ``running``, ``finished``, ``failed``, ``cancelled``) are
byte-for-byte identical so any code that string-compares status keeps
working.

## Usage

```python
from polarisopt.compat import eqsql

with eqsql.open_queue("/path/to/workspace") as queue:
    result = queue.insert_task(
        definition={"task-type": "bash-script", "command": "/path/to/run.sh"},
        exp_id="my-experiment",
        worker_id="xover.vsokolov.*",   # accepted but unused
    )
    assert result.succeeded
    task = result.value

    while not task.is_terminal():
        task = queue.task_from_id(task.task_id)
        time.sleep(30)

    for log in queue.task_logs(task):
        print(log["created_at"], log["message"])
```

## What it does under the hood

For each ``insert_task``:

1. Parse the ``definition`` (currently only ``bash-script`` is supported).
2. Write the command to a sbatch script in ``workspace/scripts/``.
3. ``sbatch`` it; record the Slurm jobid in a SQLite-backed queue table.
4. Return a ``Task`` with status ``queued``.

For ``task_from_id`` on a non-terminal task:

1. ``squeue -h -j <jobid> -o '%T'`` for active state, or
2. ``sacct -j <jobid> -X -n -P -o State,ExitCode`` for terminal state.

For ``Task.cancel``: ``scancel <jobid>``.

For ``Task.get_logs``: read the row's ``task_log`` table entries (which
include sbatch submission + state-transition records).

## Differences from real EQSQL

| Feature | Real EQSQL | polarisopt shim |
|---|---|---|
| `task-type` | bash-script, python-script, python-module, bash-module, control-task | **bash-script only** |
| Postgres DB | Required | None — uses SQLite |
| Worker pool | Required (long-lived `worker_loop.py`) | None — direct sbatch |
| `worker_id` regex | Routes to specific workers | Stored but unused |
| Cross-user queue | Yes | No |
| Globus callbacks | Built into worker | Use `polarisopt.transfer` |

The shim is intended as a migration aid — for new code, prefer the
polarisopt CLI and Study API directly.
