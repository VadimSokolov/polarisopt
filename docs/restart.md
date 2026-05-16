# Restart and resume

A long-running calibration is going to be interrupted at some point —
the master process gets OOM-killed, the user Ctrl-C's, the head node is
rebooted. ``polarisopt`` is built around that assumption.

## What gets persisted

Every Sample's lifecycle goes through the **SampleStore** (SQLite at
``<workspace>/polarisopt.db``). The store is the only source of truth.

| Table         | Holds                                                            |
|---------------|------------------------------------------------------------------|
| `studies`     | name, config_yaml, created_at                                    |
| `samples`     | one row per evaluation, with phase / iteration / inputs / metric / status / runner_task_id |
| `phase_state` | per-phase checkpoint: iteration counter + pickled RNG state      |

WAL mode is enabled on every connection so a partial write can't
corrupt the DB. Each state transition writes a fresh row via
`SampleStore.update` so there's no risk of losing per-sample progress
between checkpoints.

## How sequential phases checkpoint

At the end of every BO iteration, ``SequentialDesignStudy`` calls
``store.save_phase_state(phase, iteration, rng_state)``. On the next
``polarisopt run`` (or ``polarisopt resume``) the study:

1. Loads the latest ``phase_state`` row for the phase.
2. Restores the RNG state into the runtime ``np.random.Generator``.
3. Sets the iteration counter to the checkpointed value.
4. Refits the surrogate from the finished samples in the store.
5. Resumes the loop.

The surrogate itself is **not** serialized — it's deterministically
refit from the SampleStore on resume.

## Static phases

Static phases don't have an iteration loop; instead, ``polarisopt run``
detects any ``status=PENDING`` rows for the phase and evaluates them
instead of regenerating. So if the master dies after submitting a
design but before evaluating it, just re-run.

## What if a Slurm job is still alive

Each ``samples`` row stores ``runner_task_id`` (the Slurm jobid). On
resume the orchestrator queries Slurm for that jobid and:

- If still running → wait for it to finish.
- If terminal-success → collect output + compute metric.
- If terminal-failure → mark the sample failed.
- If unknown to Slurm (e.g. sacct has aged out) → mark unknown; user
  intervenes.

## CLI

```bash
polarisopt run study.yaml      # initial run (or resume — same code path)
polarisopt resume study.yaml   # alias for run, intended for clarity
polarisopt status study.yaml   # per-phase counts of pending/running/finished/failed
```

There is no separate "checkpoint" or "restore" command — restart is
just running the same YAML again.
