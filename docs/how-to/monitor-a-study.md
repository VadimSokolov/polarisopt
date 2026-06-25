# How to monitor a running study

You've kicked off `polarisopt run study.yaml` (or submitted the master
as a Slurm/PBS job per [Run on Slurm](run-on-slurm.md#submitting-the-master-itself-as-a-slurm-job)).
Now what? This page covers every commonly-used monitoring command +
how to interpret what you see + what to do when something looks wrong.

For *driving* the run see [the canonical loop](../operating-as-an-agent.md#the-canonical-loop).
For *post-mortem* on failures see [Debug failed samples](debug-failed-samples.md).

## Quick check (60 seconds)

```bash
polarisopt status study.yaml
```

Per-phase counts of pending / running / finished / failed. The first
thing to run when you ssh in:

```text
warmup: {finished: 16}
bo:     {finished: 12, running: 4, pending: 8}
```

The phase you're currently *in* has non-zero running. If everything's
`finished` and there's no `running`, the master has either completed
or died.

For per-sample detail:

```bash
polarisopt status study.yaml --verbose
```

```text
  id phase     status   retry jobid           runtime  folder / last log line
-----------------------------------------------------------------------------
   1 warmup    finished     -  -                12.3m  /lcrc/.../sim-000001
   2 warmup    finished     -  -                11.8m  /lcrc/.../sim-000002
  17 bo        running      -  7628354.imgt1    34.5m+ /lcrc/.../sim-000017
     └─ polaris_progress.log: sim hour 14 of 24 — events: 2.4M
```

Each running sample's `last log line` peeks at the most recent line in
the sample's logs — by default the wrapper log, but if a polarislib
progress log under `<sample>/*/log/polaris_progress.log` is more recent
(v0.13.1+), that wins. Tells you what sim-hour each binary is on
without separate `tail` calls.

Filter to one status:

```bash
polarisopt status study.yaml -v --status running
polarisopt status study.yaml -v --status failed
```

## Live tail of one sample

```bash
polarisopt logs study.yaml 17 --binary --follow --iteration=abm_init
```

Streams the POLARIS binary's `polaris_progress.log` for sample 17. The
`--iteration` filter pins to the `abm_init` directory in case a sample
produced multiple iteration dirs.

Without `--binary`, you get the polarisopt wrapper log (Python-side
errors, module-load failures). Useful for the first 2 minutes after
submission; after that the binary log is where the action is.

## Per-iteration progress (sequential / BO phases)

The `iteration` column on every sample says which batch it came from
(`0` = warm-up for sequential phases, `1..N` = BO rounds; always `0`
for static phases).

```bash
# Count of samples per iteration, by status:
polarisopt status study.yaml -v | awk '{print $2, $3}' | sort | uniq -c

# Or richer, from a notebook:
df = store.to_dataframe()
df.groupby(["phase", "iteration", "status"]).size().unstack(fill_value=0)
```

A healthy sequential study has each iteration's row going through
`pending → running → finished` in order. If iteration 3 has `finished:
14, failed: 2` and iteration 4 is `running: 14, queued: 2`, the master
is fine — it moved on after iteration 3's metrics were collected and
fed to the surrogate.

If iteration 3 has `finished: 0, running: 14` and the heartbeat log
hasn't shown a transition in over an hour, see
[Heartbeat interpretation](#heartbeat-interpretation) below.

## Best-so-far

```bash
polarisopt best study.yaml
```

```text
best sample (argmin over obj[0])
  id:        42
  phase:     bo
  iteration: 3
  inputs:    [0.234, 0.187, 1.045, ...]
  metric:    [0.0428]
  obj[0]:    0.0428  (min)
  folder:    /lcrc/.../sim-000042
```

Flags:

- `--maximize` — argmax instead of argmin
- `--objective N` — pick a column for multi-objective studies
- `--phase bo` — restrict to one phase (e.g. exclude the random
  warm-up if you only want the BO-suggested winner)
- `--json` — machine-readable, for shell pipelines

For multi-objective, you usually want the Pareto front, not a single
best:

```python
front = store.pareto_front(minimize=True)
for s in front:
    print(s.id, s.inputs, s.metric)
```

## Heartbeat interpretation

The master emits a heartbeat line every 5 min by default (configurable
via `runner.options.heartbeat_interval`):

```text
[heartbeat] 22 sample(s) outstanding after 1.4h — QUEUED=8, RUNNING=14 — +3 FINISHED, +0 FAILED since last
```

Read it like this:

- **`22 sample(s) outstanding`** — what's in the in-memory poll loop
  for the current batch
- **`after 1.4h`** — wall-clock since this batch started
- **`QUEUED=8, RUNNING=14`** — how the *runner* (PBS/Slurm) sees the
  jobs right now
- **`+3 FINISHED, +0 FAILED since last`** — what the *master*
  transitioned in the SampleStore between this heartbeat and the
  previous one

The transition deltas are the important signal:

| What you see | What it means |
|---|---|
| `+N FINISHED` non-zero | Master is doing its job. Healthy. |
| `+N FAILED` non-zero | Some samples failed — check `status -v --status failed` after a few heartbeats accumulate. |
| `+N RECOVERED` non-zero (v0.15+) | The master's in-flight disk-recovery pass harvested a sample whose runner status had gone UNKNOWN. Self-healing working as designed. |
| `0 transitions since last (master may be stalled)` | First time → wait one more heartbeat. Recurring → check the runner side (`squeue`/`qstat`) directly — the binary may be hung, the cluster may be slow, or your study has run out of work without exiting. |

v0.17+ also emits a separate WARNING line when the SampleStore has
samples in RUNNING for over 1 hour:

```text
[heartbeat] WARNING: 28 sample(s) RUNNING >1h in SampleStore: [17, 18, 19, ...]
  — consider polarisopt recover-from-disk if the master is stuck.
```

That's your canary for a stuck-master condition. If the runner says
those jobs are FINISHED but the master hasn't transitioned them, the
in-flight disk reconcile should kick in on its own — but if it
doesn't (or if the master crashed), the recovery action below is your
hammer.

## Recovery decision tree

| Symptom | Action |
|---|---|
| Master is alive, samples are moving, but one specific sample failed. | Inspect: `polarisopt logs study.yaml <id>`. Fix root cause, then `polarisopt retry-failed study.yaml --id <id> --run`. |
| Master died (SSH disconnect, login-node reaped, OOM'd) but compute jobs are still running. | `polarisopt resume study.yaml` — automatically reconciles RUNNING samples, runs in-flight disk recovery, and continues. (See [Restart and resume](../restart.md).) |
| Master died, compute jobs finished hours ago, `sacct`/`qstat` history has aged out so reconcile says "UNKNOWN" for everything. | `polarisopt recover-from-disk study.yaml` — sweeps RUNNING + FAILED samples and harvests any whose outputs are on disk regardless of what the runner remembers. |
| Master is alive, heartbeat shows `0 transitions since last (master may be stalled)` for >30 min. | First: is the binary actually progressing? Check the binary log: `polarisopt logs study.yaml <id> --binary --follow`. If sim-hour is advancing, the master is just patient. If it's hung at the same line, the binary is stuck — `polarisopt cancel study.yaml <id>` and `retry-failed --id <id> --run`. |
| Stale-running WARNING in heartbeat. | Same as above — usually means binaries are hung, occasionally means master can't see PBS history. `recover-from-disk` is safe to try; if it harvests, the master will keep going. |
| Want to add `--force` to a recovery command. | Don't unless you've read the error. `--force` bypasses the config-drift safety check; in v0.17+ the error includes a field-level diff (see [config drift](debug-failed-samples.md#config-drift-on-retry)) telling you exactly which YAML field changed since the existing samples ran. |

## Notebook dashboard (live, no polling burden)

For a long study, attaching a notebook to the SampleStore (which is
SQLite-WAL — concurrent reads are safe) gives you a richer view than
the CLI:

```python
import time
import pandas as pd
from IPython.display import clear_output

from polarisopt.config import load_study_config
from polarisopt.samples.store import SampleStore
from polarisopt.utils.paths import workspace_layout

cfg = load_study_config("study.yaml")

while True:
    store = SampleStore.open(workspace_layout(cfg.workspace)["db"], cfg.name)
    df = store.to_dataframe()
    clear_output(wait=True)

    # Phase × status grid
    print(df.groupby(["phase", "status"]).size().unstack(fill_value=0))

    # Best-so-far metric line per iteration
    finished = df[df["status"] == "finished"].copy()
    if len(finished):
        finished["obj0"] = finished["metric"].apply(lambda m: m[0] if m else None)
        best = finished.groupby(["phase", "iteration"])["obj0"].min()
        print("\nBest per iteration:")
        print(best.to_string())

    # Stale-running canary
    running = df[df["status"] == "running"]
    if len(running):
        ages = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(running["updated_at"]))
        stale = running[ages > pd.Timedelta(hours=1)]
        if len(stale):
            print(f"\n⚠️  {len(stale)} sample(s) RUNNING >1h: {stale['id'].tolist()}")

    time.sleep(60)
```

Each iteration of the loop re-opens the store (cheap) and gets a fresh
view. The master continues to write to the same DB; nothing in this
loop blocks it.

For a plot-driven dashboard, see the patterns in
[`docs/notebooks/convergence.md`](../notebooks/convergence.md) and
[`docs/notebooks/pareto-front.md`](../notebooks/pareto-front.md).

## What "completed" means at study level

Different definitions, depending on what you're asking:

- **Phase completed** — every sample in the phase reached a terminal
  state (FINISHED / FAILED / CANCELLED). Visible in `status` as
  zero in `pending` and `running` for that phase.
- **Static phase completed** — design's N samples all terminal.
- **Sequential phase completed** — stopping criterion fired
  (`max_iter`, `epsilon`, `plateau`, `hypervolume`). The phase's
  generator marks no further work needed.
- **Study completed** — every phase completed and the `polarisopt run`
  / `resume` command exited cleanly with
  ``completed: N/N samples (failed: M)``.

If `polarisopt run` is still in the foreground waiting, the study
isn't done. Don't `pkill` it — let it finish or `Ctrl-C` cleanly so
the in-flight samples get cancelled (they will, via v0.5's graceful
shutdown).

## See also

- [Operating polarisopt as an agent](../operating-as-an-agent.md) —
  the full canonical loop, not just monitoring
- [Use polarisopt from a notebook](use-from-notebook.md) — the
  programmatic API behind everything above
- [Debug failed samples](debug-failed-samples.md) — when monitoring
  surfaces a failure
- [Restart and resume](../restart.md) — recovery internals
- [Run on Slurm](run-on-slurm.md) /
  [Run on PBS](run-on-pbs.md) — scheduler-native tools
  (`squeue`/`qstat`) you can use alongside polarisopt
