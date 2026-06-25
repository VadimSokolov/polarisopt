# How to run on Slurm

polarisopt's `runner.type: slurm` submits one Slurm job per sample.
The master Python process keeps polling `squeue`/`sacct` until each
job is terminal.

## Minimal YAML

```yaml
runner:
  type: slurm
  options:
    default_resources:
      partition: bdwall
      account: POLARIS
      time: "02:00:00"
      nodes: 1
      cpus_per_task: 16
      mem: 64G
```

These resources apply to every `sbatch` unless a `JobSpec` overrides
them via `extra["resources"]` (rarely needed from YAML).

## Full `default_resources`

| Field | sbatch flag | Notes |
|---|---|---|
| `partition` | `--partition` | Required on most clusters. |
| `account` | `--account` | Required at ANL clusters. |
| `time` | `--time` | `HH:MM:SS` or `D-HH:MM:SS`. |
| `nodes` | `--nodes` | Usually `1` for single-task POLARIS. |
| `ntasks` | `--ntasks` | Defaults are usually fine. |
| `cpus_per_task` | `--cpus-per-task` | Match `num_threads` you pass to POLARIS. |
| `mem` | `--mem` | DFW-class models want 60+ GB. |
| `extra_directives` | extra `#SBATCH ...` lines | e.g. `--qos=high`, `--mail-type=END`. |

Example with the long tail:

```yaml
default_resources:
  partition: bdwall
  account: POLARIS
  time: "06:00:00"
  cpus_per_task: 32
  mem: 128G
  extra_directives:
    - "#SBATCH --qos=high"
    - "#SBATCH --constraint=ic"
```

## Study-level Slurm knobs

These live alongside `default_resources` because they affect how the
master *talks to* Slurm, not the submission itself:

| YAML field | Default | Meaning |
|---|---|---|
| `poll_interval` | 5 (seconds) | How often the master polls `squeue`/`sacct`. |
| `orphan_threshold` | 3 | Consecutive UNKNOWN polls before a sample is force-FAILED. |

Set `orphan_threshold: 0` to disable orphan detection (master polls
forever on UNKNOWN). Useful on flaky clusters where `sacct` lags badly.

## What an sbatch script looks like

Submitted automatically per sample. Saved to
``<workspace>/experiments/sim-<id>/<name>.slurm`` for inspection:

```bash
#!/bin/bash
set -euo pipefail
#SBATCH --job-name=polaris-sample-42
#SBATCH --partition=bdwall
#SBATCH --account=POLARIS
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/.../sim-000042/polaris.stdout.log
#SBATCH --error=/.../sim-000042/polaris.stderr.log

export POLARIS_NUM_THREADS=16
cd /.../sim-000042
/lcrc/.../Integrated_Model.sif /.../scenario_abm.json 16
```

## Monitoring

Conventional Slurm tools work — every sample's Slurm jobid is in
the SampleStore (`samples.runner_task_id`):

```bash
squeue -u $USER                       # all your jobs
sacct  -j <jobid> --format=...        # accounting
scancel <jobid>                       # or use `polarisopt cancel`
```

For polarisopt-side monitoring — `status -v`, the binary's
`polaris_progress.log`, heartbeat interpretation, and the
recovery-decision tree — see
[Monitor a running study](monitor-a-study.md).

## Submitting the master itself as a Slurm job

By default the polarisopt master is a Python process running in your
shell (login node, terminal, notebook, CI runner). If that controlling
shell dies — SSH disconnect, `tmux` reaped, CI step timeout, login-node
idle cleanup — the master dies too. The compute jobs you've already
sbatch'd keep running and write artifacts to disk, but no one is
polling slurm, calling `collect_output`, or generating the next BO
batch. The study stalls until you manually `polarisopt resume`.

For long studies (sequential BO loops, large LHS sweeps, anything
running more than a few hours), submit the master *itself* as a slurm
job. The master is lightweight — typically 1 CPU, <500 MB RAM, mostly
idle polling — so a single oversubscribed core is enough.

### Wrapper sbatch script

```bash
#!/bin/bash
# submit_polarisopt_master.sh <yaml> [run|resume]
YAML=$(realpath "${1:?usage: $0 <yaml> [run|resume]}")
ACTION="${2:-run}"
STEM=$(basename "$YAML" .yaml)
LOG=/path/to/logs/${STEM}.master.%j.log

sbatch \
    --job-name="popt-${STEM}" \
    --partition=<your_partition> \
    --account=<your_account> \
    --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem=2G --oversubscribe \
    --time=2-00:00:00 \
    --output="$LOG" --error="$LOG" \
    --wrap "
        module load <your_env_modules>
        polarisopt ${ACTION} '${YAML}'
    "
```

`--oversubscribe` asks slurm to pack the master onto a partially-used
node alongside other workloads, which on most partitions means it
doesn't consume an exclusive allocation. If your partition policy is
`OverSubscribe=EXCLUSIVE` the flag is ignored — you'll get a full
node, but the cost is still bounded by the master's tiny actual usage.

### What this fixes

- **SSH/tmux death.** Master runs inside slurm's process tree, not the
  user's shell tree. Killing the launching terminal has no effect.
- **Login-node idle cleanup.** Many clusters reap long-running processes
  on login nodes. Slurm jobs are exempt.
- **Restart hygiene.** If the master itself crashes (e.g., a network
  blip kills its `sacct` query), the slurm job stays "running" and you
  can re-exec inside the allocation: `srun --jobid=<jobid>
  polarisopt resume <yaml>`. Or just resubmit the master script —
  resume will reconcile.

### Recovering after a master crash (v0.10.1+)

If the master dies after compute jobs have completed and the run
artifacts are sitting on disk, `polarisopt resume` and
`polarisopt recover-from-disk` will harvest them. Both call
`simulator.collect_output` + `metric.compute` against each previously-
RUNNING sample's workspace and flip them to FINISHED with the metric
populated, regardless of whether `sacct` still remembers the jobids
(disk artifacts are ground truth — they beat the runner's verdict).

- **`polarisopt resume study.yaml`** — does this as part of the
  reconcile step before continuing the run.
- **`polarisopt recover-from-disk study.yaml`** — standalone sweep;
  use when reconcile can't help (e.g. `sacct` retention has aged the
  jobids out so the runner says UNKNOWN for everything).
  `--include-cancelled` to also harvest cancelled samples.

Active jobs (runner says RUNNING/QUEUED) are left alone to avoid
racing a partial write. Cancelled samples are preserved as cancelled.

### When NOT to do this

For quick interactive work — notebooks, debugging a YAML, a
5-minute branin demo — running the master in your shell is simpler
and gives you immediate access to its log output. Only switch to
slurm-submitted masters when the study's wall time exceeds the
session lifetime you can guarantee.

## Common failure modes

- **OOM at start of POLARIS**: bump `mem`. DFW-class models routinely
  need 60–120 GB.
- **Time-limit hit**: bump `time`, or shrink `num_threads`/`batch_size`
  if you're contention-bound.
- **QOSGrpJobsLimit**: too many in-flight at once. Reduce
  `batch_size` so the master doesn't submit a wave that exceeds your
  QOS-allowed concurrent count.
- **Jobs vanish from `squeue` without writing output**: orphan
  detection will eventually mark them FAILED. Inspect
  `polaris.stderr.log` for the root cause.

## See also

- [SlurmRunner API](../reference/api/runners/slurm.md)
- [Tutorial 05 · First POLARIS run](../tutorials/05-first-polaris.md)
- [How-to: Debug failed samples](debug-failed-samples.md)
