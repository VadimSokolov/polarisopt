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

```bash
polarisopt status study.yaml          # per-phase counts
polarisopt logs   study.yaml 42       # stdout + stderr for sample 42
polarisopt logs   study.yaml 42 -f    # tail -f the largest log
```

Conventional Slurm tools also work — every sample's Slurm jobid is in
the SampleStore (`samples.runner_task_id`):

```bash
squeue -u $USER                       # all your jobs
sacct  -j <jobid> --format=...        # accounting
scancel <jobid>                       # or use `polarisopt cancel`
```

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
