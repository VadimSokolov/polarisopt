# How to run on PBS Pro

For clusters that use PBS Professional instead of Slurm. LCRC users
hit this on **Improv** (the user's secondary calibration target) and
**Bebop**. YAML usage is identical to Slurm; only `runner.type` and
the resource field names change.

## Minimal YAML

```yaml
runner:
  type: pbs
  options:
    default_resources:
      queue: compute
      account: POLARIS
      walltime: "04:00:00"
      select: 1
      ncpus: 16
      mem: 96gb              # lowercase — PBS Pro convention
      place: excl            # whole-node — parallel to Slurm --exclusive
      setup_commands:
        - "module load apptainer/1.1.9 miniforge3"
```

## Full `default_resources`

| Field | Maps to | Notes |
|---|---|---|
| `queue` | `#PBS -q <queue>` | LCRC Improv: `compute`, `bigmem`, `debug`, `routing_queue` |
| `account` | `#PBS -A <account>` | LCRC Improv: `POLARIS` |
| `walltime` | `#PBS -l walltime=…` | **Not** `time` — PBS uses `walltime` |
| `select` | `#PBS -l select=N:…` | Number of chunks; default 1 |
| `ncpus` | `#PBS -l select=N:ncpus=M` | CPUs per chunk |
| `mpiprocs` | `#PBS -l select=N:mpiprocs=…` | Optional |
| `mem` | `#PBS -l select=N:mem=…` | **Lowercase** units (`96gb`, not `96GB`) |
| `place` | `#PBS -l place=…` | `excl` (whole-node), `shared` (pack), `free` |
| `join_output` | `#PBS -j oe` | Default `true` — joins stdout+stderr |
| `extra_directives` | raw `#PBS …` lines | For `-W`, `-m`, `-M`, `-r`, etc. |
| `setup_commands` | shell lines after directives | `module load` and friends |

Polarisopt-side knobs (`poll_interval`, `orphan_threshold`,
`heartbeat_interval`, `max_retries`) live in `runner.options`
identically across Slurm and PBS — the orchestrator pops them before
constructing the runner.

## What an sbatch script looks like (well, qsub)

For a sample with the YAML above, polarisopt writes
`<workspace>/experiments/sim-NNNNNN/<jobname>.pbs`:

```bash
#!/bin/bash
#PBS -N polaris-sample-1
#PBS -q compute
#PBS -A POLARIS
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=16:mem=96gb
#PBS -l place=excl
#PBS -j oe
#PBS -o /lcrc/.../experiments/sim-000001/polaris.stdout.log

set -euo pipefail

export POLARIS_NUM_THREADS=16

module load apptainer/1.1.9 miniforge3

cd /lcrc/.../experiments/sim-000001
apptainer run -B ... polaris.sif Integrated_Model scenario_abm.json 16
```

PBS directives must precede the first executable line, same rule as
Slurm's `#SBATCH`.

## Monitoring

Conventional PBS tools work — every sample's PBS jobid
(`<number>.<hostname>`, e.g. `7609762.imgt1`) is in the SampleStore
as `samples.runner_task_id`:

```bash
qstat -u $USER                        # all your jobs
qstat -fx <jobid>                     # full info, including terminated jobs
qdel <jobid>                          # or use `polarisopt cancel`
```

`qstat -fx` is the PBS equivalent of Slurm's `sacct` — the `-x` flag
includes history for completed jobs, which `qstat <jobid>` (without
`-f -x`) won't show after job termination.

For polarisopt-side monitoring — `status -v`, the binary's
`polaris_progress.log`, heartbeat interpretation, and the
recovery-decision tree — see
[Monitor a running study](monitor-a-study.md).

## Slurm-to-PBS translation table

| Slurm | PBS Pro |
|---|---|
| `sbatch script.sh` → `Submitted batch job NNNN` | `qsub script.sh` → `NNNN.hostname` |
| `squeue -h -j NNN -o %T` | `qstat -f -x NNN \| grep job_state` |
| `sacct -j NNN -X -n -P -o State,ExitCode` | `qstat -fx NNN` (history) |
| `scancel NNN` | `qdel NNN` |
| `#SBATCH --partition=X` | `#PBS -q X` |
| `#SBATCH --account=X` | `#PBS -A X` |
| `#SBATCH --time=HH:MM:SS` | `#PBS -l walltime=HH:MM:SS` |
| `#SBATCH --nodes=N --cpus-per-task=M` | `#PBS -l select=N:ncpus=M` |
| `#SBATCH --mem=NG` | `#PBS -l select=1:mem=Ngb` |
| `#SBATCH --output=path` | `#PBS -o path` |
| `#SBATCH --error=path` | `#PBS -e path` (or `-j oe` to join) |
| `#SBATCH --exclusive` | `#PBS -l place=excl` |
| `#SBATCH --oversubscribe` | `#PBS -l place=shared` |

## Status mapping

| PBS `job_state` | polarisopt `JobStatus` |
|---|---|
| `Q` (queued) | `QUEUED` |
| `H` (held) | `QUEUED` |
| `R` (running) | `RUNNING` |
| `E` (exiting) | `RUNNING` |
| `C` / `F` (completed) with `exit_status == 0` | `FINISHED` |
| `C` / `F` with `exit_status != 0` | `FAILED` (seg fault, OOM, kill) |
| `S` (suspended), no state, etc. | `UNKNOWN` |

The `exit_status` check matters: PBS marks the job `F` regardless of
whether the binary succeeded or seg-faulted. Without checking
`exit_status`, every crashed sample would look FINISHED to
polarisopt.

## LCRC Improv specifics (verified 2026-06-17)

```yaml
runner:
  type: pbs
  options:
    default_resources:
      queue: compute        # main queue; use `debug` for tutorials
      account: POLARIS
      walltime: "04:00:00"
      ncpus: 16
      mem: 96gb
      place: excl
      setup_commands:
        - "module load gcc/11.4.0 hdf5/1.14.2-gcc-11.4.0 miniforge3 libspatialite apptainer"
```

The module-load line matches `polarislib/bin/hpc/worker_loop_lcrc.sh`'s
Improv branch — same modules the existing EQSQL workers use.

Job IDs come back as `<number>.<host>` (e.g. `7609762.imgt1`).
polarisopt stores and forwards the full string — `qstat` and `qdel`
need the host suffix.

## Common failure modes

- **`qsub: Invalid credential`** — your account or queue is wrong.
  Check `qstat -Qf <queue>` for queue policy; `qmgr -c "list account"`
  for valid accounts.
- **`PBS Error: Resource temporarily unavailable`** — queue limits.
  Reduce `batch_size` or wait.
- **Job stays in `Q` for hours** — usually queue priority. Check
  `qstat -Q` for queue depth; `qstat -fx <jobid>` for `comment` field
  explaining the hold.
- **`exit_status = 137`** — SIGKILL, almost always OOM. Bump `mem`.
- **`exit_status = 271`** — walltime exceeded. Bump `walltime`.

See [Debug failed samples](debug-failed-samples.md) for the deeper
diagnostic loop; works the same way on PBS as Slurm.

## See also

- [Run on Slurm](run-on-slurm.md) — for cluster-side details that
  apply identically (master-as-Slurm-job pattern, workspace lock,
  retry policy)
- [SlurmRunner API](../reference/api/runners/slurm.md)
- [Tutorial 05 · First POLARIS run](../tutorials/05-first-polaris.md)
