# How to debug failed samples

When `polarisopt status study.yaml` shows `failed: N > 0`, here's the
debugging path.

## 1. Find the failed sample IDs

```python
from polarisopt.config import load_study_config
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.utils.paths import workspace_layout

cfg = load_study_config("study.yaml")
store = SampleStore.open(workspace_layout(cfg.workspace)["db"], cfg.name)
for s in store.list(status=SampleStatus.FAILED):
    print(s.id, s.iteration, s.message, s.folder)
```

## 2. Read each sample's logs

```bash
polarisopt logs study.yaml 42
```

Prints every `*.log` / `*.out` / `*.err` file in the sample folder. The
usual suspects:

| File | Source | Look for |
|---|---|---|
| `polaris.stdout.log` | POLARIS C++ output | progress lines, "iteration N of M" |
| `polaris.stderr.log` | POLARIS C++ errors | segfaults, "file not found", schema errors |
| `slurm-<jobid>.out` | sbatch wrapper | OOM, time-limit |
| `<jobname>.slurm` | the generated sbatch script | resource directives that look wrong |

For long files, page or tail:

```bash
polarisopt logs study.yaml 42 -n 200       # last 200 lines per file
polarisopt logs study.yaml 42 --follow      # stream live
```

## 3. Common failure signatures

### "killed" / SIGKILL / OOM

The kernel OOM-killed POLARIS. Bump `runner.options.default_resources.mem`:

```yaml
runner:
  type: slurm
  options:
    default_resources:
      mem: 128G   # was 64G
```

### "TIME LIMIT" in slurm output

Bump `time`:

```yaml
default_resources:
  time: "06:00:00"   # was 02:00:00
```

### "Schema error" / "Location.dir" / migration mismatch

Your POLARIS binary and your model's SQLite schema disagree. Either:

- Update the binary to match the migration in the model, or
- `polaris check`/`polaris upgrade` the model against the binary's
  supported migrations.

This is the same skew documented in the DOE_RUNBOOK Lesson 2.

### "Job orphaned (Slurm lost track of jobid=…)"

`squeue` and `sacct` both stopped reporting the job. Possible causes:

- Job genuinely finished but sacct hasn't caught up — bump
  `runner.options.orphan_threshold` to be more patient (e.g. 10).
- Sysadmin killed the job out-of-band — check `sacctmgr` / cluster
  notifications.
- Slurm controller hiccup — usually transient; the sample retries on
  resume.

### "metric failed: ..."

The Slurm job finished but `Metric.compute` couldn't read its outputs.
Look at the post-evaluation logs:

```bash
ls -la /path/to/study/experiments/sim-000042/
```

Common issues:

- `DFW-Result.h5` missing — POLARIS crashed silently mid-run; the
  `polaris.stderr.log` will show why.
- HDF5 schema doesn't match what `link_moe` expects — POLARIS output
  layout may have changed; update the metric or the target schema.

## 4. Reproduce one sample locally

The generated sbatch script is at
`<workspace>/experiments/sim-<id>/<jobname>.slurm`. To reproduce
outside Slurm, drop the `#SBATCH` directives and run the script body
directly:

```bash
cd /path/to/study/experiments/sim-000042
bash <jobname>.slurm   # ← will skip the #SBATCH lines
```

For deeper instrumentation, add `set -x` at the top of the script or
prefix the binary with `strace`/`ltrace`/`gdb`.

## 5. Retry just the failed samples

There's no "retry-failed" subcommand in v0.2 (it's planned). Workaround:
manually flip FAILED → PENDING in the store, then resume:

```python
from polarisopt.samples.sample import SampleStatus
from polarisopt.samples.store import SampleStore
from polarisopt.utils.paths import workspace_layout

store = SampleStore.open(workspace_layout("/path/to/ws")["db"], "study-name")
for s in store.list(status=SampleStatus.FAILED):
    s.status = SampleStatus.PENDING
    s.message = (s.message or "") + " | retry"
    store.update(s)
```

```bash
polarisopt resume study.yaml
```

## See also

- [CLI reference](../reference/cli.md) — full `polarisopt logs` flags
- [How-to: Run on Slurm](run-on-slurm.md) — resource tuning
- [Concept: Restart correctness](../concepts/restart-correctness.md)
