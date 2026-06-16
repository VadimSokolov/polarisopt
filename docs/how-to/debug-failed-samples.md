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
| `polaris.stdout.log` | polarisopt wrapper (Python runner) | runner Python tracebacks, missing-module errors |
| `polaris.stderr.log` | wrapper stderr | "file not found", schema errors at init |
| `slurm-<jobid>.out` | sbatch wrapper | OOM, time-limit |
| `<jobname>.slurm` | the generated sbatch script | resource directives that look wrong |

For long files, page or tail:

```bash
polarisopt logs study.yaml 42 -n 200       # last 200 lines per file
polarisopt logs study.yaml 42 --follow      # stream live
```

For `polaris_convergence` simulators, the wrapper logs above are only
the Python runner's view. The POLARIS C++ binary writes its own
progress log to `<output_dir>/log/polaris_progress.log` — that's the
one that tells you what sim-hour the run is in:

```bash
polarisopt logs study.yaml 42 --binary --follow
```

If the sample produced multiple iteration dirs (e.g. `abm_init` + a
`normal_iteration_1`), pin the tail to one:

```bash
polarisopt logs study.yaml 42 --binary --iteration=abm_init
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

```bash
polarisopt retry-failed study.yaml --run
```

Flips every FAILED → PENDING in the store and immediately re-runs the
phase to evaluate them. Target specific samples with `--id`:

```bash
polarisopt retry-failed study.yaml --id 42 --id 51 --run
```

Without `--run`, the command only flips the status — useful if you
want to inspect the store first or queue them for a later
`polarisopt resume`.

### Config drift on retry

`retry-failed` records a 16-char fingerprint of the simulator+runner
config on every sample at submit time. If you've edited `simulator.*`
or `runner.*` in the YAML since the failed samples ran, retry refuses:

```text
ConfigDriftError: simulator/runner config has changed since 3 of 5
failed sample(s) ran (recorded fingerprints: ['a1b2c3d4...']; current:
'5e6f7a8b...'). Pass force=True (CLI: --force) to retry under the new
config, or start a fresh workspace to keep the run history clean.
```

This is intentional — mixing results from different `population_scale_factor`
or `num_threads` settings in one SampleStore silently corrupts downstream
analysis. Two options:

- **`--force`** if you genuinely want to retry under the new config
  (and accept that the store now contains heterogeneous samples).
- **Fresh workspace** if you're tuning the YAML — change `workspace:`
  to a distinct path. Convention: tag the path with the scale or
  variant, e.g. `polarisopt-runs/<study>-1pc/` vs
  `polarisopt-runs/<study>-5pc/`.

Orchestrator knobs (`poll_interval`, `orphan_threshold`,
`heartbeat_interval`) are excluded from the fingerprint, so tweaking
those doesn't trigger drift.

## See also

- [CLI reference](../reference/cli.md) — full `polarisopt logs` flags
- [How-to: Run on Slurm](run-on-slurm.md) — resource tuning
- [Concept: Restart correctness](../concepts/restart-correctness.md)
