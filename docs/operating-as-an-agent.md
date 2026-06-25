# Operating polarisopt as an AI agent

If you're an AI agent driving polarisopt to run POLARIS studies for a
user — not modifying polarisopt's source — this is your playbook. It's
the canonical loop plus every foot-gun previous agents have already
hit so you don't have to re-discover them.

For *contributing* to polarisopt itself, see [AGENTS.md](../AGENTS.md)
instead.

## The canonical loop

```bash
polarisopt validate study.yaml        # <1s — schema + plugin-option typecheck
polarisopt plan     study.yaml        # ~10s — stage sample 0, render sbatch script, don't submit
polarisopt run      study.yaml        # submit + poll
polarisopt status   study.yaml -v     # per-sample table (jobid, retry count, runtime, last log line)
polarisopt logs     study.yaml 42 --binary --iteration=abm_init -n 200
polarisopt resume   study.yaml        # pick up after master death (also harvests disk artifacts)
polarisopt retry-failed study.yaml --run
polarisopt recover-from-disk study.yaml   # standalone disk sweep when sacct aged jobids out
polarisopt best     study.yaml        # argmin sample
```

Always `validate` and `plan` before submitting. Both are sub-30-second
checks; they catch a class of mistakes that otherwise burn 30+ minutes
of staging + a Slurm allocation per try.

For the *monitoring* side of the loop (interpreting `status`, the
heartbeat output, and the recovery decision tree) see
[Monitor a running study](how-to/monitor-a-study.md). That page also
documents the notebook-side live dashboard pattern for long runs.

## Workspace conventions

`workspace:` is where the SampleStore + per-sample folders live. **Tag
the path per scale/variant** — different `population_scale_factor`,
different `num_threads`, different scenario file:

```yaml
workspace: /lcrc/project/POLARIS/<user>/polarisopt-runs/<study>-<tag>/
# e.g. polarisopt-runs/calibration-1pc/ vs polarisopt-runs/calibration-5pc/
```

`retry-failed` and `resume` both record a 16-char fingerprint of the
simulator+runner config and refuse to mix runs across edited YAMLs.
Use one workspace per variant and drift detection stays out of your
way. `--force` overrides but produces a heterogeneous store that's
hard to analyze cleanly later.

## The typo class

This is the most expensive class of mistake. Catch it before any
compute is burned by running `polarisopt validate`:

| You wrote | Real arg | Plugin |
|---|---|---|
| `distance: l1` | `aggregation: l1` | `ChoiceShareMetric` |
| `sim_key: demand_db` | `source_key: demand_db` | `ChoiceShareMetric` |
| `keys_to_extract: value` | `keys: value` | `IdentityMetric` |
| `population_scal_factor: 0.05` | `population_scale_factor: 0.05` | `polaris_convergence` runner_options |
| `num_threds: "16"` | `num_threads: "16"` | `polaris` / `polaris_convergence` |

v0.8+ `polarisopt validate` does an MRO-walking signature check
against each plugin's `__init__` and reports unknown keys in <1s.

For `polaris_convergence` `runner_options` (forwarded to polarislib's
ConvergenceConfig), `polarisopt plan` cross-checks against the
`KNOWN_RUNNER_OPTIONS` whitelist. Same workflow, deeper coverage.

See [common-mistakes](how-to/common-mistakes.md) for the full
recipe.

## Three retry paths — pick the right one

| Path | When it fires | When to use |
|---|---|---|
| **`max_retries: N` in `runner.options`** | Automatic, inside `_evaluate_batch`. Each FAILED transition increments `sample.extra["retry_count"]`; sample re-submitted until `retry_count >= max_retries`. | Transient failures — node failure, occasional OOM on a contended node, slurm hiccups. Default 0 (off). |
| **`polarisopt retry-failed --run`** | Manual. Flips FAILED → PENDING; re-runs the phase. | Permanent failures that you fixed (e.g. patched a runner script, freed disk space, bumped time-limit). |
| **`polarisopt recover-from-disk`** / disk-first `reconcile_running` | Automatic on `resume`. Tries `simulator.collect_output` + `metric.compute` against the workspace; FINISHED if outputs parse. | Master died after compute jobs completed and wrote artifacts — but slurm forgot the jobid (sacct GC) so retry would re-submit. The disk wins. |

These are stacked: `resume` first reconciles (consults runner + tries
disk), then sweeps with `recover_from_disk`, then `StudyRunner.run()`
re-submits any still-PENDING samples (including the ones `max_retries`
flipped back).

## Master-death recovery

Master process (Python orchestrator) is mortal — SSH disconnect, tmux
reaping, login-node idle cleanup, CI timeout. The compute jobs keep
running and write outputs to disk; the master is the only thing that
calls `collect_output` and writes the metric back.

**Prevention.** For studies >2h, submit the master itself as a Slurm
job with `--oversubscribe --mem=2G --cpus-per-task=1 --time=2-00:00:00`.
Wrapper sbatch script in
[run-on-slurm.md](how-to/run-on-slurm.md#submitting-the-master-itself-as-a-slurm-job).

**Recovery (post-fact).** `polarisopt resume study.yaml`. v0.10.1+'s
disk-first reconcile + the v0.11.0 `recover_from_disk` pass make this
mostly automatic — zombies whose outputs parse become FINISHED with
the metric set, without you needing to know `recover-from-disk` exists
as a separate verb.

## Workspace lock (v0.12.1+)

`polarisopt run` and `polarisopt resume` acquire an exclusive
`flock(2)` on `<workspace>/.polarisopt.lock` for the duration of the
master process. If another master already holds it, you get a
fail-fast error pointing at the holder's PID, hostname, start time,
and polarisopt version. This prevents two orchestrators from racing
on the same SampleStore — the failure mode where one master submits
sample N while the other is also submitting it, both call
`collect_output`, and state thrashes.

The lock is kernel-managed and auto-releases on process death — no
stale-state cleanup needed. The metadata sidecar (`.polarisopt.lock.meta`)
is best-effort cleaned on graceful exit; a stale metadata file
alongside a free lock is benign.

`--force` on `run` / `resume` bypasses the check. Use only when you
knowingly accept the racing-masters consequences (or in rare
filesystems where flock is unreliable — LCRC's GPFS is reliable; NFS
varies).

The short-lived mutators (`cancel`, `abort`, `retry-failed`,
`recover-from-disk`) do **not** acquire the lock — they're typically
operator interventions that should run anytime, including while a
master is alive.

## Heartbeat output

For studies that take more than ~5 minutes, the master emits a
periodic `[heartbeat] N sample(s) outstanding after T — RUNNING=K,
QUEUED=L` log line so you know it's alive. Default interval 300s.

For 25h studies this dominates the log (~300 heartbeats). Pass
`--quiet-heartbeat` on `run` / `resume` to filter heartbeats out of
the default INFO output. State transitions still log normally.

```bash
polarisopt run study.yaml --quiet-heartbeat
```

Configure the interval per study via `runner.options.heartbeat_interval`.

## What does NOT belong in YAML

These are orchestrator-only knobs — they live in `runner.options` but
get popped by `StudyRunner` before the runner is constructed:

- `poll_interval` — seconds between `runner.status()` calls (default 5s)
- `orphan_threshold` — consecutive UNKNOWN polls before FAILED (default 3)
- `heartbeat_interval` — seconds between heartbeat lines (default 300)
- `max_retries` — auto-retry budget per sample (default 0)

`polarisopt validate` allowlists these so they don't show up as
unknown-option warnings.

## Don't edit polarisopt source

A previous agent landed an inline patch to `simulator/polaris.py` to
work around a bug, then a polarisopt release shipped a different fix
and the user got a merge conflict that took an hour to untangle. Since
then: **don't edit polarisopt source. File feedback for the polarisopt
maintainer agent instead.**

If you find a bug:
1. Confirm by reading the relevant module — sometimes the fix already
   exists on a later version.
2. Identify the smallest YAML / minimal repro.
3. Pass the feedback up to the user verbatim. The polarisopt maintainer
   agent will triage, fix, and ship.

If you need a workaround in the meantime, build it *outside*
polarisopt — a wrapper script, a pre-processing step, a different
YAML, a notebook cell. Don't fork the library.

## What's safe to do without asking

- Read polarisopt source (`src/polarisopt/`), the SampleStore SQLite
  file, the test suite, the how-to docs.
- Write study YAMLs.
- Call the CLI (`polarisopt validate / plan / run / status / logs /
  resume / retry-failed / recover-from-disk / best / smoke-test`).
- Open the SampleStore from a notebook (it's WAL — concurrent reads
  alongside a live master are safe).
- Stage / re-stage model directories under the workspace.

## What to ask before doing

- `--force` flags on `retry-failed` / `resume` (overrides the
  config-drift safety check).
- `polarisopt abort study.yaml` (cancels every non-terminal sample;
  destructive).
- `polarisopt cancel study.yaml <id>` (cancels one sample; less
  destructive but still user-visible).
- Editing the workspace path mid-study.
- Editing `simulator` / `runner` / `metric` blocks of a YAML that
  already has FINISHED samples in its store (will trip the drift
  check).

## Feedback format

When something doesn't fit the canonical loop, pass it up in this shape:

1. **The minimum repro.** YAML excerpt + exact command.
2. **What you expected.** "I expected `polarisopt plan` to fail-fast on
   the typo, but it accepted the YAML and the failure showed up after
   30s of staging."
3. **What you observed.** Error message, exit code, sample state at
   the time of the issue.
4. **Severity from your perspective.** Burned compute? Stalled a
   study? Mild annoyance?
5. **What you tried as a workaround.** So the maintainer agent
   doesn't suggest something you already tried.

Previous high-leverage feedback that shaped the library:

- v0.9.0 (SIF wrapping): "10/10 samples failed in 3s — the binary
  printed its boot banner then said 'No such scenario'"
- v0.10.1 (disk recovery): "master died, 8 zombies in RUNNING forever,
  the data is right there on disk"
- v0.11.0 (resume reliability): "polarisopt run accepted poll_interval,
  resume crashed three sessions later with TypeError"

Each of those changed a class of pain into a non-event. Be specific.

## See also

- [Getting started](getting-started.md) — first study end-to-end
- [Common mistakes](how-to/common-mistakes.md) — the typo class
- [Debug failed samples](how-to/debug-failed-samples.md) — when
  things go wrong
- [Use polarisopt from a notebook](how-to/use-from-notebook.md) —
  programmatic API
- [Run on Slurm](how-to/run-on-slurm.md) — including the
  master-as-slurm-job pattern
- [Migrate from EQSQL](how-to/migrate-from-eqsql.md) — if you're
  porting an existing pipeline
