# Changelog

Notable changes per release. Format inspired by [Keep a Changelog](https://keepachangelog.com/).

## 0.17.3 — 2026-06-25

Hot fix: wheel build was broken. Downstream users hitting
`pip install .` or `pip install <git-url>` saw
`metadata-generation-failed`; only editable installs (`pip install -e`)
worked. Merged from a fix branch pushed by a sibling agent.

### Bug fix

- **Remove redundant `[tool.hatch.build.targets.wheel.force-include]`
  entry.** It mapped `src/polarisopt/examples → polarisopt/examples`,
  but that directory already ships via `packages = ["src/polarisopt"]`
  (hatchling includes non-Python data files inside packages). The
  duplicate made hatchling refuse:
  ```
  ValueError: A second file is being added to the wheel archive at
  the same path: polarisopt/examples/__init__.py
  ```
  Removed the force-include (and the empty shared-data table). Verified
  the built wheel still contains all four bundled example YAMLs +
  `examples/__init__.py`, no duplicates.

## 0.17.2 — 2026-06-25

Docs-only release. Consolidates the monitoring story — previously
scattered across `operating-as-an-agent.md`, `run-on-slurm.md`,
`use-from-notebook.md`, `debug-failed-samples.md`, `restart.md`, plus
inferred-from-changelog material about the v0.15 heartbeat deltas
and v0.17 stale-running canary.

### Documentation

- **New `docs/how-to/monitor-a-study.md`.** Covers: quick check
  with `status` + `status -v`, live tail of the binary's
  `polaris_progress.log`, per-iteration progress on sequential/BO
  phases, `polarisopt best` (single-obj) + `pareto_front` (multi-obj),
  heartbeat interpretation table (what `+N FINISHED since last`
  means, when "0 transitions since last" is a stall), the v0.17
  stale-running WARNING, a recovery decision tree mapping symptoms
  to `retry-failed` / `resume` / `recover-from-disk`, and a
  copy-pasteable notebook live-dashboard loop.
- **Cross-links.** `operating-as-an-agent.md` (after the canonical
  loop block), `run-on-slurm.md` and `run-on-pbs.md` (Monitoring
  sections now hand off to the new page after the scheduler-native
  `squeue`/`qstat` recipes), `how-to/index.md` (new entry between
  Run-on-Slurm and Debug-failed-samples).

## 0.17.1 — 2026-06-25

Hot fix. v0.17 introduced a regression that interacts badly with
v0.16's `cleanup_on_success` hook. The DFW DOE agent reported iter
2 of a BO loop losing 12 of ~32 samples: each marked FAILED with
``metric failed: scenario file missing in /lcrc/POLARIS/...`` even
though the binary ran successfully and the results were already on
VMS via `results_transfer`.

### Bug fix

- **`_finalize_terminal_sample` is now idempotent.** If the sample
  is already in a terminal state (FINISHED / FAILED / CANCELLED),
  the call returns immediately without re-running `collect_output`.
  Without this guard, a duplicate call against a workspace that
  `cleanup_on_success` had already deleted would raise "scenario
  file missing" and **downgrade the sample from FINISHED to FAILED**,
  silently clobbering the metric.
- **FINISHED is sticky even if the guard is bypassed.** If
  `collect_output` raises inside `_finalize_terminal_sample`, we
  re-read the store row before marking FAILED. If the row is
  already FINISHED (some prior call committed it), we preserve
  FINISHED and log a WARNING. Defense in depth — operational
  invariant is "FINISHED never goes backwards to FAILED."
- **`_try_recover_from_disk` short-circuits on FINISHED.** Same
  pattern: returns True immediately for an already-FINISHED sample
  without re-running `collect_output`. Prevents the same workspace-
  cleanup race in the disk-recovery path.

### What I don't know

The exact trigger that produces the duplicate `_finalize_terminal_sample`
call is unclear from reading the v0.17 code paths — `outstanding.pop(sid)`
in the poll loop should prevent re-processing. The defensive guards
prevent the symptom regardless. If you run with debug logging and
catch a ``sample N already terminal; skipping duplicate finalize``
message, please report back so the root cause can be pinned down.

### Workaround for users still on v0.17.0

Set `cleanup_on_success: false` on the simulator in the YAML. The
bug requires workspace deletion to trigger — without it the
duplicate finalize is a no-op even without the v0.17.1 guard.
Quota will grow but the data won't be lost.

## 0.17.0 — 2026-06-19

The DFW DOE agent reported a 2-hour stuck-master incident *after* v0.15
shipped its "self-healing live master" claim — same symptom: 16
samples finished cleanly in PBS, master kept polling on a 5-min
heartbeat without ever transitioning them in the SampleStore. Audit
revealed v0.15's in-flight reconcile only covered the UNKNOWN path,
not the FINISHED-but-not-yet-collected path. v0.17 fixes the actual
root cause.

### Bug fixes

- **Per-sample finalize in the poll loop.** Previously
  `_evaluate_batch` followed "submit all → poll until all terminate
  → collect everyone in step 3." A single slow / hung sample held
  every already-FINISHED sibling in SampleStore RUNNING until the
  whole batch completed. v0.17 finalizes each sample (collect_output
  + metric.compute + store.update + post-success hooks) inline as
  soon as its job goes terminal. One hung sample no longer blocks
  the other 15. The deferred "step 3" is replaced by a defensive
  sweep that warns + falls back to disk recovery if any sample
  somehow exited the poll loop still RUNNING (shouldn't happen
  post-refactor).
- **Disk-recovered samples now also run post-success hooks.** The
  v0.16 inconsistency: a sample harvested by `_try_recover_from_disk`
  (via in-flight UNKNOWN reconcile or `polarisopt resume`'s
  reconcile / recover-from-disk) skipped `transfer_results` +
  `cleanup_after_success`, defeating the v0.16 "ephemeral
  per-sample disk" pattern for any sample that came back via the
  disk-recovery path. Both hooks now fire after a successful disk
  recovery, in the same order the live-master FINISHED path uses.

### Added

- **Field-level diff in `ConfigDriftError`.** Previously `retry-failed`
  / `resume --force`-gated errors only said "fingerprints differ"
  (`recorded: ['925e9ca21b23e32e']; current: 'a379e80dc175e379'`)
  and the operator had to guess what changed. v0.17 stores a
  `config_snapshot` dict on every sample at submit time and shows
  the recorded → current diff in the error:
  ```text
  Field-level diff (recorded → current), based on sample 73:
    simulator.options.population_scale_factor: 0.05 → 0.01
    runner.options.heartbeat_interval: 300.0 → 60.0
  ```
  Pre-v0.17 samples without a snapshot fall back to the hash-only
  message with an explanation of why the diff isn't available.
- **Stale-running canary in heartbeats.** At each heartbeat, the
  master queries the store for samples with `status=RUNNING` and
  `updated_at > 1 hour` old. If any, emits a separate WARNING line
  with the IDs and a pointer at `polarisopt recover-from-disk`.
  Compounds with the per-sample finalize fix: post-v0.17, a
  stale-running entry is a real stall, not the deferred-collection
  artifact it was in v0.15/v0.16.

### Internal

- **`simulator_config_payload(config)`** extracted from
  `simulator_config_fingerprint` so the snapshot and the hash feed
  off the same dict shape. Anything that drifts in the snapshot
  drifts in the fingerprint and vice-versa.

### Known gap / deferred

- **PBS user-limit aware submission** (item 3 from the agent's batch).
  Today a batch larger than the per-user concurrent-job limit thrashes
  the v0.15 transient-backoff path and can permanently fail the last
  few submissions. Workaround: set `phase.batch_size` below the
  cluster's per-user limit. Proper auto-detect (`qstat -Bf` parse +
  paced submission) deferred to a future release; it needs more
  architecture than this batch.

## 0.16.0 — 2026-06-18

Ephemeral per-sample disk. v0.14 made FAILED workspaces cleanable
(opt-in); successful samples also produce 1.5–3 GB each, which fills
LCRC quotas independently of failure rate. The user's runner had been
hand-rolling VMS copy + manual unlink of heavy files inside the iter
dir — project-local code every consuming agent would re-invent.
v0.16 absorbs that pattern.

### Added

- **`PolarisSimulator(cleanup_on_success=False)`** + new
  `cleanup_after_success(sample)` hook. Mirrors v0.14's
  `cleanup_on_failure` shape. The orchestrator calls the hook after
  a sample reaches FINISHED and the metric is committed; the
  simulator either rm-rf's the workspace or prunes it (see next).
  Default `False` (preserve results for analysis — opt in for
  quota-tight studies). `polaris_convergence` keeps the same
  default — success cleanup is more destructive than failure
  cleanup; the user opts in deliberately.
- **`keep_files_after_success: list[str]`** allowlist of
  `fnmatch`-style glob patterns relative to the workspace.
  When `cleanup_on_success=True` + non-empty allowlist, the prune
  walk preserves matching files (and the directories that contain
  them) and deletes the rest. Empty list (default) = full wipe.
  Examples: `["DFW-Demand.sqlite", "log/*.log", "**/result.h5"]`.
- **`results_transfer: {type, options, dest}`** option on
  `PolarisSimulator`, symmetric counterpart to today's `transfer`
  (which only handles staging IN). When set, after
  `collect_output` succeeds the orchestrator calls
  `simulator.transfer_results(sample, output)` which pushes the
  output directory to remote storage via the named transfer
  backend (`local` / `anl` / `globus`). Destination path:
  `<dest>/<phase>/<sim-NNNNNN>/<output_dir_name>/`. Combine with
  `cleanup_on_success=True` for fully ephemeral per-sample disk —
  the workflow becomes "stage in → run → push out → clean up,"
  with no net disk growth per sample.

### Orchestrator wiring

A new `_post_success_hooks(sample, output)` step runs immediately
after the FINISHED transition commits. Order: `transfer_results`
first (because cleanup may delete the source), then
`cleanup_after_success`. Both are best-effort — if either raises,
the sample stays FINISHED with a WARNING logged. The metric is
already persisted before either fires.

### Compatibility

Backwards-compatible. All three new options default to off /
preserve-everything. Existing YAMLs and project-local cleanup code
(like the user's runner-side trim) continue to work unchanged. The
recommended migration: enable `results_transfer` + `cleanup_on_success`
in the simulator block, drop the project-local trim code.

## 0.15.0 — 2026-06-18

Self-healing live masters. The DFW DOE agent reported a 2h+ stuck-master
incident on Improv: PBS aged jobids out of `qstat` accounting, polarisopt
defensively kept the samples in RUNNING (correct), but had no in-flight
mechanism to harvest the on-disk artifacts. Manual `recover-from-disk`
worked but required the operator to notice. v0.15 closes the loop —
the master self-heals on UNKNOWN, retries transient submit errors with
backoff, and reports stalled state via heartbeat deltas.

### Added

- **In-flight disk reconcile** (item 1 from the agent's batch — highest
  leverage). When `runner.status()` returns UNKNOWN inside
  `_evaluate_batch`'s poll loop, the master now tries
  `simulator.collect_output` + `metric.compute` against the workspace
  *before* incrementing the orphan counter. If outputs parse → sample
  goes FINISHED with metric set, no restart needed. Falls through to
  the existing orphan-threshold logic when disk recovery fails. Same
  mechanism v0.10.1 added to `resume`, now also live inside `run`.
- **Transient submit-error backoff** (item 3). `PBSRunner.submit` and
  `SlurmRunner.submit` detect known-transient error patterns and
  retry with exponential backoff (10s → 30s → 60s → 120s → 240s,
  default 5 attempts) instead of marking the sample FAILED. PBS:
  per-user limit (`would exceed complex's per-user limit`), server
  busy, resource temporarily unavailable. Slurm: `QOSGrpJobsLimit`,
  `AssocGrpJobsLimit`, controller busy. Permanent errors (unknown
  queue, bad account, invalid partition) still raise immediately.
  Backoff schedule is per-instance configurable via
  `submit_retry_backoff_s`.
- **Heartbeat transition deltas** (item 2). The `[heartbeat]` log line
  now shows `+3 FINISHED, +0 FAILED, +0 RECOVERED since last` so a
  stalled master is visible in the log even when the PBS-side counts
  don't move. Empty deltas surface as ``0 transitions since last
  (master may be stalled)``.

### Changed

- **`PolarisConvergenceSimulator.cleanup_on_failure` defaults `True`**
  (item 4). polarislib workloads stage 1.5–3 GB per sample; preserving
  every FAILED workspace fills the filesystem fast. The base
  `PolarisSimulator` keeps the default `False` (forensic preservation
  is the more common use case for hand-rolled simulations). Explicit
  YAML overrides win.

## 0.14.0 — 2026-06-17

Quota / disk-full survival kit. The DFW DOE agent's master crashed
mid-staging on sample 73 of 100 because the workspace filesystem
filled up — partial copies left stranded, no clean error message,
no way to free space without manually `find`-ing for failed sample
folders. This batch closes that path: catch quota errors cleanly,
refuse to start a copy that won't fit, opt in to per-sample cleanup,
sweep retroactively.

### Added

- **`QuotaExceededError`** (subclass of `TransferError`). The local
  transfer wrapper catches `errno.EDQUOT` (122 — per-user quota) and
  `errno.ENOSPC` (28 — filesystem full), surfaces them as a structured
  exception so the master can report
  ``copy /lcrc/.../m → /lcrc/.../sim-073 hit EDQUOT: Disk quota
  exceeded`` instead of a 50-line traceback. Also detects ENOSPC
  buried inside `shutil.Error`'s per-file batch from `copytree`.
  Existing `except TransferError` callers still catch it
  (backwards-compatible subclass).
- **Pre-stage quota check** in `PolarisSimulator.prepare()`. Before
  `transfer.copy`, compute `du -sb`-equivalent of `model_source`,
  call `os.statvfs` on the workspace, refuse with `QuotaExceededError`
  if `free < model_size × quota_safety_multiplier`. Default
  multiplier 1.5×. Catches the cascade at sample 1 instead of
  sample 73.
- **`PolarisSimulator(quota_check=True)`**, **`quota_safety_multiplier=1.5`**
  knobs to tune or disable the check.
- **`PolarisSimulator(cleanup_on_failure=False)`** + new
  `Simulator.cleanup_after_failure(sample)` hook. When `True`,
  rm -rf's the workspace after a sample reaches a terminal FAILED
  state (after retries, if any). Default `False` to preserve forensic
  artifacts. Hook runs as step 5 of `_evaluate_batch` so it can't
  delete a workspace that's about to be retried.
- **`polarisopt clean --failed [--dry-run] <study.yaml>`** — retroactive
  sweep. Enumerates FAILED samples with on-disk folders, sums total
  bytes, deletes (or with `--dry-run`, prints the plan). Store rows
  stay FAILED so the failure messages survive — only the workspaces
  are removed. Idempotent.

### Why the asymmetry on defaults

`quota_check` defaults **on** because it's a pure safety net (refuses
to start something doomed). `cleanup_on_failure` defaults **off**
because forensic artifacts are usually what you want to keep when
something fails — the agent who's debugging a failed run wants the
logs, not "we cleaned up so you can't see what went wrong."

## 0.13.1 — 2026-06-17

Hot fix flagged by the DFW DOE agent ~60 minutes into the first
real Phase 3a calibration on Improv: `polarisopt status --verbose`
was showing an hour-stale "Spreading across nodes" line as the
sample's "last log line" while the binary was minutes-fresh.

### Bug fixes

- **`polarisopt status --verbose` now peeks at nested
  `polaris_progress.log` files.** Previously only globbed top-level
  `*.log` / `*.out` / `*.err` in the sample folder. After the
  binary's boot phase, the wrapper log (`polaris.stdout.log`) goes
  silent because actual simulation activity is written to
  `<output_dir>/log/polaris_progress.log` instead — for
  polaris_convergence runs, that file is the only source of recent
  signal. `_last_log_line` now globs `*/log/polaris_progress.log`
  too and picks whichever file has the most recent mtime.
- **Practical effect:** for a sample that's been running for an
  hour, `status -v` now shows ``sim hour 12 of 24`` instead of
  ``=> Spreading across nodes [[Node 0 free=16]]``.

## 0.13.0 — 2026-06-17

Turns polarisopt from a single-cluster (Slurm) tool into the LCRC-wide
choice for POLARIS calibration. LCRC users routinely move studies
between Crossover (Slurm) and Improv / Bebop (PBS Pro); without a
PBSRunner, polarisopt was scheduler-locked to one.

### Added

- **`runner.type: pbs`** — new `PBSRunner` plugin parallel to
  `SlurmRunner`. Submits via `qsub`, polls via `qstat -fx`, cancels
  via `qdel`. YAML usage is identical to Slurm; only the runner type
  and resource field names change.
- **`PBSResources` dataclass** — `queue`, `account`, `walltime`,
  `select`, `ncpus`, `mpiprocs`, `mem` (lowercase units —
  `96gb`, not `96GB`), `place` (`excl` / `shared` / `free` —
  parallel to Slurm `--exclusive` / `--oversubscribe`),
  `join_output`, plus the same `extra_directives` + `setup_commands`
  carve-outs `SlurmResources` exposes.
- **`exit_status`-aware FINISHED → FAILED promotion.** PBS marks a
  job `F` whether the binary succeeded or seg-faulted; the runner
  reads `exit_status` from `qstat -fx` and routes non-zero exits to
  `FAILED` so the orchestrator doesn't treat crashes as successful
  metric collections.
- **Single-call status path.** `qstat -fx` covers both live and
  historical jobs, so unlike Slurm there's no separate
  squeue-then-sacct fallback. Cleaner code, fewer subprocess calls
  per poll.
- **Full PBS job-id preserved.** Job IDs come back as
  `<number>.<hostname>` (e.g. `7609762.imgt1`); polarisopt stores
  and forwards the full string — `qstat` / `qdel` reject the bare
  number.
- **`docs/how-to/run-on-pbs.md`** — new top-level how-to mirroring
  `run-on-slurm.md`. Includes the full Slurm↔PBS translation table,
  status-code mapping, LCRC Improv specifics (queues, account format,
  module-load line matching `worker_loop_lcrc.sh`'s Improv branch),
  and common failure modes with diagnostic recipes.

### Internal

- **`runners.factory.make_runner`** now hydrates `default_resources`
  dicts for both Slurm (`SlurmResources`) and PBS (`PBSResources`)
  using the same nested-dict pattern.
- **`runners` package exports** `PBSJob` / `PBSResources` /
  `PBSRunner` alongside their Slurm counterparts.

## 0.12.1 — 2026-06-17

Workspace lock — closes the "two masters racing on the same
SampleStore" footgun before it bit.

### Added

- **`flock(2)`-based workspace lock.** `polarisopt run` and
  `polarisopt resume` acquire an exclusive lock on
  `<workspace>/.polarisopt.lock` for the duration of the master
  process. If another live master holds it, both commands fail fast
  with a friendly error pointing at the holder's PID, hostname,
  start time, polarisopt version, and the action it's running:
  ```
  another polarisopt master holds the workspace lock:
    PID:      12345
    host:     xover-login1
    started:  2026-06-17T09:23:45+00:00
    version:  0.12.0
    action:   run
    lock:     /lcrc/.../calibration-1pc/.polarisopt.lock
  ```
  The lock is kernel-managed (auto-releases on process death — no
  stale-state cleanup). Metadata sidecar
  (`.polarisopt.lock.meta`) is best-effort cleaned on graceful exit.
- **`--force` on `polarisopt run`** to bypass the lock check.
  (`resume`'s existing `--force` flag now also bypasses the lock in
  addition to the drift check.) Use only when knowingly accepting
  the racing-masters consequences.
- **`docs/operating-as-an-agent.md`** — new "Workspace lock" section
  documenting the guarantee and the short-mutator carve-outs
  (`cancel` / `abort` / `retry-failed` / `recover-from-disk` skip the
  lock check since they're operator interventions that should run
  anytime).

### Why it matters

Submitting the master as a Slurm job (per
`docs/how-to/run-on-slurm.md`) made it trivial to forget there's one
already running and fire a second `polarisopt resume` from a login
shell. Two orchestrators racing means: duplicate submissions,
cancelled-but-still-alive jobs, recursive-retry doubling, state
thrash on FINISHED/RUNNING transitions. WAL mode protected the
SQLite layer but not the application-level decisions.

## 0.12.0 — 2026-06-17

Closes the third recovery path — automatic per-sample retries — and
ships the agent-operating playbook that previous DOE / calibration
agents had to assemble from scratch.

### Added

- **`runner.options.max_retries: N`** — automatic per-sample retry
  budget inside `_evaluate_batch`. On each FAILED transition the
  sample's `extra["retry_count"]` increments; the orchestrator
  re-submits up to `max_retries` times before letting it stay
  FAILED. Closes the manual `polarisopt retry-failed --run` loop for
  the transient-failure case (occasional OOM, NODE_FAIL, time-limit
  near the boundary). Default `0` — no auto-retry, same as v0.11.x.
  Permanent failures still get rejected after exhausting the budget,
  so this never burns infinite compute on a semantic bug.
- **Retry audit trail.** Each retry appends to
  `sample.extra["retry_log"]` with `{attempt, max, prior_message}`
  so the `sample.message` field can still reflect the final/current
  failure reason without losing history.
- **`polarisopt status --verbose` shows a `retry` column.** Lets you
  spot "this sample failed 3× → real bug" at a glance vs "1× →
  probably transient."
- **`docs/operating-as-an-agent.md`** — new top-level page for AI
  agents *driving* polarisopt (as opposed to AI agents *modifying*
  polarisopt — that's still `AGENTS.md`). Covers the canonical loop,
  workspace conventions, the typo class, the three retry paths,
  master-death recovery, heartbeat output, orchestrator knobs,
  "don't edit polarisopt source," and the feedback format that's
  driven v0.6–v0.11 evolution. Cross-linked from `docs/index.md`
  and `AGENTS.md`.

### Internal

- **`max_retries` added to `_ORCHESTRATOR_RUNNER_OPTIONS`.** Joins
  `poll_interval` / `orphan_threshold` / `heartbeat_interval` as a
  YAML key that lives in `runner.options` but is consumed by the
  orchestrator, not the runner constructor. Excluded from the
  config-drift fingerprint so tweaking retry policy doesn't trip
  `retry-failed` / `resume` drift checks.

## 0.11.0 — 2026-06-12

Six items from the DFW DOE agent after a 25h study + recovery cycle.
Three bug fixes that eliminate a class of "config-knob in YAML breaks
some entry point but not others" pain, plus three quality-of-life
additions.

### Bug fixes

- **`build_runner` strips orchestrator knobs.** `reconcile_running`,
  `cancel_sample`, `abort_study`, and friends all go through
  `build_runner`, which was passing `cfg.runner.options` raw to
  `make_runner` while `StudyRunner.__init__` popped them. Result:
  `polarisopt run` accepted a YAML with `poll_interval: 60`, but
  `polarisopt resume` crashed three sessions later with
  `TypeError: SlurmRunner.__init__() got an unexpected keyword
  argument 'poll_interval'`. Now consistent across every entry point.
- **`polarisopt resume` calls `recover_from_disk` automatically.**
  After `reconcile_running` finishes, resume sweeps any remaining
  non-FINISHED samples for on-disk artifacts. Catches zombies that
  reconcile missed (runner.status raised, metric changed since the
  binary wrote outputs, etc.) without requiring the user to know
  the `recover-from-disk` subcommand exists. Skipped under
  `--skip-reconcile`.
- **Config-drift check on `polarisopt resume`.** Symmetric with
  `retry-failed`'s v0.7 check. Resume now compares the current
  simulator+runner fingerprint against the fingerprints recorded on
  existing samples and refuses with a clear error if they've
  diverged. `--force` overrides for the genuine "resume under new
  config" case.

### Added

- **`polarisopt best <study.yaml>`** — wraps `SampleStore.best_so_far`.
  Prints id / phase / iteration / inputs / metric / folder for the
  argmin sample (or argmax with `--maximize`). `--objective N` picks
  the column for multi-objective studies; `--phase` restricts; `--json`
  emits a machine-readable payload for shell pipelines.
- **`--quiet-heartbeat` flag on `polarisopt run` / `resume`.**
  Filters the periodic `[heartbeat] N sample(s) outstanding…` log
  lines out of the default output. State transitions still log at
  INFO. For 25h studies where heartbeats otherwise dominate the log
  (~300 lines that have to be `grep -v`'d).

### Documentation

- **`docs/how-to/use-from-notebook.md`** — new Partitioning by phase
  and iteration subsection. The `samples.iteration` column has been
  populated since v0.1 (warm-up = 0, BO rounds = 1..N for sequential
  phases) but wasn't documented; analysis queries had to infer batch
  boundaries from sample id ranges instead. Now shows the
  `groupby("iteration").min()` pattern explicitly.

### Deferred

- `phase_iteration` schema migration — the column under that name
  doesn't exist, but `samples.iteration` does the same job. No schema
  change needed; see the docs update above.

## 0.10.1 — 2026-06-11

The "zombie sample" recovery release. Master process death during a
long BO loop used to leave samples in RUNNING forever with NULL
metrics, even though Slurm had completed the jobs and written all
artifacts to disk. The BO surrogate then ignored them because it
filters on non-NULL metric values. Result: wasted compute every
master crash.

Flagged by the DFW DOE agent with the clearest framing yet of this
gap — disk artifacts are ground truth; polarisopt should use them.

### Bug fixes / behavioral changes

- **`reconcile_running` is now disk-first.** For each previously-
  RUNNING sample, after the runner-status check, polarisopt tries
  `simulator.collect_output` + `metric.compute` against the sample's
  workspace. If both succeed the sample is FINISHED with the metric
  value persisted — *regardless* of what the runner says. Disk
  beats Slurm's verdict:
  - runner UNKNOWN + outputs on disk → FINISHED (the zombie case
    when sacct retention aged the jobid out)
  - runner FAILED + outputs on disk → FINISHED (binary wrote
    before exiting non-zero)
  - runner FINISHED + outputs on disk → FINISHED with metric
- **Active jobs are left alone.** Runner status RUNNING/QUEUED
  skips disk recovery to avoid racing a partial write.
- **CANCELLED preserves user intent.** Runner-CANCELLED samples
  stay CANCELLED even if outputs exist on disk.
- **FINISHED with missing output is now FAILED** (was: silently
  left as RUNNING forever, no path to harvest). The v0.5 reconcile
  comment "leave for the regular poll loop" was lying — there was
  no such loop.

### Added

- **`recover_from_disk(config)` API function.** Sweeps RUNNING +
  FAILED samples (plus CANCELLED with `include_cancelled=True`)
  and harvests any whose outputs exist on disk. Doesn't consult
  the runner at all — useful when `sacct` has aged jobids out so
  reconcile can't disambiguate orphans from zombies.
- **`polarisopt recover-from-disk study.yaml` CLI.** The manual
  entry point for the standalone sweep. `--include-cancelled` to
  also recover cancelled samples.

### Documentation

- **`docs/how-to/run-on-slurm.md`** — new Submitting the master itself
  as a Slurm job section between Monitoring and Common failure modes.
  Covers the wrapper sbatch script pattern (lightweight master with
  `--oversubscribe`), what it fixes (SSH/tmux death, login-node idle
  cleanup, restart hygiene), and a Recovering after a master crash
  subsection pointing readers at the new `recover-from-disk` CLI plus
  the disk-recovery story in `resume`.

## 0.10.0 — 2026-06-11

New simulator capability flagged by the DFW DOE agent after their
end-to-end smoke wrapped: parameters that drive a Python pre-processing
step (build a demand DB, materialize skim tables, transform model
files) rather than landing in scenario JSON. Today's
`Parameter.file: <some>.json` injection only handles the latter.

### Added

- **`PolarisSimulator(pre_script=...)`** — optional Python script
  invoked before the POLARIS binary, with every sample parameter
  forwarded as `--<dashified-name>=<value>` (`am_sigma` →
  `--am-sigma`). Booleans render as `true`/`false`, mirroring the
  `polaris_convergence` `runner_options` forwarding convention.
  Values are shell-escaped. `set -e` is emitted in the rendered
  command so a pre-step failure aborts the sample before the binary
  runs (no silent feeding-stale-demand-into-POLARIS).
  Use case: `am_sigma` / `pm_sigma` (and other parameters that drive
  `build_demand.py`-style pre-processing).
- **`PolarisSimulator(pre_script_interpreter=...)`** — Python
  interpreter for `pre_script`. Defaults to `sys.executable`.

Backwards compatible — both default to `None`, and YAMLs that don't
mention them get the exact same rendered command as v0.9.x.

### Internal

- **`_arg_value` helper moved to `simulator/polaris.py`**, since both
  `PolarisSimulator.pre_script` and
  `PolarisConvergenceSimulator.runner_options` use the same
  rendering convention. `polaris_convergence` now imports it.

## 0.9.3 — 2026-06-11

Operational gotcha flagged by the DFW DOE agent: on Crossover, the
bundled `polaris-slurm.yaml` example let Slurm co-locate four
polarisopt samples on one node, and the kernel OOM-killed one when
their combined working set exceeded 256 GB — even though each
per-job `--mem` was within its own limit.

### Added

- **`SlurmResources(exclusive=True)`** — renders `#SBATCH --exclusive`
  so the sample gets a whole node to itself. Default `False`
  (matches prior behavior — backwards compatible).
- **Bundled `polaris-slurm.yaml`** now sets `exclusive: true` with an
  explanatory comment so users following the docs don't hit the same
  OOM-via-co-location trap.

## 0.9.2 — 2026-06-11

Bug-fix release. Both bugs flagged by a third agent doing a polarisopt
port (the taxidemo emukit demo).

### Bug fixes

- **`polarisopt plan` no longer fails on YAMLs that set
  `poll_interval` / `orphan_threshold` / `heartbeat_interval` under
  `runner.options`.** These are valid YAML — `StudyRunner` pops them
  before constructing the runner — but `plan_study` was passing them
  straight through to `make_runner`, which made `SlurmRunner.__init__`
  reject them. The bundled `polaris-slurm.yaml` example was unrunnable
  through `polarisopt plan` for this reason. `plan_study` now mirrors
  `StudyRunner`'s strip step. Regression test covers both the synthetic
  case and the bundled example by name.
- **Bundled `polaris-slurm.yaml`: `account: TPS` → `account: tps`.**
  Crossover's Slurm account is lowercase; uppercase was rejected by
  the controller. Partition stays `TPS` (which it actually wants).

### Internal

- **Single source of truth for the orchestrator-knob set.** Previously
  `_ORCHESTRATOR_RUNNER_OPTIONS` (in `studies/ops.py`) and
  `_RUNNER_ORCHESTRATOR_KEYS` (in `studies/validate.py`) drifted as two
  copies of the same constant. Consolidated into one in `ops.py`;
  `validate.py` and `plan.py` now import it. Future additions to the
  orchestrator-knob set only need to land in one place.

## 0.9.1 — 2026-05-23

Docs-only release. Closes the notebook-usability documentation gap —
analysis was well-documented but driving polarisopt from a notebook
(and the full `SampleStore` analysis surface beyond `to_dataframe`)
wasn't.

### Documentation

- **New `docs/how-to/use-from-notebook.md`** — covers:
  - Programmatic API mirror of every CLI subcommand (`validate_study`,
    `plan_study`, `StudyRunner`, `cancel_sample`, `abort_study`,
    `retry_failed`, `reconcile_running`, etc.) — none of which had
    narrative docs before.
  - Full `SampleStore` analysis surface (`finished_samples`,
    `metric_matrix`, `best_so_far`, `pareto_front`) with signatures
    and a "drop into matplotlib/seaborn" framing. Only `to_dataframe`
    was previously shown in any prose doc.
  - The "read while a study is still running" pattern — the SQLite
    WAL mode makes concurrent notebook reads safe alongside the CLI
    writer; this property was used in production but never written
    down.
  - Recommended workflow: drive long runs via CLI in a terminal,
    drive validate/plan + analysis from the notebook.
- **`docs/how-to/index.md`** — adds the new how-to to the index.
- **`docs/getting-started.md`** — the SampleStore snippet now points
  readers at the new how-to for the deeper analysis API.

## 0.9.0 — 2026-05-23

Fixes the bug a second calibration agent hit on its first end-to-end
smoke run: SIF binaries on Crossover/TPS were invoked bare instead of
under `apptainer run -B …`, so the container's default mount namespace
couldn't see `/lcrc/` and every sample failed in ~3 s.

### Bug fixes

- **`PolarisSimulator` now wraps SIF binaries with `apptainer run`.**
  When `binary` ends in `.sif`, the rendered command is
  `apptainer run -B <workspace> -B <binary_parent> [-B <user_binds>] <SIF> [<entrypoint>] <scenario> <threads>`
  instead of the historical bare-exec. Native binaries are unchanged.
  Closes "10/10 samples failed with `No such scenario config file`
  inside the container" from the demand-DOE port.

### Added

- **`PolarisSimulator(apptainer_binary="apptainer")`** — defaults to
  `apptainer`; set `"singularity"` on older clusters.
- **`PolarisSimulator(singularity_binds=[...])`** — extra `-B` specs
  for paths the scenario JSON references outside the workspace + SIF
  parent (e.g. shared skim caches). Each entry is a host path or a
  `host:container` mapping. Auto-bind dedups user entries that
  duplicate the defaults.
- **`PolarisSimulator(sif_entrypoint="Integrated_Model")`** — for the
  newer POLARIS SIF format where the runscript dispatches by executable
  name, this string becomes the first positional arg after the SIF.
  Default `None` (historical bare invocation).

## 0.8.1 — 2026-05-22

Docs-only release. Refreshes the how-to guides that fell behind v0.6–v0.8
work and ships the v0.8 typo-class lessons learned from the live
calibration agent.

### Documentation

- **`docs/how-to/debug-failed-samples.md`** — §5 rewritten. The "no
  retry-failed in v0.2 (it's planned)" workaround is gone; replaced
  with the actual v0.7 `retry-failed --run` workflow plus a Config
  drift on retry section covering the v0.8 fingerprint check and the
  ``--force`` escape. §2 now documents `--binary` / `--iteration` for
  tailing POLARIS's per-iteration progress log.
- **`docs/how-to/migrate-from-eqsql.md`** — new Why per-user worker
  pinning drops out paragraph explaining that the contamination-defense
  pattern from EQSQL (`my_workers_regex()` and friends) is unnecessary
  under `SlurmRunner` because there's no shared queue to contaminate.
- **`docs/getting-started.md`** — Python version corrected (3.10+, was
  3.11+; stale since v0.6). `polaris_convergence` simulator and
  `setup_commands` on `default_resources` are now mentioned in "Use
  with POLARIS." New Workspace path convention and Validate before
  submitting sections.
- **`docs/how-to/common-mistakes.md`** — new how-to covering the typo
  class (`distance`/`aggregation`, `sim_key`/`source_key`, etc.) that
  v0.8's plugin-option signature check catches. Documents the
  `validate` → `plan` workflow as the daily-edit cadence.

## 0.8.0 — 2026-05-21

Third pass of calibration-agent feedback. All five items in the v0.8
batch shipped; no design pivots.

### Added

- **`PolarisConvergenceSimulator(single_iteration=True)`** — sugar for
  the choice-model calibration use case. Injects `num_abm_runs=0` and
  `num_dta_runs=0` into `runner_options` so polarislib runs only the
  configured `iteration_type` once with no follow-up `normal_iteration`
  (roughly halves wall time). Conflicting explicit values raise.
  `collect_output` asserts no other iteration_type dirs slipped past —
  catches a misbehaving runner script before it's mistaken for the
  sugar working.
- **`PolarisConvergenceSimulator(disable_async_callback=True)`** — now
  the default. Forwarded as `--disable-async-callback=true` so the
  runner script can pass a no-op for polarislib's `async_end_of_loop_fn`
  (which otherwise tarballs per-iteration DBs out from under metrics
  that need them). Preserve-artifacts is the right default for the
  calibration use case; explicit `runner_options.disable_async_callback`
  wins.
- **`polarisopt logs --binary --iteration=<substr>`** — when a sample
  produced multiple iteration dirs (abm_init + normal_iteration), the
  filter pins the tail to the matching one. Default still picks the
  latest mtime.
- **Plugin-option signature check in `polarisopt validate`** — every
  `options:` block is now typechecked against its plugin's `__init__`
  signature (walks the MRO so subclass `**kwargs` forwarding is handled).
  Catches `distance: l1` (real arg `aggregation`) and `sim_key: demand_db`
  (real arg `source_key`) at validate time instead of after a 30s
  staging round-trip. Classes with `**kwargs` in their own `__init__`
  downgrade unknown keys to warnings (might be legitimately forwarded).
  Runner orchestrator keys (`poll_interval`, `orphan_threshold`,
  `heartbeat_interval`) are allowlisted since `StudyRunner` pops them
  before the runner is built.

### Changed

- **`PolarisConvergenceSimulator.collect_output`** now sets
  `iteration: 0` (was `None`) when the resolved output dir is the
  polarislib unsuffixed form (`<db>_<iter_str>` with no `_<N>`).
  `IdentityMetric` and friends that read `iteration` no longer need a
  `None`-special-case for baselines.

## 0.7.0 — 2026-05-19

Second pass of feedback from the live DFW calibration. All UX/operability,
no design pivots.

### Added

- **Periodic poll-loop heartbeat.** `StudyContext.heartbeat_interval`
  (default 300s) emits an INFO line summarizing every outstanding
  sample. Closes the silent gap between submit and the next state
  transition on long-running batches. Set to 0 to disable. Configurable
  per study via `runner.options.heartbeat_interval`.
- **`polarisopt status --verbose`** — one row per sample with id, phase,
  status, jobid, runtime, folder, and the last line of the most-recent
  log file. `--status` flag filters by sample state.
- **`polarisopt logs --binary`** — tails
  `<workspace>/*/log/polaris_progress.log` (POLARIS's per-iteration
  progress log) instead of the polarisopt wrapper logs. This is what
  tells you what sim-hour the run is in.
- **Config-drift detection on `retry-failed`.** Every sample now records
  a 16-char fingerprint of the simulator+runner config at submit time
  (`sample.extra["config_fingerprint"]`). `retry_failed` refuses with
  `ConfigDriftError` if the YAML has changed since the failed samples
  ran. `--force` overrides for the genuine "retry under new config"
  case. Orchestrator knobs (`poll_interval`, `orphan_threshold`,
  `heartbeat_interval`) are excluded from the fingerprint.
- **`runner_options` soft whitelist.** `PolarisConvergenceSimulator`
  exposes `KNOWN_RUNNER_OPTIONS` (the polarislib `ConvergenceConfig`
  fields we know about). `polarisopt plan` surfaces unknown keys as
  warnings — catches `population_scal_factor`-style typos in <1s
  instead of after a 30s staging round-trip. Branch-specific knobs
  still pass through; this is a soft check, not a hard schema.
- **`progress_log_path` in `collect_output()`** — the polarislib
  binary's per-iteration progress log is now in the simulator's output
  dict (or `None` if it doesn't exist yet), so downstream metrics /
  notebooks don't need to find for it.

### Changed

- **`PolarisConvergenceSimulator.DEFAULT_OUTPUT_DIR_KEY`** is now
  `("Output controls", "output_dir_name")`. polarislib scenarios use
  `output_dir_name`; the base-class default `output_directory` was
  never right for `polaris_convergence`. YAMLs that spelled out
  `output_dir_key` continue to work — this only affects users who
  relied on the default.
- **`PolarisConvergenceSimulator` docstring** now documents that
  `abm_init` runs a full 24-hour traffic simulation regardless of
  `num_dta_runs`. `num_dta_runs=0` means "no extra DTA passes," not
  "no traffic." For cheap calibration: drop `population_scale_factor`
  to 0.01 or use a different `iteration_type`.

## 0.6.0 — 2026-05-16

First release driven by feedback from a real POLARIS calibration run.

### Bug fixes

- **Slurm `#SBATCH` directive ordering.** `_render_script` emitted
  `set -euo pipefail` before the directives, which made Slurm silently
  ignore every directive after that ("No partition specified" even
  though one was set in YAML). Directives now precede any executable
  line.
- **Shell-escape `runner_options` values** in
  `PolarisConvergenceSimulator`. Spaces / shell metacharacters in
  user-supplied runner options would previously corrupt the rendered
  command. (CodeRabbit finding.)

### Added

- **`PolarisConvergenceSimulator`** (`type: polaris_convergence`) —
  first-class simulator that hands a sample to polarislib's
  convergence loop via a user-supplied runner script. Master process
  still never imports polarislib. Forwards arbitrary `run_config`
  knobs (`population_scale_factor`, `num_abm_runs`, `do_skim`, …) to
  the runner as CLI flags. Handles polarislib's
  `<db_name>_<iter_str>[_N]` output-directory naming (both numbered
  and unnumbered).
- **`SlurmResources.setup_commands`** — list of bash lines run after
  `#SBATCH` directives but before the user command. Module loads,
  `source ~/.bashrc`, etc. Cleaner than baking them into every
  simulator's command string.
- **`ConstantMetric`** (`type: constant`, alias `null_metric`) —
  fixed value for studies that produce artifacts, not objectives.
  Documents intent.
- **`polarisopt plan <study.yaml>`** — dry-run: stage sample 0,
  render its `JobSpec`, optionally render the sbatch script, **don't**
  submit. Catches operational failures (missing modules, scenario
  JSON key typos, runner script paths, parameter file relpaths)
  before burning a Slurm allocation.
- **`polarisopt validate --deep`** — extends `validate` with the same
  staging + JobSpec rendering as `plan`.
- **`utils/_compat.py`** — consolidates the 3.10 backport shims
  (`datetime.UTC`, `enum.StrEnum`) so each module imports from one
  place instead of repeating the try/except block.

### Changed

- **Python 3.10 support.** `requires-python = ">=3.10"` (was 3.11).
  The codebase uses no 3.11-only syntax; only `datetime.UTC` and
  `enum.StrEnum` needed shims.
- **Workspace layout.** `StudyRunner` no longer auto-creates `logs/`
  and `scripts/` — they were never populated by polarisopt itself
  (per-sample logs live in `experiments/sim-NNN/`).
  `workspace_layout()` still returns them as available paths for
  backends that want them.
- **Example `polaris-slurm.yaml`** — partition/account corrected to
  `TPS` (Crossover convention), `setup_commands` for module loading
  added, `output_dir_key` annotated.
- **Parameter `file:` docstring** now documents that the relpath
  supports subdirectories (e.g. `config/choice_models/Foo.json`).

## 0.5.0 — 2026-05-16

### Added
- **`SampleStore.pareto_front()`** — return non-dominated finished
  samples as a public API. Single-objective collapses to a 1-element
  list (the best sample). Multi-objective gives the Pareto front.
- **`SampleStore.best_so_far()`** — argmin/argmax over an objective,
  optional phase filter.
- **`SampleStore.finished_samples()`**, **`SampleStore.metric_matrix()`** —
  vector / matrix helpers for notebook analysis.
- **`polarisopt smoke-test [--workspace DIR] [--keep]`** — end-to-end
  install check. Runs an LHS+mock study end-to-end in ~5 seconds.
  Verifies imports, SampleStore, LocalRunner, metric round-trip.
- **Resume reconcile** — at the top of `polarisopt resume`, every
  previously-RUNNING sample is reconciled with the runner (Slurm).
  Terminal jobs (FINISHED/FAILED/CANCELLED) are moved into the store
  before the loop runs; orphans (UNKNOWN forever) become FAILED with
  an "orphaned on resume" message. Bypass with `--skip-reconcile`.

### Changed
- **Graceful Ctrl-C**: the orchestrator's poll loop now catches
  `KeyboardInterrupt`, cancels every in-flight Slurm job via
  `runner.cancel`, marks the affected samples CANCELLED in the store,
  and re-raises. No more orphaned compute when the user kills the
  master.

### Documentation
- New `CHANGELOG.md` (this file) with retroactive release notes.

## 0.4.0 — 2026-05-16

### Added
- **`polarisopt validate <study.yaml>`** — pre-flight schema and plugin
  check. Catches typos in `type:` strings, missing parameter files,
  missing simulator binaries (warning), batch_size < 1, etc. Exits
  nonzero on errors. `--warnings-as-errors` flag for CI.
- **`polarisopt diff <a.yaml> <b.yaml>`** — side-by-side comparison
  of two studies' SampleStores: sample/finished/failed counts, best
  metrics, Pareto-front size.
- **`GlobusTransfer`** (`transfer.type=globus`) — direct globus-sdk
  backend for non-ANL deployments. Users register endpoint UUIDs in
  YAML; longest-prefix endpoint matching; refresh-token auth cached
  under `~/.globus/polarisopt/`. New `[globus]` extra.
- **Convergence-aware `PolarisSimulator`**: `num_iterations=N` wraps
  the binary call in a bash for-loop. `collect_output` picks the
  highest-numbered `_iteration_K` directory and surfaces the
  iteration index in the output dict.
- **Notebook gallery** under `docs/notebooks/`: convergence plots,
  Pareto-front (2D and 3D), Morris sensitivity, comparing two runs.

## 0.3.0 — 2026-05-16

### Added
- **`polarisopt retry-failed`** — flip FAILED samples back to PENDING.
  Optional `--id N` to target specific samples; `--run` to immediately
  re-run after flipping.
- **N-dimensional `HypervolumeStop`** — for ≥3 objectives, falls back
  to BoTorch's `Hypervolume` instead of the 2-D hand-rolled formula.
- **Entry-points-based plugin discovery** — external packages can
  register designs / surrogates / etc. via `[project.entry-points]`
  in their pyproject.toml. CLI auto-loads them on startup.
- **`MultiTaskGPSurrogate`** (`mtgp`) — BoTorch `KroneckerMultiTaskGP`
  for correlated multi-output problems.
- **`polarisopt examples {list,show,copy}`** — 4 bundled example
  study YAMLs ship in the wheel.

## 0.2.0 — 2026-05-16

### Added
- **`polarisopt cancel <sample_id>`** — `scancel` the underlying Slurm
  job, mark sample CANCELLED.
- **`polarisopt abort`** — cancel every non-terminal sample at once.
- **`polarisopt logs <sample_id> [--follow] [-n N]`** — `cat` or
  `tail -f` the sample's stdout/stderr files.
- **Orphan detection** — consecutive `UNKNOWN` poll responses beyond
  a configurable threshold mark a sample FAILED instead of hanging
  the master forever.
- **mkdocstrings + auto-generated API docs** — 67 API pages built from
  NumPy-style docstrings at https://anl-polaris.github.io/polaris-hpc/.
- **Extensive narrative docs** — 6 tutorials, 6 how-to guides, 5
  concept docs, AGENTS.md for AI coding agents, llms.txt.

### Changed
- **Flattened `cli/` and `compat/` subpackages** — `cli/__main__.py`
  → `cli.py`, `compat/eqsql.py` → `eqsql_compat.py`. Cleaner imports.
- **NumPy-style docstrings everywhere** with runnable Examples
  blocks. `pytest --doctest-modules` covers 13 docstring examples.

## 0.1.0 — 2026-05-15

Initial release. Full master/slave architecture with plugin
registries for every algorithm and infrastructure component.

### Core packages
- `parameters` — ParameterSpace + POLARIS JSON injection
- `samples` — Sample dataclass + SQLite-backed SampleStore (WAL mode,
  restart-safe)
- `config` — pydantic study YAML schema + Jinja2 templating
- `design` — Static DOE: LHS, Morris, Sobol, manual
- `surrogates` — Surrogate ABC + BoTorch GP (Matern-ARD)
- `acquisition` — LogEI, qLogEI, qLogEHVI
- `generators` — random + acquisition (batch-first)
- `stop` — max_iter, epsilon, plateau, hypervolume (2D), any/all
- `metrics` — identity, link_moe, choice_share
- `simulator` — MockSimulator (Branin/Rosenbrock/Hartmann-6) +
  PolarisSimulator
- `runners` — LocalRunner + SlurmRunner (sbatch / squeue / sacct /
  scancel)
- `transfer` — LocalTransfer + AnlTransfer (polarislib magic_copy)
- `studies` — Static and Sequential orchestrators + StudyRunner
- `cli` — `polarisopt run|status|resume`
- `eqsql_compat` — drop-in shim for `polaris.hpc.eqsql`

### Build
- src layout, hatchling, Python 3.11+
- Extras: `[bo]` (BoTorch+GPyTorch), `[anl]` (polaris-studio),
  `[dev]` (pytest, ruff, mypy, mkdocs)
