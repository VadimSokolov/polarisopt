# Changelog

Notable changes per release. Format inspired by [Keep a Changelog](https://keepachangelog.com/).

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
