# Study YAML reference

A study YAML has a fixed top-level shape; each pluggable section uses
`{type: <name>, options: {...}}` so users (and external plugin authors)
can swap implementations without touching the schema.

## Top level

```yaml
name: <str>                # unique study name (also the SampleStore study key)
workspace: <path>          # root directory for experiments, logs, db
seed: <int | null>         # RNG seed (optional; null → random)

simulator: { type, options }
runner:    { type, options }
parameters: { source | inline }
metric:    { type, options }

phases:    [ <PhaseConfig>, ... ]   # at least one
```

## simulator

```yaml
simulator:
  type: polaris
  options:
    binary: /path/to/Integrated_Model.sif
    model_source: /path/to/DFW_2050_20251028
    scenario_file: scenario_abm.json
    output_db_filename: DFW-Result.h5
    num_threads: "16"
    transfer:
      type: local            # or "anl" (Globus via polaris-studio)
      options: {}
    # Reclaim scratch after each successful sample, but keep just enough
    # to re-score the metric later without re-running the simulator.
    # Recommended for large studies (DFW: ~2.5 GB → ~100 MB per sample).
    cleanup_on_success: true
    keep_files_after_success:
      - "*/*-Demand.sqlite"           # per-iteration Demand DB — choice_share source
      - "*/summary.csv"               # POLARIS network summary (VMT/VHT)
      - "*/popsyn_fit_results.csv"    # pop-synthesis fit stats
      - "*/log/polaris_progress.log"
```

Nested-ASC contraction (v0.28+ / β-calibration; BLP-1995 β outer,
ASC inner). polarisopt validates the config shape and forwards
`--nested-asc-<key>=<value>` flags to the runner; the runner (which
imports polarislib) is expected to invoke
`polarislib.runs.calibrate.mode_choice.calibrate` after each POLARIS
run at candidate β:

```yaml
simulator:
  type: polaris_convergence
  options:
    # ...standard fields as above...
    nested_asc_contraction:
      enabled: true
      calibrator: polarislib.runs.calibrate.mode_choice.calibrate
      num_planned_activity_iterations: 3
      step_size: 2.0
      target_csv_dir: calibration_targets
      cache_post_contraction: true
      timeout_minutes: 30
      on_convergence_failure: use_last   # or mark_sample_failed
```

Off by default. Unknown keys are rejected — catches typos like
`timout_minutes` early. v0.32 fixed a critical bug where
`_resolve_output_dir` was picking polarislib's `_calib_N`
contraction artifacts (empty Trip tables) as the "highest-numbered
iteration"; that's now handled correctly when this feature is on.

Or for tests:

```yaml
simulator:
  type: mock
  options:
    function: branin        # branin | rosenbrock | hartmann6 | quadratic
```

## runner

```yaml
runner:
  type: slurm
  options:
    default_resources:
      partition: bdwall
      account: POLARIS
      time: "02:00:00"
      nodes: 1
      ntasks: 1
      cpus_per_task: 16
      mem: 64G
      extra_directives:
        - "#SBATCH --qos=high"
```

Or local:

```yaml
runner:
  type: local
  options: {}
```

## parameters

Exactly one of `source` or `inline`:

```yaml
parameters:
  source: ./params.yaml
```

```yaml
parameters:
  inline:
    - { name: x1, file: DestinationChoice.json, min: -5.0, max: 10.0, type: float }
    - { name: x2, file: DestinationChoice.json, min: 0,    max: 20,   type: int   }
```

Each record names which POLARIS JSON file holds the variable and the
search bounds. ``type`` may be ``float`` (default) or ``int``.

**v0.27+ optional per-parameter fields** — used by β-calibration
studies (v0.29 identifiability, v0.31 history matching, MAP wrappers):

```yaml
parameters:
  inline:
    - name: b_IVTT_Auto_Mean
      file: config/choice_models/ModeChoiceModel.json
      min: -0.30
      max: -0.001
      prior:
        type: gaussian                       # gaussian | log_normal | truncated_normal | uniform | beta
        mean: -0.10                          # ≈ VOT $10/hr at $18/hr median wage
        std:  0.035                          # Small-Verhoef 2007 dispersion
      hold_at_prior_mean_if_unidentified: true   # v0.29 auto-drop pins here
```

Prior-type shapes:
- `gaussian`: `mean, std`
- `log_normal`: `log_mean, log_std` (x > 0)
- `truncated_normal`: `loc, scale, low, high`
- `uniform`: `low, high`
- `beta`: `alpha, beta` (x ∈ (0, 1))

A prior whose `.mean` falls outside `[min, max]` is rejected at
config-load time — almost always a config bug.

## metric

```yaml
metric:
  type: link_moe
  options:
    target: /path/to/target/DFW-Result.h5
    aggregation: rmse        # rmse | mse | mae
```

```yaml
metric:
  type: choice_share
  options:
    target_db: /path/to/target/Demand.sqlite
    sql: "SELECT mode AS category, COUNT(*) AS count FROM trip GROUP BY mode"
    aggregation: sum_abs     # sum_abs | rmse | cross_entropy | kl_divergence | jensen_shannon | vector
    # For cross_entropy / kl_divergence, sim shares are add-alpha smoothed
    # via (n_k + alpha) / (N + K * alpha). Default alpha=1 (Dirichlet(1)
    # posterior mean). Set to 0 to disable and fall back to eps-clipping
    # (v0.20 behavior, kept for reproducibility of older studies).
    laplace_smoothing_alpha: 1.0
```

Multi-moment (v0.26+) — vector residuals across N named moments with
per-moment CSV targets. Enabler for history-matching and POM
calibration:

```yaml
metric:
  type: moment_set
  options:
    source_key: demand_db
    # BO consumers want a scalar. History matching (v0.31+) wants the
    # vector. Options: none | sum_squared_weighted | max_implausibility
    # | mean_implausibility
    scalarize: none
    moments:
      - name: mode_shares_by_purpose
        source_sql: |
          SELECT purpose, mode,
                 COUNT(*)*1.0 / (SELECT COUNT(*) FROM Trip WHERE purpose IS NOT NULL)
                 AS share
          FROM Trip GROUP BY purpose, mode
        target: calibration_targets/mode_shares.csv
        target_key_cols: [purpose, mode]
        target_value_col: share
        obs_noise_std: 0.005
        model_discrepancy_std: 0.02   # required — Vernon 2010 §3.5
      - name: boarding_by_agency
        source_sql: "SELECT agency, type, SUM(n) AS boardings FROM MM_Trip GROUP BY agency, type"
        target: calibration_targets/boardings.csv
        target_key_cols: [agency, type]
        target_value_col: boardings
        aggregation: log_ratio_residual    # for count moments spanning decades
        obs_noise_std: 0.10
        model_discrepancy_std: 0.20
```

```yaml
metric:
  type: identity
  options:
    keys: value              # or [obj1, obj2] for multi-objective benchmarks
```

## phases

### Static

```yaml
- name: screening
  type: static
  design:
    type: lhs               # lhs | morris | sobol | manual
    options:
      n: 16
```

Morris:

```yaml
- name: screening
  type: static
  design:
    type: morris
    options:
      n_trajectories: 8
      num_levels: 4
```

Manual:

```yaml
- name: replay
  type: static
  design:
    type: manual
    options:
      points:
        - [0.5, 1.0]
        - [0.8, 2.5]
```

### Sequential

```yaml
- name: bo
  type: sequential
  warm_up:                  # optional initial design (skipped if FINISHED samples exist)
    type: lhs
    options: { n: 8 }
  generator:
    type: acquisition
    options:
      surrogate:
        type: gp
        options: { nu: 2.5 }
      acquisition:
        type: qei            # ei | qei | qehvi
        options:
          mc_samples: 256
  batch_size: 4
  minimize: true             # set false to maximize
  stop:
    type: any
    criteria:
      - { type: max_iter, options: { n: 20 } }
      - { type: epsilon,  options: { epsilon: 0.01 } }
```

Multi-objective with qLogEHVI:

```yaml
- name: pareto
  type: sequential
  warm_up: { type: lhs, options: { n: 12 } }
  generator:
    type: acquisition
    options:
      surrogate:    { type: gp,    options: {} }
      acquisition:  { type: qehvi, options: { ref_point: [10.0, 10.0] } }
  batch_size: 4
  minimize: true
  stop: { type: max_iter, options: { n: 30 } }
```

### History matching (v0.31+)

Vernon-Goldstein-Bower 2010 wave protocol. Requires a `moment_set`
metric with `scalarize: none` so per-moment residuals reach the
emulator. Emits `nroy_wave{N}.parquet` under `output_dir`.

```yaml
- name: hm-wave-1
  type: history_matching
  warm_up:
    type: lhs
    options:
      n: 200
      include_prior_mean_anchor: true      # v0.27 P11 — replaces LHS row 0 with prior means
  emulator:
    type: gp_per_moment                    # only shipped emulator type in v0.31
    options: {}
  implausibility:
    type: max                              # max | second_max
    cutoff: 3.0                            # Pukelsheim's 3-σ rule
    include_prior_terms: true              # add ((θ−μ)/σ)² virtual moments
  moments_included: []                     # empty = all moments; list to include subset
  nroy_grid_size: 8192                     # Sobol dense-grid size for the NROY prune
  output_dir: null                         # default: <workspace>/history_matching/
```

Rejected at construction: non-`moment_set` metric (TypeError),
`scalarize != none` on the metric (ValueError). Fall-through to
CSV if pyarrow/fastparquet isn't installed — the artifact always
lands somewhere.

## Stop criteria (recursive)

| `type`        | Options                                                              | Notes                                        |
|---------------|----------------------------------------------------------------------|----------------------------------------------|
| `max_iter`    | `n: int`                                                             | Stop after N sequential iterations.          |
| `epsilon`     | `epsilon: float`, `target: float = 0.0`, `objective_index: int = 0`  | Stop when `|best − target| < epsilon`.       |
| `plateau`     | `tol: float`, `window: int = 5`, `objective_index: int = 0`          | Stop on spread-of-best < tol across window.  |
| `hypervolume` | `ref_point: [float, float]`, `tol: float = 1e-3`, `patience: int = 3`| 2-D Pareto-HV stagnation (multi-obj).        |
| `any`         | `criteria: [<StopConfig>, ...]`                                      | Logical OR.                                  |
| `all`         | `criteria: [<StopConfig>, ...]`                                      | Logical AND.                                 |

## Jinja2 templating

Anywhere in the YAML you can use:

- `{{ env.<NAME> }}` — environment variables
- `{{ now('%Y%m%d') }}` — current UTC time, strftime-formatted

```yaml
workspace: /lcrc/.../runs/dfw-{{ now('%Y%m%d-%H%M%S') }}
simulator:
  options:
    binary: "{{ env.POLARIS_BIN }}"
```
