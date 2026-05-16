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
```

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
    aggregation: sum_abs     # sum_abs | rmse | vector
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
