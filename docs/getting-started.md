# Getting started

## Install

`polarisopt` requires Python 3.11+.

```bash
# Core
pip install polarisopt

# Bayesian optimization (BoTorch + GPyTorch + PyTorch)
pip install 'polarisopt[bo]'

# Argonne integrations (Globus via polaris-studio)
pip install 'polarisopt[anl]'

# Development
pip install 'polarisopt[dev]'
```

For development from a checkout:

```bash
git clone https://github.com/anl-polaris/polaris-hpc.git
cd polaris-hpc
pip install -e '.[dev,bo]'
```

## Verify

```bash
polarisopt --version
polarisopt --help
```

## Run a benchmark study (no POLARIS required)

A minimal study using the `mock` simulator and the Branin test function:

```yaml
# branin.yaml
name: branin-demo
workspace: /tmp/branin-demo
seed: 42

simulator:
  type: mock
  options: { function: branin }

runner:
  type: local
  options: {}

parameters:
  inline:
    - { name: x1, file: dummy.json, min: -5.0, max: 10.0 }
    - { name: x2, file: dummy.json, min:  0.0, max: 15.0 }

metric:
  type: identity
  options: { keys: value }

phases:
  - name: bo
    type: sequential
    warm_up: { type: lhs, options: { n: 8 } }
    generator:
      type: acquisition
      options:
        surrogate: { type: gp, options: {} }
        acquisition: { type: qei, options: { mc_samples: 64 } }
    batch_size: 2
    stop:
      type: max_iter
      options: { n: 6 }
```

```bash
polarisopt run branin.yaml
polarisopt status branin.yaml
```

The SampleStore lives at `/tmp/branin-demo/polarisopt.db`. Open it from a notebook:

```python
from polarisopt.samples.store import SampleStore
from polarisopt.utils.paths import workspace_layout

layout = workspace_layout("/tmp/branin-demo")
store = SampleStore.open(layout["db"], "branin-demo")
df = store.to_dataframe()
df.head()
```

## Use with POLARIS

See [Study YAML reference](yaml-reference.md) for the full schema and the
top-level `index.md` for a complete DFW-style study. The key pieces:

- ``simulator.type: polaris`` and point ``model_source`` at your model
  directory (e.g. ``/lcrc/project/POLARIS/.../DFW_2050_20251028``).
- ``runner.type: slurm`` with the sbatch resources for your cluster.
- ``parameters.source`` pointing at a YAML/JSON list of calibration
  parameter definitions (one record per variable; each record names the
  POLARIS JSON file it lives in).
- ``metric.type: link_moe`` for a single-objective link-volume RMSE, or
  ``choice_share`` for categorical share matching.
