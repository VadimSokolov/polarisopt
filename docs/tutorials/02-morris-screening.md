# 02 · Morris screening for sensitivity analysis

Before sinking compute into Bayesian optimization, you usually want to
**rank** parameters by influence. The Morris method (elementary effects)
does this with one static design — no surrogate, no iteration loop.

## When to use Morris

- You have 10+ candidate parameters and aren't sure which matter.
- You want to focus the BO loop on the top-K influential subset.
- You're publishing a paper and need a sensitivity analysis table.

For 2–4 parameters that you already know matter, skip Morris and go
straight to BO.

## 1. Write the YAML

Save as `morris.yaml`:

```yaml
name: morris-rosenbrock
workspace: /tmp/morris-demo
seed: 7

simulator:
  type: mock
  options:
    function: rosenbrock

runner:
  type: local
  options: {}

parameters:
  inline:
    - { name: x1, file: dummy.json, min: -2.0, max: 2.0 }
    - { name: x2, file: dummy.json, min: -2.0, max: 2.0 }
    - { name: x3, file: dummy.json, min: -2.0, max: 2.0 }
    - { name: x4, file: dummy.json, min: -2.0, max: 2.0 }

metric:
  type: identity
  options: { keys: value }

phases:
  - name: morris
    type: static
    design:
      type: morris
      options:
        n_trajectories: 10
        num_levels: 4
```

SALib's Morris emits ``N·(d+1)`` rows. Here that's ``10·(4+1) = 50``
evaluations.

## 2. Run

```bash
polarisopt run morris.yaml
polarisopt status morris.yaml
```

## 3. Compute the elementary effects

Morris doesn't ship its analysis step in polarisopt v0.2 — yet. For now,
use SALib directly against the SampleStore:

```python
import numpy as np
from SALib.analyze import morris as morris_analyze
from polarisopt.config import load_study_config
from polarisopt.samples.store import SampleStore
from polarisopt.utils.paths import workspace_layout

cfg = load_study_config("morris.yaml")
layout = workspace_layout(cfg.workspace)
store = SampleStore.open(layout["db"], cfg.name)
df = store.to_dataframe()
df = df[df["status"] == "finished"]

X = np.stack(df["inputs"].apply(np.asarray).to_list())
Y = np.array([m[0] for m in df["metric"]])

problem = {
    "num_vars": 4,
    "names": ["x1", "x2", "x3", "x4"],
    "bounds": [[-2, 2]] * 4,
}
Si = morris_analyze.analyze(problem, X, Y, conf_level=0.95, num_levels=4)
print(Si.to_df())
```

You'll see something like:

|     |   mu  | mu_star |  sigma | mu_star_conf |
|-----|------:|--------:|-------:|-------------:|
| x1  |  3.1  |    8.4  |  10.2  |        2.1   |
| x2  |  4.8  |   12.1  |  14.3  |        3.0   |
| x3  |  0.2  |    1.1  |   1.4  |        0.3   |
| x4  |  0.1  |    0.7  |   0.9  |        0.2   |

``mu_star`` is the canonical Morris importance metric — bigger is more
influential. In this example x1 and x2 dominate (rosenbrock's pinch
parameters); x3 and x4 are essentially decoupled.

## 4. Use the result to focus a BO study

Drop low-importance parameters from the next study YAML by removing
them from ``parameters.inline``, then run a sequential phase with a
much smaller search space:

```yaml
parameters:
  inline:
    - { name: x1, file: dummy.json, min: -2.0, max: 2.0 }
    - { name: x2, file: dummy.json, min: -2.0, max: 2.0 }

phases:
  - name: bo
    type: sequential
    # ... qLogEI on a 2D space, way fewer evaluations
```

## See also

- [Morris design API](../reference/api/design/morris.md)
- SALib docs: https://salib.readthedocs.io/en/latest/api.html#method-of-morris
