# 03 · Multi-objective BO with qLogEHVI

In real POLARIS calibration you often have multiple metrics that fight
each other — e.g. link-volume RMSE vs. mode-share KL. polarisopt handles
multi-objective from day one via BoTorch's qLogEHVI (q-Expected Log
Hypervolume Improvement).

This tutorial uses the [DTLZ2](https://pymoo.org/problems/many/dtlz.html#DTLZ2)
test problem reduced to 2 objectives so we don't need POLARIS.

## 1. Custom 2-objective mock

polarisopt's built-in benchmarks are all single-output. Drop a small
custom simulator into your project (or write a `MultiBenchmark` plugin
once and reuse it everywhere). For the tutorial we'll just use two
overlapping copies of the `quadratic` benchmark via the trick of
running two phases and stitching the results.

A cleaner approach in production: write a plugin Simulator. See
[Tutorial 06 · Writing a plugin](06-write-a-plugin.md).

For now, register an ad-hoc 2-objective mock at the top of your script:

```python
# multi_objective_demo.py
from polarisopt.simulator.benchmarks import BENCHMARKS
import numpy as np

def two_quads(x: np.ndarray) -> float:
    # Hack: return concatenated for storage; metric splits later.
    raise NotImplementedError  # Real path: write a Simulator subclass.
```

For the tutorial we'll instead lean on the **identity metric with two
keys** and a custom Simulator that writes both outputs. The full plugin
is left as the exercise in Tutorial 06.

## 2. The qLogEHVI configuration

The interesting part is the YAML — even if you stub out the simulator
side, this is what a multi-objective phase looks like:

```yaml
phases:
  - name: pareto-front
    type: sequential
    warm_up:
      type: lhs
      options:
        n: 16
    generator:
      type: acquisition
      options:
        surrogate:   { type: gp,    options: {} }
        acquisition:
          type: qehvi
          options:
            ref_point: [10.0, 10.0]   # worst-case in user-space
            mc_samples: 128
    batch_size: 4
    minimize: true
    stop:
      type: any
      criteria:
        - { type: max_iter,     options: { n: 25 } }
        - { type: hypervolume,  options: { ref_point: [10.0, 10.0],
                                            tol: 1e-3, patience: 5 } }
```

Key things to know:

- **`ref_point`** must be worse than every observed objective. For
  minimization (default), "worse" means **larger**. A good rule of thumb
  is the max of each objective in the warm-up + a safety margin.
- **`hypervolume` stop** terminates when the Pareto-front hypervolume
  stops growing for `patience` consecutive iterations.
- **Multi-output GP** — polarisopt's `GPSurrogate` wraps one
  `SingleTaskGP` per objective in BoTorch's `ModelListGP` automatically.

## 3. Reading the Pareto front

After the run:

```python
import numpy as np
from polarisopt.samples.store import SampleStore
from polarisopt.utils.paths import workspace_layout
from polarisopt.config import load_study_config
from polarisopt.stop.hypervolume import _pareto_mask  # internal helper

cfg = load_study_config("study.yaml")
store = SampleStore.open(workspace_layout(cfg.workspace)["db"], cfg.name)
df = store.to_dataframe()
df = df[df["status"] == "finished"]

Y = np.stack([np.asarray(m) for m in df["metric"]])
mask = _pareto_mask(Y, minimize=True)
front = Y[mask]
print(f"Pareto front: {front.shape[0]} points")
```

Plot:

```python
import matplotlib.pyplot as plt
plt.scatter(Y[:, 0], Y[:, 1], c="lightgray", label="dominated")
plt.scatter(front[:, 0], front[:, 1], c="red", label="Pareto front")
plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.legend(); plt.grid(); plt.show()
```

## 4. Scalarization as an alternative

If you don't actually want a Pareto front — you just have multiple
objectives and want a single optimum — combine them into a scalar before
optimization. polarisopt v0.2 lets you do this with a custom Metric:

```python
from polarisopt.metrics.base import Metric, metric_registry
import numpy as np

@metric_registry.register("weighted_sum")
class WeightedSum(Metric):
    def __init__(self, keys, weights):
        self._keys = list(keys); self._w = np.asarray(weights, dtype=float)
    @property
    def n_objectives(self): return 1
    def compute(self, output):
        vec = np.array([output[k] for k in self._keys])
        return np.array([float(np.dot(self._w, vec))])
```

Reference it from YAML:

```yaml
metric:
  type: weighted_sum
  options:
    keys: [rmse_volume, kl_mode]
    weights: [1.0, 0.5]
```

Now the study reverts to single-objective qLogEI — usually cheaper than
qLogEHVI for the same evaluation budget.

## See also

- [qLogEHVI API](../reference/api/acquisition/qehvi.md)
- [Concept: Multi-objective](../concepts/multi-objective.md)
