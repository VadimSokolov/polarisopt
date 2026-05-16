# Multi-objective optimization

Most POLARIS calibrations balance multiple metrics — e.g. link-volume
RMSE *and* mode-share KL divergence. polarisopt supports two ways to
handle this.

## Option 1 — Pareto: optimize for a *front* of trade-offs

Use a vector-valued Metric and the [`qehvi`](../reference/api/acquisition/qehvi.md)
acquisition. The BO loop maintains a Pareto front: each iteration the
acquisition picks points that maximize the *expected hypervolume
improvement* — the expected expansion of the dominated region beyond
some worst-case reference point.

```yaml
metric:
  type: link_moe_and_mode_share   # a custom 2-objective Metric
  options: { ... }

phases:
  - name: pareto
    type: sequential
    warm_up: { type: lhs, options: { n: 16 } }
    generator:
      type: acquisition
      options:
        surrogate:  { type: gp, options: {} }
        acquisition:
          type: qehvi
          options:
            ref_point: [10.0, 5.0]   # in user-facing objective units
            mc_samples: 128
    batch_size: 4
    minimize: true
    stop:
      type: hypervolume
      options:
        ref_point: [10.0, 5.0]
        tol: 1e-3
        patience: 5
```

Output: a set of non-dominated points on the Pareto front. The user
picks one based on the application's relative weighting of objectives.

### Reference point

The `ref_point` is a **worst-case** point in objective space — every
realistic observation should dominate it. For minimization that means
larger than any expected value:

```python
import numpy as np
warm_max = Y_warmup.max(axis=0)
ref_point = warm_max * 1.1   # 10% buffer past the worst warm-up sample
```

A poorly chosen reference point either inflates hypervolume (too far)
or excludes useful trade-offs (too close).

## Option 2 — Scalarization: collapse to a single number first

If you don't need a front — you just want *one* optimum that balances
the metrics in a user-chosen way — write a single-output Metric that
combines the underlying outputs:

```python
from polarisopt.metrics.base import Metric, metric_registry
import numpy as np

@metric_registry.register("weighted_sum")
class WeightedSum(Metric):
    """Weighted sum of multiple simulator outputs."""

    def __init__(self, keys: list[str], weights: list[float]) -> None:
        assert len(keys) == len(weights)
        self._keys = list(keys)
        self._w = np.asarray(weights, dtype=float)

    @property
    def n_objectives(self) -> int:
        return 1

    def compute(self, output: dict) -> np.ndarray:
        vec = np.array([float(output[k]) for k in self._keys])
        return np.array([float(np.dot(self._w, vec))])
```

```yaml
metric:
  type: weighted_sum
  options:
    keys:    [rmse_volume, kl_mode_share]
    weights: [1.0, 0.5]
```

Now use the cheaper [`qei`](../reference/api/acquisition/qei.md) instead
of qLogEHVI — the problem is single-objective from the optimizer's
point of view.

Other scalarizations: Chebyshev (max instead of weighted sum),
augmented Chebyshev, Tchebycheff scalarization. All are <30 LOC and
plug in as Metric subclasses.

## Which to choose

| You want… | Use |
|---|---|
| A Pareto front to explore trade-offs | Pareto (qLogEHVI) |
| One optimum, weights pre-decided | Scalarization (qLogEI) |
| One optimum, weights tunable post-hoc | Pareto, then pick a point on the front |
| ≥ 4 objectives | Scalarization (qLogEHVI hypervolume gets expensive in high m) |
| 2–3 objectives, clear trade-offs | Pareto |

## How the GP handles vector outputs

[`GPSurrogate`](../reference/api/surrogates/gp.md) auto-detects ``m`` at
fit time:

- ``m = 1`` → one `SingleTaskGP`.
- ``m > 1`` → ``m`` independent `SingleTaskGP`s wrapped in
  `ModelListGP`. Each output gets its own kernel hyperparameters.

This is the "independent outputs" multi-task assumption. For correlated
outputs you'd want a multi-task GP — a future surrogate plugin.

## Reading a Pareto front from the SampleStore

```python
import numpy as np
from polarisopt.samples.store import SampleStore
from polarisopt.utils.paths import workspace_layout
from polarisopt.config import load_study_config

cfg = load_study_config("study.yaml")
store = SampleStore.open(workspace_layout(cfg.workspace)["db"], cfg.name)
df = store.to_dataframe()
df = df[df["status"] == "finished"]

Y = np.stack([np.asarray(m) for m in df["metric"]])

# Pareto front via elementwise dominance
def pareto_mask(Y, minimize=True):
    Y = Y if minimize else -Y
    n = Y.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]: continue
        diff = Y - Y[i]
        dominated = (diff <= 0).all(axis=1) & (diff < 0).any(axis=1)
        keep[i] = not dominated.any()
    return keep

front = Y[pareto_mask(Y)]
print(f"{len(front)} non-dominated points")
```

(There's also a private helper `polarisopt.stop.hypervolume._pareto_mask`
that does the same thing — exposed as a public utility in a future
version.)

## See also

- [Tutorial 03 · Multi-objective](../tutorials/03-multi-objective.md)
- [Bayesian optimization concepts](bayesian-optimization.md)
